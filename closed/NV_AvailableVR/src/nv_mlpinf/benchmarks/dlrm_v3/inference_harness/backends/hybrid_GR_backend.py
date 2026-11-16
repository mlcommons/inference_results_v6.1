"""
Hybrid GR (Generative Recommender) backend for DLRM inference.

Implements a custom backend that combines NVE embeddings for large tables
with optimized dense model inference, supporting distributed embedding lookups.
"""

from inference_harness.backends.base import DLRMBackend
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from typing import Dict, Tuple, List, Set, Optional
from dataclasses import dataclass, field
from generative_recommenders.modules.stu import STULayerConfig, STUStack
from inference_harness.backends.model.dlrm_hstu_custom import DlrmHSTUCustom
from generative_recommenders.dlrm_v3.inference.inference_modules import set_is_inference
from .ops import NVEEmbeddingCollection, NVEEmbeddingCollectionConfig
from inference_harness.backends.model.STU_custom import STULayerCustom
from torchrec.modules.embedding_modules import EmbeddingCollection
from generative_recommenders.modules.stu import STULayerConfig
from generative_recommenders.dlrm_v3.inference.inference_modules import move_sparse_output_to_device
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from generative_recommenders.dlrm_v3.datasets.dataset import Samples

from typing import Union
from generative_recommenders.dlrm_v3.checkpoint import (
    load_nonsparse_checkpoint,
)
import nvtx
from torchrec.distributed.types import ShardedTensor
import torch
import pynve.torch.nve_layers as nve_layers
import pynve.nve as nve

from generative_recommenders.modules.dlrm_hstu import (
    DlrmHSTUConfig,
    SequenceEmbedding,
)

import logging
# Suppress verbose logs from generative_recommenders (e.g., "Initialize HSTU module with configs...")
logging.getLogger("generative_recommenders.modules.dlrm_hstu").setLevel(logging.WARNING)
logging.getLogger("generative_recommenders.dlrm_v3.checkpoint").setLevel(logging.WARNING)


def is_sparse_key(k: str, v: torch.Tensor) -> bool:
    return isinstance(v, ShardedTensor) or "embedding_collection" in k


from torch.distributed.checkpoint.stateful import Stateful


class SparseState(Stateful):
    def __init__(self, model: torch.nn.Module, sparse_tensor_keys: Set[str]) -> None:
        self.model = model
        self.sparse_tensor_keys = sparse_tensor_keys

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out_dict: Dict[str, torch.Tensor] = {}
        is_sharded_tensor: Optional[bool] = None
        for k, v in self.model.state_dict().items():
            if k in self.sparse_tensor_keys:
                if is_sharded_tensor is None:
                    is_sharded_tensor = isinstance(v, ShardedTensor)
                assert is_sharded_tensor == isinstance(v, ShardedTensor)
                out_dict[k] = v
        return out_dict

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        assert not incompatible_keys.unexpected_keys


class SparseStateRankRest(Stateful):
    def __init__(self, model: torch.nn.Module, sparse_tensor_keys: Set[str]) -> None:
        self.model = model
        self.sparse_tensor_keys = sparse_tensor_keys

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out_dict: Dict[str, torch.Tensor] = {}
        is_sharded_tensor: Optional[bool] = None
        for k, v in self.model.state_dict().items():
            if k in self.sparse_tensor_keys:
                if is_sharded_tensor is None:
                    is_sharded_tensor = isinstance(v, ShardedTensor)
                assert is_sharded_tensor == isinstance(v, ShardedTensor)
                out_dict[k] = v
        return out_dict

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        assert not incompatible_keys.unexpected_keys


@dataclass
class HybridGRBackendConfig:
    batch_size: int = 1
    perf_mode: str = "performance"
    use_custom_stu: bool = True
    use_nve: bool = True
    use_multi_gpu: bool = False
    use_mpi: bool = False  # Use MPI-based multi-GPU when both use_multi_gpu and use_mpi are True
    gpu_cache_size_in_gigabytes: int = 10
    device_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    nve_memblock: nve.NVLMemBlock = None


@dataclass
class CustomJaggedTensor:
    values: torch.Tensor
    lengths: torch.Tensor
    offsets: torch.Tensor
    max_length: int
    embeddings: torch.Tensor


class HybridGRBackend(DLRMBackend):
    def __init__(self, model_name: str, device: torch.device = torch.device("cuda:0")):
        super().__init__(model_name=model_name)
        self.backend = "GR"
        self.model_impl: DlrmHSTUCustom = None
        self.backend_config: HybridGRBackendConfig = None
        self.device: torch.device = device

    def initialize(self, hstu_config: DlrmHSTUConfig, embedding_table_config: Dict[str, EmbeddingConfig], backend_config: HybridGRBackendConfig):
        self.hstu_config: DlrmHSTUConfig = hstu_config
        self.embedding_table_config: Dict[str, EmbeddingConfig] = embedding_table_config
        self.backend_config: HybridGRBackendConfig = backend_config
        self.perf_mode = self.backend_config.perf_mode
        self.is_inference = True if self.backend_config.perf_mode == "performance" else False
        torch.cuda.set_device(self.device)

        self.dummy_tensor = torch.ones(1, self.backend_config.batch_size * 2048, dtype=torch.bfloat16)
        # Dummy labels and weights for warmup in accuracy mode (float32 to match _serialize_result expectations)
        self.dummy_labels = torch.ones(1, self.backend_config.batch_size * 2048, dtype=torch.float32)
        self.dummy_weights = torch.ones(1, self.backend_config.batch_size * 2048, dtype=torch.float32)

        set_is_inference(is_inference=self.is_inference)
        model_dense = DlrmHSTUCustom(
            hstu_configs=self.hstu_config,
            embedding_tables=self.embedding_table_config,
            is_dense=True,
            is_inference=self.is_inference,
        )

        # Replace STU layers with custom implementation if requested
        if self.backend_config.use_custom_stu:
            stu_module = STUStack(
                stu_list=[
                    STULayerCustom(
                        config=STULayerConfig(
                            embedding_dim=hstu_config.hstu_transducer_embedding_dim,
                            num_heads=hstu_config.hstu_num_heads,
                            hidden_dim=hstu_config.hstu_attn_linear_dim,
                            attention_dim=hstu_config.hstu_attn_qk_dim,
                            output_dropout_ratio=hstu_config.hstu_linear_dropout_rate,
                            use_group_norm=hstu_config.hstu_group_norm,
                            causal=True,
                            target_aware=True,
                            max_attn_len=None,
                            attn_alpha=None,
                            recompute_normed_x=True,
                            recompute_uvqk=True,
                            recompute_y=True,
                            sort_by_length=True,
                            contextual_seq_len=0,
                        ),
                        is_inference=self.is_inference,
                        dtype=torch.bfloat16,  # bf16 is the default dtype for the model
                        device=self.device
                    )
                    for _ in range(hstu_config.hstu_attn_num_layers)
                ],
                is_inference=self.is_inference,

            )
            model_dense._hstu_transducer._stu_module = stu_module
        # Move model to device and set data type
        model_dense.eval()
        model_dense.recursive_setattr("_use_triton_cc", False)
        model_dense = model_dense.to(self.device).to(torch.bfloat16)
        model_dense.set_training_dtype(torch.bfloat16)

        # Materialize embeddings if sparse
        model_sparse = None
        if self.backend_config.use_nve:

            nve_config = []
            for table_name in self.embedding_table_config:
                if table_name == "item_id":
                    # Movie ID uses NVL/MPI with cache or NoCache depending on multi-GPU setting
                    if self.backend_config.use_multi_gpu:
                        # Use MPI memblock if use_mpi is enabled, otherwise use the provided memblock
                        # memblock_to_use = mpi_memblock if self.backend_config.use_mpi else self.backend_config.nve_memblock
                        memblock_to_use = self.backend_config.nve_memblock
                        nve_config.append(
                            NVEEmbeddingCollectionConfig(
                                cache_type=nve_layers.CacheType.LinearUVM,
                                gpu_cache_size_in_bytes=self.backend_config.gpu_cache_size_in_gigabytes * 1024 * 1024 * 1024,
                                memblock=memblock_to_use,
                                device=self.device,
                            )
                        )
                    else:
                        # Single GPU, no cache
                        nve_config.append(
                            NVEEmbeddingCollectionConfig(
                                cache_type=nve_layers.CacheType.NoCache,
                                gpu_cache_size_in_bytes=0,
                                memblock=None,
                                device=self.device,
                            )
                        )
                elif table_name == "user_id" or table_name == "item_category_id":
                    # User ID uses NoCache (all on GPU)
                    nve_config.append(
                        NVEEmbeddingCollectionConfig(
                            cache_type=nve_layers.CacheType.NoCache,
                            gpu_cache_size_in_bytes=0,
                            memblock=None,
                            device=self.device,
                        )
                    )
                else:
                    raise ValueError(f"Unknown table name: {table_name}")

            embedding_collection = NVEEmbeddingCollection(list(self.embedding_table_config.values()), nve_config)
            model_dense._embedding_collection = embedding_collection
        else:
            model_sparse = DlrmHSTUCustom(
                hstu_configs=self.hstu_config,
                embedding_tables=self.embedding_table_config,
                is_dense=False
            )
            for _, module in model_sparse.named_modules():
                if isinstance(module, EmbeddingCollection):
                    module.to_empty(device="cpu")
            model_sparse.eval()
            model_dense._embedding_collection = model_sparse._embedding_collection

        self.model_impl = model_dense

    def load_model_sparse(self, checkpoint_path: str, main_rank: bool = False):
        sparse_tensor_keys = {
            k for k, v in self.model_impl.state_dict().items() if is_sparse_key(k, v)
        }
        if not main_rank:
            sparse_tensor_keys.remove('_embedding_collection.embeddings.item_id.weight')

        sparse_dict = {"sparse_dict": SparseState(self.model_impl, sparse_tensor_keys)}
        torch.distributed.checkpoint.load(
            sparse_dict,
            storage_reader=torch.distributed.checkpoint.FileSystemReader(checkpoint_path + "/sparse/"),
        )
        self.model_impl.state_dict()['_embedding_collection.embeddings.item_category_id.weight'] == self.model_impl._embedding_collection.embeddings.item_category_id.weight

    def load_model_dense(self, checkpoint_path: str):
        load_nonsparse_checkpoint(model=self.model_impl, device=self.device, optimizer=None, path=checkpoint_path)

    # WIP feature, reduce PCIe data transfer + optimize CPU here

    def optimized_embedding_lookup(self, uih_features: Dict[str, CustomJaggedTensor], candidates_features: Dict[str, CustomJaggedTensor]):
        tensor_dict_uih = uih_features
        tensor_dict_candidates = candidates_features

        with nvtx.annotate(f"hybrid_GR_backend - optimized_embedding_lookup - embedding lookup", color="orange"):
            self.model_impl._embedding_collection(tensor_dict_uih)
            self.model_impl._embedding_collection(tensor_dict_candidates)
            seq_embeddings_dict = self._pack_sequence_embeddings(tensor_dict_uih, tensor_dict_candidates)

        with nvtx.annotate(f"hybrid_GR_backend - optimized_embedding_lookup - payload features", color="orange"):
            max_uih_len = tensor_dict_uih["item_id"].max_length
            uih_seq_lengths = tensor_dict_uih["item_id"].lengths
            max_num_candidates = tensor_dict_candidates["item_candidate_id"].max_length
            num_candidates = tensor_dict_candidates["item_candidate_id"].lengths

            payload_features = {
                "uih_offsets": tensor_dict_uih["item_id"].offsets,
                "candidate_offsets": tensor_dict_candidates["item_candidate_id"].offsets,
                # need
                "action_timestamp": torch.ones(tensor_dict_uih["item_id"].offsets[-1], device=self.device, dtype=torch.int64),
                "item_query_time": torch.ones(tensor_dict_candidates["item_candidate_id"].offsets[-1], device=self.device, dtype=torch.int64),
                "item_weights": tensor_dict_uih["item_weights"].values,
                "item_action_weights": tensor_dict_candidates["item_action_weights"].values,
            }
        return seq_embeddings_dict, payload_features, max_uih_len, uih_seq_lengths, max_num_candidates, num_candidates

    # WIP feature, reduce PCIe data transfer + optimize CPU here
    def _pack_sequence_embeddings(self, tensor_dict_uih: Dict[str, CustomJaggedTensor], tensor_dict_candidates: Dict[str, CustomJaggedTensor]) -> Dict[str, SequenceEmbedding]:
        with nvtx.annotate(f"hybrid_GR_backend - optimized_embedding_lookup - pack sequence embeddings", color="orange"):
            sequence_embeddings = {}
            sequence_embeddings["user_id"] = SequenceEmbedding(
                lengths=tensor_dict_uih["user_id"].lengths,
                embedding=tensor_dict_uih["user_id"].embeddings,
            )
            sequence_embeddings["item_id"] = SequenceEmbedding(
                lengths=tensor_dict_uih["item_id"].lengths,
                embedding=tensor_dict_uih["item_id"].embeddings,
            )
            sequence_embeddings["item_category_id"] = SequenceEmbedding(
                lengths=tensor_dict_uih["item_category_id"].lengths,
                embedding=tensor_dict_uih["item_category_id"].embeddings,
            )
            sequence_embeddings["item_candidate_id"] = SequenceEmbedding(
                lengths=tensor_dict_candidates["item_candidate_id"].lengths,
                embedding=tensor_dict_candidates["item_candidate_id"].embeddings,
            )
            sequence_embeddings["item_candidate_category_id"] = SequenceEmbedding(
                lengths=tensor_dict_candidates["item_candidate_category_id"].lengths,
                embedding=tensor_dict_candidates["item_candidate_category_id"].embeddings,
            )
        return sequence_embeddings

    def embedding_lookup(self, uih_features: KeyedJaggedTensor, candidates_features: KeyedJaggedTensor):
        with nvtx.annotate(f"hybrid_GR_backend - embedding_lookup - transfer to device", color="orange"):
            if self.backend_config.use_nve:
                uih_features = uih_features.to(self.device).to(torch.int64)
                candidates_features = candidates_features.to(self.device).to(torch.int64)
            else:
                # cpu look up, base impl
                uih_features = uih_features.to(torch.int64)
                candidates_features = candidates_features.to(torch.int64)
        with nvtx.annotate(f"hybrid_GR_backend - embedding_lookup - embedding lookup", color="green"):
            seq_embeddings, payload_features, max_uih_len, uih_seq_lengths, max_num_candidates, num_candidates = \
                self.model_impl.preprocess(uih_features=uih_features, candidates_features=candidates_features)
        return seq_embeddings, payload_features, max_uih_len, uih_seq_lengths, max_num_candidates, num_candidates

    def predict(self, Samples: Union[List[Dict[str, CustomJaggedTensor]], Samples]):
        # Use original embedding lookup path
        uih_features = Samples.uih_features_kjt
        candidates_features = Samples.candidates_features_kjt
        seq_embeddings, payload_features, max_uih_len, uih_seq_lengths, max_num_candidates, num_candidates = self.embedding_lookup(uih_features, candidates_features)

        if self.backend_config.use_nve:
            with nvtx.annotate(f"hybrid_GR_backend - converting sequence embeddings to bfloat16", color="blue"):
                seq_embeddings_bf16 = {
                    k: SequenceEmbedding(
                        lengths=seq_embeddings[k].lengths,
                        embedding=seq_embeddings[k].embedding.to(torch.bfloat16),
                    )
                    for k in seq_embeddings.keys()
                }
            with nvtx.annotate(f"hybrid_GR_backend - main forward", color="yellow"):
                out = self.model_impl.main_forward(
                    seq_embeddings_bf16, payload_features, max_uih_len,
                    uih_seq_lengths, max_num_candidates, num_candidates
                )
        else:
            with nvtx.annotate(f"hybrid_GR_backend - moving to device", color="blue"):
                seq_embeddings_bf16, payload_features_bf16, uih_seq_lengths_bf16, num_candidates_bf16 = \
                    move_sparse_output_to_device(
                        seq_embeddings=seq_embeddings,
                        payload_features=payload_features,
                        uih_seq_lengths=uih_seq_lengths,
                        num_candidates=num_candidates,
                        device=self.device,
                    )
            with nvtx.annotate(f"hybrid_GR_backend - main forward", color="yellow"):
                out = self.model_impl.main_forward(
                    seq_embeddings_bf16, payload_features_bf16, max_uih_len,
                    uih_seq_lengths_bf16, max_num_candidates, num_candidates_bf16
                )
        return out

    def predict_dummy(self, Samples: Union[List[Dict[str, CustomJaggedTensor]], Samples]):
        return self.dummy_tensor, self.dummy_labels, self.dummy_weights
