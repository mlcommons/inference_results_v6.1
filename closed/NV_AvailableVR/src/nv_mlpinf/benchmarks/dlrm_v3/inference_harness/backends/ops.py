"""
Custom operators and embedding collections for DLRM inference.

Provides NVE (NVIDIA Embedding)-based embedding collections with GPU caching
and optimized lookup operations for large-scale recommendation models.
"""

from typing import List, Dict
import torch
from torchrec import EmbeddingBagCollectionInterface
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor, JaggedTensor
import pynve.torch.nve_layers as nve_layers
import pynve.nve as nve
from torchrec.modules.embedding_configs import DataType
from dataclasses import dataclass
import nvtx as nvtx


def torchrec_data_type_to_torch_data_type(data_type: DataType):
    """
    Convert TorchRec DataType to PyTorch dtype.

    Args:
        data_type: TorchRec DataType enum value.

    Returns:
        torch.dtype: Corresponding PyTorch data type.

    Raises:
        ValueError: If data type is not supported.
    """
    if data_type == DataType.FP32:
        return torch.float32
    elif data_type == DataType.FP16:
        return torch.float16
    elif data_type == DataType.BF16:
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid data type: {data_type}")


def torch_data_type_to_nve_data_type(data_type: torch.dtype):
    """
    Convert PyTorch dtype to NVE DataType for memblock creation.

    Args:
        data_type: PyTorch data type.

    Returns:
        nve.DataType_t: Corresponding NVE data type.

    Raises:
        ValueError: If data type is not supported by NVE.
    """
    if data_type == torch.float32:
        return nve.DataType_t.Float32
    elif data_type == torch.float16:
        return nve.DataType_t.Float16
    else:
        raise ValueError(f"Unsupported NVE data type: {data_type}")


@dataclass
class NVEEmbeddingCollectionConfig:
    """
    Configuration for NVE embedding collection.

    Attributes:
        cache_type: Type of GPU cache to use (e.g., NoCache, LRU).
        gpu_cache_size_in_bytes: Size of GPU cache in bytes (default: 1GB).
        memblock: NVE memory block for embedding storage.
        device: CUDA device for embedding operations.
    """
    cache_type: nve_layers.CacheType = nve_layers.CacheType.NoCache
    gpu_cache_size_in_bytes: int = 1024 * 1024 * 1024  # 1GB default
    memblock: nve.MemBlock = None
    device: torch.device = None


class NVEEmbeddingCollection(EmbeddingBagCollectionInterface):
    """
    NVE-based embedding collection for large-scale recommendation models.

    Implements TorchRec's EmbeddingBagCollectionInterface using NVIDIA's
    NVE (NVIDIA Embedding) backend for optimized embedding lookups with
    GPU caching support for large embedding tables.

    Attributes:
        embeddings: Dictionary of NVEmbedding modules, one per table.
        _embedding_configs: List of embedding table configurations.
        _nve_config: List of NVE-specific configurations.
        _feature_names: List of feature names for each embedding table.
    """

    def __init__(self,
                 table_configs: List[EmbeddingConfig],
                 nve_config: List[NVEEmbeddingCollectionConfig]
                 ):
        """
        Initialize NVE embedding collection.

        Args:
            table_configs: List of embedding table configurations from TorchRec.
            nve_config: List of NVE-specific configurations (one per table).

        Raises:
            ValueError: If number of table configs doesn't match number of NVE configs.
        """
        super().__init__()
        assert len(table_configs) == len(nve_config), "Number of table configs and config must be the same"
        self.embeddings: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self._embedding_configs = table_configs
        self._nve_config = nve_config
        self._lengths_per_embedding: List[int] = []

        table_names = set()
        for i, embedding_config in enumerate(table_configs):
            if embedding_config.name in table_names:
                raise ValueError(f"Duplicate table name {embedding_config.name}")
            table_names.add(embedding_config.name)
            config = {"kernel_mode": 1, "logging_interval": -1}
            self.embeddings[embedding_config.name] = nve_layers.NVEmbedding(
                num_embeddings=embedding_config.num_embeddings,
                embedding_size=embedding_config.embedding_dim,
                data_type=torchrec_data_type_to_torch_data_type(embedding_config.data_type),
                cache_type=nve_config[i].cache_type,
                gpu_cache_size=nve_config[i].gpu_cache_size_in_bytes,
                memblock=nve_config[i].memblock,
                weight_init=None,
                device=nve_config[i].device,
                config=config
            )

        self._feature_names: List[List[str]] = [table.feature_names for table in table_configs]

    def forward(self, features: Dict[str, List[torch.Tensor]]) -> Dict[str, JaggedTensor]:
        """
        Perform embedding lookups for input features.

        Supports two input formats:
        1. KeyedJaggedTensor: Returns Dict[str, JaggedTensor] with embeddings
        2. Dict[str, CustomJaggedTensor]: Updates each tensor in-place with embeddings

        Args:
            features: Input features as KeyedJaggedTensor or Dict[str, CustomJaggedTensor].

        Returns:
            Dict[str, JaggedTensor]: Dictionary mapping feature names to embedding tensors.
        """
        flat_feature_names: List[str] = []
        for names in self._feature_names:
            flat_feature_names.extend(names)

        # Handle KeyedJaggedTensor input
        if isinstance(features, KeyedJaggedTensor):
            feature_dict = features.to_dict()
            feature_embeddings: Dict[str, JaggedTensor] = {}

            for i, embedding in enumerate(self.embeddings.values()):
                for feature_name in self._feature_names[i]:
                    if feature_name not in feature_dict:
                        continue
                    f = feature_dict[feature_name]
                    with nvtx.annotate(f"embedding_{feature_name}", color="blue"):
                        res = embedding(keys=f.values())
                    feature_embeddings[feature_name] = JaggedTensor(
                        values=res,
                        lengths=f.lengths(),
                    )
            return feature_embeddings
        else:
            # input is Dict[str, List[torch.Tensor]]
            keys = features.keys()
            for i, embedding in enumerate(self.embeddings.values()):
                for feature_name in self._feature_names[i]:
                    if feature_name not in keys:
                        continue
                    with nvtx.annotate(f"embedding_{feature_name}", color="blue"):
                        res = embedding(keys=features[feature_name].values)
                    with nvtx.annotate(f"embedding_{feature_name} - packing", color="blue"):
                        features[feature_name].embeddings = res
            return features

    def embedding_bag_configs(self):
        """
        Get embedding table configurations.

        Returns:
            List[EmbeddingConfig]: List of embedding table configurations.
        """
        return self._embedding_configs

    def is_weighted(self) -> bool:
        """
        Check if embedding collection uses weighted pooling.

        Returns:
            bool: False (NVE embeddings don't use weighted pooling).
        """
        return False
