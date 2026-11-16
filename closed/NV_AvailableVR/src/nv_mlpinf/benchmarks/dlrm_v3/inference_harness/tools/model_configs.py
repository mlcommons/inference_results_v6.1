"""
Model configuration utilities for DLRM-HSTU inference.

Provides configuration factories for HSTU models, embedding tables, backends,
datasets, and communication settings. Supports both production and debug
configurations for flexible testing and deployment.
"""

from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from typing import Dict, Union
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig
from inference_harness.backends.hybrid_GR_backend import HybridGRBackendConfig
from inference_harness.backends.ops import torch_data_type_to_nve_data_type
from inference_harness.mpi_utils import ZMQRequestSenderShardedConfig
from inference_harness.dataset.mlperf_streaming_qsl import DLRMv3StreamingMLPerfDataset
from inference_harness.dataset.streaming_query_sampler import StreamingQuerySamplerRef
import pynve.nve as nve
import torch

# Model configuration constants
HSTU_EMBEDDING_DIM = 512
MOVIE_EMBEDDING_TABLE_SIZE = 500_000_000
USER_EMBEDDING_TABLE_SIZE = 3_000_000


import logging
import sys
logger = logging.getLogger(__name__)


def get_hstu_configs(model: str = "production") -> DlrmHSTUConfig:
    """
    Get HSTU model configuration.

    Args:
        model: Configuration preset ("production" or "debug").

    Returns:
        DlrmHSTUConfig: HSTU model configuration.

    Raises:
        ValueError: If model preset is not recognized.
    """
    if model == "production":
        hstu_config = DlrmHSTUConfig(
            hstu_num_heads=4,
            hstu_attn_linear_dim=128,
            hstu_attn_qk_dim=128,
            hstu_attn_num_layers=5,
            hstu_embedding_table_dim=HSTU_EMBEDDING_DIM,
            hstu_preprocessor_hidden_dim=256,
            hstu_transducer_embedding_dim=512,
            hstu_group_norm=False,
            hstu_input_dropout_ratio=0.2,
            hstu_linear_dropout_rate=0.1,
            causal_multitask_weights=0.2,
        )
        hstu_config.user_embedding_feature_names = [
            "item_id",
            "user_id",
            "item_category_id",
        ]
        hstu_config.item_embedding_feature_names = [
            "item_candidate_id",
            "item_candidate_category_id",
        ]
        hstu_config.uih_post_id_feature_name = "item_id"
        hstu_config.uih_action_time_feature_name = "action_timestamp"
        hstu_config.candidates_querytime_feature_name = "item_query_time"
        hstu_config.candidates_weight_feature_name = "item_action_weights"
        hstu_config.uih_weight_feature_name = "item_weights"
        hstu_config.candidates_watchtime_feature_name = "item_rating"
        hstu_config.action_weights = [1, 2, 4, 8, 16]
        hstu_config.action_embedding_init_std = 5.0
        hstu_config.contextual_feature_to_max_length = {"user_id": 1}
        hstu_config.contextual_feature_to_min_uih_length = {"user_id": 20}
        hstu_config.merge_uih_candidate_feature_mapping = [
            ("item_id", "item_candidate_id"),
            ("item_rating", "item_candidate_rating"),
            ("action_timestamp", "item_query_time"),
            ("item_weights", "item_action_weights"),
            ("dummy_watch_time", "item_dummy_watchtime"),
            ("item_category_id", "item_candidate_category_id"),
        ]
        hstu_config.hstu_uih_feature_names = [
            "user_id",
            "item_id",
            "item_rating",
            "action_timestamp",
            "item_weights",
            "dummy_watch_time",
            "item_category_id",
        ]
        hstu_config.hstu_candidate_feature_names = [
            "item_candidate_id",
            "item_candidate_rating",
            "item_query_time",
            "item_action_weights",
            "item_dummy_watchtime",
            "item_candidate_category_id",
        ]
        hstu_config.max_num_candidates = 32
        hstu_config.max_num_candidates_inference = 2048
        hstu_config.multitask_configs = [
            TaskConfig(
                task_name="rating",
                task_weight=1,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            )
        ]
        return hstu_config
    elif model == "debug":
        hstu_config = get_hstu_configs("production")
        return hstu_config

    else:
        raise NotImplementedError(f"Model {model} not implemented")


def get_embedding_table_config(model: str = "production") -> Dict[str, EmbeddingConfig]:
    """
    Get embedding table configurations.

    Args:
        model: Configuration preset ("production" or "debug").

    Returns:
        Dict[str, EmbeddingConfig]: Dictionary mapping table names to configurations.

    Raises:
        ValueError: If model preset is not recognized.
    """
    if model == "production":
        return {
            "item_id": EmbeddingConfig(
                num_embeddings=1_000_000_000,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="item_id",
                data_type=DataType.FP16,
                feature_names=["item_id", "item_candidate_id"],
            ),
            "item_category_id": EmbeddingConfig(
                num_embeddings=128,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="item_category_id",
                data_type=DataType.FP16,
                weight_init_max=1.0,
                weight_init_min=-1.0,
                feature_names=["item_category_id", "item_candidate_category_id"],
            ),
            "user_id": EmbeddingConfig(
                num_embeddings=10_000_000,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="user_id",
                data_type=DataType.FP16,
                feature_names=["user_id"],
            ),
        }
    elif model == "debug":
        return {
            "item_id": EmbeddingConfig(
                num_embeddings=1_000_000_000,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="item_id",
                data_type=DataType.FP16,
                feature_names=["item_id", "item_candidate_id"],
            ),
            "item_category_id": EmbeddingConfig(
                num_embeddings=128,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="item_category_id",
                data_type=DataType.FP16,
                weight_init_max=1.0,
                weight_init_min=-1.0,
                feature_names=["item_category_id", "item_candidate_category_id"],
            ),
            "user_id": EmbeddingConfig(
                num_embeddings=50000,
                embedding_dim=HSTU_EMBEDDING_DIM,
                name="user_id",
                data_type=DataType.FP16,
                feature_names=["user_id"],
            ),
        }
    else:
        raise NotImplementedError(f"Model {model} not implemented")


def get_backend_config(args, embedding_table_config: Dict[str, EmbeddingConfig], world_size: int) -> HybridGRBackendConfig:
    """
    Get backend configuration for hybrid GR inference.

    Args:
        args: Command-line arguments containing backend settings.
        embedding_table_config: Embedding table configurations.
        world_size: Total number of MPI processes.

    Returns:
        HybridGRBackendConfig: Backend configuration.
    """
    if args.use_mpi_lookup:
        partial_ranks = list(range(world_size - 1))
        local_device_ids = [i % torch.cuda.device_count() for i in range(world_size - 1)]
        item_table_config = embedding_table_config["item_id"]

        NVL = nve.MPIMemBlock(item_table_config.embedding_dim, item_table_config.num_embeddings, torch_data_type_to_nve_data_type(torch.float16), partial_ranks, local_device_ids)
        backend_config = HybridGRBackendConfig(
            batch_size=args.batch_size,
            perf_mode=args.mode,
            use_custom_stu=True,
            use_nve=True,
            use_multi_gpu=True,
            use_mpi=True,
            device_ids=[],
            gpu_cache_size_in_gigabytes=10,
            nve_memblock=NVL,
        )
    else:
        # base line meta's ref implementation, run with mpirun -n 2, 1 rank for entire backend, and modify issue_query to send to rank 0
        backend_config = HybridGRBackendConfig(
            batch_size=args.batch_size,
            perf_mode=args.mode,
            use_custom_stu=True,
            use_nve=False,
            use_multi_gpu=False,
        )
    return backend_config


def get_communicator_config(args) -> ZMQRequestSenderShardedConfig:
    """
    Get communicator configuration for inter-process communication.

    Args:
        args: Command-line arguments containing communicator settings.

    Returns:
        ZMQRequestSenderShardedConfig: Communicator configuration.

    Raises:
        NotImplementedError: If MPI communicator is selected (not yet implemented).
    """
    if args.communicator_type == "mpi":
        raise NotImplementedError("MPI communicator is not implemented yet")
    else:
        return ZMQRequestSenderShardedConfig(
            loadgen_hostname=args.loadgen_hostname,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            num_preds=2048,
            mode=args.mode,
            gpus_per_node=args.gpus_per_node,
        )


def get_dataset_latest(
    hstu_config,
    dataset_path: str,
    mode: str,
    total_queries: int,
    dataset_percentage: float,
    device: torch.device,
    scenario_name: str,
    offline_target_qps: int,
    target_duration: float,
    compute_eval: bool = False,
    batching_on_gpu: bool = False,
    max_buffer_indices: int = 1500000,
    max_buffer_lengths: int = 256,
) -> StreamingQuerySamplerRef:
    """
    Create streaming query sampler for MLPerf dataset.

    Args:
        hstu_config: HSTU model configuration.
        dataset_path: Path to the dataset directory.
        mode: Operating mode ("performance" or "accuracy").
        total_queries: Total number of queries to generate.
        dataset_percentage: Percentage of dataset to use (0.0-1.0).
        device: CUDA device for GPU batching (None for CPU-only).
        scenario_name: MLPerf scenario name ("Server" or "Offline").
        offline_target_qps: Target QPS for Offline scenario.
        target_duration: Target duration in milliseconds.
        compute_eval: Whether to compute evaluation metrics.
        batching_on_gpu: Whether to enable GPU-accelerated batching.
        max_buffer_indices: Maximum buffer size for indices.
        max_buffer_lengths: Maximum buffer size for lengths.

    Returns:
        StreamingQuerySamplerRef: Configured query sampler.
    """
    logger.info(f"Preparing dataset for mode: {mode}")
    if mode == "accuracy":
        is_inference = False
        total_queries = None
    else:
        is_inference = True
    dataset = DLRMv3StreamingMLPerfDataset(
        hstu_config=hstu_config,
        ratings_file_prefix=dataset_path,
        is_inference=is_inference,
        train_ts=90,
        total_ts=100,
        num_files=1,
        num_users=50000,
        num_items=1_000_000_000,
        num_categories=128,
        device=device,
        batching_on_gpu=batching_on_gpu,
        max_buffer_indices=max_buffer_indices,
        max_buffer_lengths=max_buffer_lengths,
    )
    streaming_query_sampler = StreamingQuerySamplerRef(
        ds=dataset,
        dataset_percentage=dataset_percentage,
        scenario_name=scenario_name,
        offline_target_qps=offline_target_qps,
        target_duration=target_duration,
        input_queries=total_queries,
        compute_eval=compute_eval,
    )
    # always load the entire dataset
    return streaming_query_sampler
