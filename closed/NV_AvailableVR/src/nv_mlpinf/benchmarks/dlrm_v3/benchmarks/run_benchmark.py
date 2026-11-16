#!/usr/bin/env python3
"""
Multi-GPU benchmark script for DLRMv3 with MLPerf LoadGen.
Uses a "small world" setup where loadgen runs as rank 8 in the same MPI world.
Automatically adds project root to Python path for module imports.
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress TorchScript dtype annotation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit.annotations")

import torch
from typing import Dict
import socket
rank = int(os.environ.get("SLURM_PROCID", 0))
os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_rank_{rank}"

# Add project root to Python path to allow imports from inference_harness
# This handles the case where the script is run directly without PYTHONPATH set
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mpi4py import MPI
from inference_harness.inference_server import DLRMInferenceServer
from inference_harness.tools.model_configs import get_hstu_configs, get_embedding_table_config, get_backend_config, get_dataset_latest, get_communicator_config
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from inference_harness.backends.hybrid_GR_backend import HybridGRBackendConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from inference_harness.mpi_utils import ZMQRequestSenderShardedConfig
from inference_harness.test_runner import TestRunner, clear_accuracy_buffers
from inference_harness.test_runner import parse_user_conf


import logging
import argparse

import mlperf_loadgen as lg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    """
    Parse command line arguments for the multi-GPU benchmark script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing dataset paths,
            model configuration, LoadGen settings, and performance tuning options.
    """
    parser = argparse.ArgumentParser(
        description="MLPerf LoadGen integration for DLRMv3 with DLRMv3MLPerfDataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the preprocessed dataset directory"
    )
    parser.add_argument(
        "--dataset-percentage",
        type=float,
        default=1,
        help="Percentage of dataset to load (0.0-1.0)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the checkpoint file"
    )
    # LoadGen configuration args
    parser.add_argument(
        "--scenario",
        type=str,
        default="Server",
        choices=["Server", "Offline"],
        help="MLPerf scenario to run"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="performance",
        choices=["performance", "accuracy"],
        help="MLPerf test mode"
    )
    parser.add_argument(
        "--user-conf",
        type=str,
        default="",
        help="Path to user.conf file (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for results"
    )
    # NVE configuration args
    parser.add_argument(
        "--use-mpi-lookup",
        action="store_true",
        help="Enable MPI-based embedding lookup with NVE"
    )
    # Performance tuning args
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (Server mode only)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup inference steps before benchmarking"
    )

    # Communicator configuration arguments
    parser.add_argument(
        "--communicator-type",
        type=str,
        choices=["mpi", "zmq"],
        default="zmq",
        help="Communicator type: 'mpi' (MPI) or 'zmq' (ZMQ), MPI is not supported yet"
    )

    # ZMQ-specific configuration (not applicable when using MPI communicator)
    parser.add_argument(
        "--loadgen-hostname",
        type=str,
        default=None,
        help="Hostname of the LoadGen rank (for ZMQ communicator)"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Number of ZMQ socket shards (typically equals num_nodes). Reduces communication fan-in from N:1 to N/shards:1"
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
        help="Number of GPUs per node (used for shard calculation)"
    )

    # Dataset GPU batching arguments
    parser.add_argument(
        "--batching-on-gpu",
        action="store_true",
        help="Enable GPU batching for KJT collation in dataset (faster but uses more GPU memory; Server mode only)"
    )
    parser.add_argument(
        "--use-cuda-streams",
        action="store_true",
        help="Enable CUDA streams for overlapped batching and inference (Server mode only)"
    )
    return parser.parse_args()


def initialize_inference_server(args,
                                local_rank,
                                local_device_id,
                                loadgen_rank,
                                hstu_config: DlrmHSTUConfig,
                                embedding_table_config: Dict[str, EmbeddingConfig],
                                backend_config: HybridGRBackendConfig,
                                communicator_config: ZMQRequestSenderShardedConfig,
                                settings):
    """
    Initialize and configure the DLRMInferenceServer for worker processes.

    Args:
        args: Command line arguments containing dataset and model configuration.
        local_rank: MPI rank of the current worker process.
        local_device_id: CUDA device ID for the current worker.
        loadgen_rank: MPI rank of the LoadGen process.
        hstu_config: Configuration for the HSTU model architecture.
        embedding_table_config: Configuration for embedding tables.
        backend_config: Configuration for the hybrid GR backend.
        communicator_config: Configuration for inter-process communication.
        settings: LoadGen test settings parsed from user.conf.

    Returns:
        DLRMInferenceServer: Initialized inference server ready to process queries.
    """
    current_device = torch.device(f"cuda:{local_device_id}")
    # hack, in dataset instantiation, we need to know
    # how many sample user is trying to run, in order to form this dataset
    # hence, we get offline target qps and min duration from loadgen user.conf, which is used to calculate
    # total number of queries loadgen will issue.

    streaming_query_sampler = get_dataset_latest(
        hstu_config=hstu_config,
        dataset_path=args.dataset_path,
        mode=args.mode,
        total_queries=settings.min_query_count,
        dataset_percentage=args.dataset_percentage,
        device=current_device,
        scenario_name=args.scenario,
        offline_target_qps=settings.offline_expected_qps,
        target_duration=settings.min_duration_ms,
        compute_eval=True if args.mode == "accuracy" else False,
        batching_on_gpu=args.batching_on_gpu,
        max_buffer_indices=args.batch_size * 48000,  # 48000 is per sample's rough index vector's size
        max_buffer_lengths=args.batch_size * 8,  # 8 is per sample's length vector's size
    )
    streaming_query_sampler.load_query_samples_preprocessed(args.dataset_path)

    # Calculate shard_id based on node assignment
    # Workers on the same node share a shard to reduce communication fan-in
    shard_id = local_rank // args.gpus_per_node

    inf_server = DLRMInferenceServer(
        query_streaming_sampler=streaming_query_sampler,
        device=current_device,
        local_rank=local_rank,
        batch_size=args.batch_size,
        verbose=0,
        loadgen_rank=loadgen_rank,
        mode=args.mode,
        use_cuda_streams=args.use_cuda_streams
    )
    inf_server.init_communicator(
        shard_id=shard_id,
        communicator_config=communicator_config
    )
    inf_server.init_backend(hstu_config=hstu_config, embedding_table_config=embedding_table_config, backend_config=backend_config, checkpoint_path=args.checkpoint_path)

    # Model must be loaded on every worker rank for distributed inference
    logger.info(f"[Worker Comm: {local_rank}] Inference server initialized successfully.")
    return inf_server


def initialize_loadgen_runner(args, local_rank, worker_world_size, hstu_config: DlrmHSTUConfig, communicator_config: ZMQRequestSenderShardedConfig, settings):
    """
    Initialize the MLPerf LoadGen test runner.

    Args:
        args: Command line arguments containing dataset and LoadGen configuration.
        local_rank: MPI rank of the LoadGen process.
        worker_world_size: Total number of worker processes (excluding LoadGen).
        hstu_config: Configuration for the HSTU model architecture.
        communicator_config: Configuration for inter-process communication.
        settings: LoadGen test settings parsed from user.conf.

    Returns:
        TestRunner: Initialized test runner ready to execute MLPerf LoadGen tests.
    """
    streaming_query_sampler = get_dataset_latest(
        hstu_config=hstu_config,
        dataset_path=args.dataset_path,
        mode=args.mode,
        total_queries=settings.min_query_count,
        dataset_percentage=args.dataset_percentage,
        device=None,
        scenario_name=args.scenario,
        offline_target_qps=settings.offline_expected_qps,
        target_duration=settings.min_duration_ms,
        compute_eval=True if args.mode == "accuracy" else False,
        batching_on_gpu=False,  # Loadgen doesn't use GPU batching
        max_buffer_indices=args.batch_size * 48000,  # 48000 is per sample's rough index vector's size
        max_buffer_lengths=args.batch_size * 8,  # 8 is per sample's length vector's size
    )
    runner = TestRunner(
        streaming_query_sampler=streaming_query_sampler,
        batch_size=args.batch_size,
        rank=local_rank,
        worker_world_size=worker_world_size,
        verbose=0,
        mode=args.mode,
        scenario=args.scenario,
        communicator_config=communicator_config
    )
    return runner


def run_loadgen(runner: TestRunner, args, settings, log_settings, local_rank):
    """
    Execute the MLPerf LoadGen benchmark test.

    Sets up and runs the LoadGen test with the configured scenario and mode,
    then cleans up resources after completion.

    Args:
        runner: Configured TestRunner instance ready to execute the test.
        args: Command line arguments.
        settings: LoadGen test settings.
        log_settings: LoadGen log settings.
        local_rank: MPI rank of the LoadGen process.
    """
    sut, qsl = runner.setup_loadgen(args)

    logger.info(f"[Loadgen Child Comm: {local_rank}] Starting LoadGen test: {args.scenario} / {args.mode}")
    logger.info(f"[Loadgen Child Comm: {local_rank}] {'=' * 80}")

    # Execute the MLPerf LoadGen benchmark
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)

    # Clear accuracy buffers to free memory held by completion thread pool
    clear_accuracy_buffers()

    lg.DestroySUT(sut)
    lg.DestroyQSL(qsl)

    # Stop the results listener thread
    runner.result_stop_event.set()
    runner.results_thread.join(timeout=5.0)


if __name__ == '__main__':
    # ========== MPI Initialization ==========
    comm = MPI.COMM_WORLD
    local_rank = comm.Get_rank()
    local_device_id = local_rank % torch.cuda.device_count()
    world_size = comm.Get_size()

    # LoadGen always runs on the last rank; all other ranks are workers
    loadgen_rank = world_size - 1
    worker_world_size = world_size - 1

    # ========== Configuration Initialization ==========
    # Parse arguments and load model/dataset configurations
    args = get_args()
    model_config = "production"
    hstu_config = get_hstu_configs(model_config)
    embedding_table_config = get_embedding_table_config(model_config)
    communicator_config = get_communicator_config(args)

    # Parse user.conf once and reuse settings throughout
    settings, log_settings = parse_user_conf(args)

    # Backend config must be created on every rank, even those not storing embedding tables
    # This is required for proper MPI memory block allocation
    backend_config = get_backend_config(args=args, embedding_table_config=embedding_table_config, world_size=world_size)

    # ========== Worker and LoadGen Initialization ==========
    inf_server = None
    runner = None

    if local_rank != loadgen_rank:
        inf_server = initialize_inference_server(args,
                                                 local_rank,
                                                 local_device_id,
                                                 loadgen_rank,
                                                 hstu_config,
                                                 embedding_table_config,
                                                 backend_config,
                                                 communicator_config,
                                                 settings
                                                 )
        comm.Barrier()
        inf_server.warmup(warmup_steps=args.warmup_steps)
        comm.Barrier()
        comm.Barrier()
    else:
        # LoadGen rank: Initialize test runner
        runner = initialize_loadgen_runner(args,
                                           local_rank,
                                           worker_world_size,
                                           hstu_config,
                                           communicator_config,
                                           settings
                                           )

        # Wait for all workers to complete set up including loading weights
        comm.Barrier()
        # wait for all workers to complete warmup
        comm.Barrier()
        runner.benchmark_zmq_latency(num_requests=50000)
        comm.Barrier()

    # ========== Run MLPerf LoadGen Benchmark ==========
    # LoadGen starts measuring performance; workers listen for queries
    if local_rank == loadgen_rank:
        run_loadgen(runner, args, settings, log_settings, local_rank)
    comm.Barrier()

    # ========== Cleanup and Shutdown ==========
    if local_rank != loadgen_rank:
        # Worker ranks: Stop inference server and optionally dump performance metrics
        inf_server._stop_request_listener(timeout=10.0)
        inf_server.stop_batching(timeout=10.0)
        # batching latency debugger
        # inf_server.dump_latency(output_dir=args.output_dir)
        logger.info(f"[Worker: {local_rank}] Inference server stopped.")
    else:
        # LoadGen rank: Shutdown test runner
        runner.shutdown()
        logger.info(f"[Loadgen: {local_rank}] Loadgen runner stopped.")

    # Ensure all ranks complete before exiting
    comm.Barrier()
