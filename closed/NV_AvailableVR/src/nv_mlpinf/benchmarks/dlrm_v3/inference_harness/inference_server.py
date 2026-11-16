from .backends.hybrid_GR_backend import HybridGRBackend, HybridGRBackendConfig
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from .dataset.streaming_query_sampler import StreamingQuerySamplerRef
from .mpi_utils import MPIDataPacketDSIndex, MPIDataPacketTSRequest
from generative_recommenders.dlrm_v3.datasets.dataset import (
    Samples,
)
from .backends.hybrid_GR_backend import CustomJaggedTensor
import gc
import torch
import queue
from threading import Event, Thread
from typing import Dict
from dataclasses import dataclass
import nvtx
import time
import logging
import sys
import os
import csv
from .mpi_utils import ZMQRequestSenderShardedConfig, ZMQRequestSenderSharded


@dataclass
class StreamDataPacket:
    """
    Data packet for producer-consumer pattern with CUDA streams.

    The producer (listener thread) prepares data on data_stream and records
    an event when done. The consumer (dispatch thread) waits on this event
    before running inference on the default stream.
    """
    query_ids: list
    batch: any  # The prepared batch (Samples or tensor dicts)
    transfer_done_event: torch.cuda.Event  # Signals when data prep is complete
    is_warmup: bool = False
    results: any = None


logger = logging.getLogger(__name__)


def sample_to_batch_pcie(sample: Samples, device: torch.cuda.device, non_blocking: bool = True) -> Dict:
    """
    Transfer sample data from CPU to GPU via PCIe for inference.

    Converts KeyedJaggedTensor (KJT) features into CustomJaggedTensor format
    and transfers them to the specified GPU device.

    Args:
        sample: Sample data containing UIH and candidate features as KJTs.
        device: Target CUDA device for data transfer.
        non_blocking: Whether to use non-blocking (asynchronous) transfers.

    Returns:
        tuple: (tensor_dict_uih, tensor_dict_candidates) containing CustomJaggedTensors
            on the target GPU device.
    """
    tensor_dict_uih = {}
    tensor_dict_candidates = {}

    with nvtx.annotate(f"hybrid_GR_backend - optimized_embedding_lookup - tensor prepare", color="orange"):
        for i in sample.uih_features_kjt.keys():
            tensor_dict_uih[i] = CustomJaggedTensor(
                values=sample.uih_features_kjt[i].values().to(device, non_blocking=non_blocking),
                lengths=sample.uih_features_kjt[i].lengths().to(device, non_blocking=non_blocking),
                offsets=sample.uih_features_kjt[i].offsets().to(device, non_blocking=non_blocking),
                max_length=sample.uih_features_kjt[i].lengths().max().item(),
                embeddings=None,
            )
        for i in sample.candidates_features_kjt.keys():
            tensor_dict_candidates[i] = CustomJaggedTensor(
                values=sample.candidates_features_kjt[i].values().to(device, non_blocking=non_blocking),
                lengths=sample.candidates_features_kjt[i].lengths().to(device, non_blocking=non_blocking),
                offsets=sample.candidates_features_kjt[i].offsets().to(device, non_blocking=non_blocking),
                max_length=sample.candidates_features_kjt[i].lengths().max().item(),
                embeddings=None,
            )
    return tensor_dict_uih, tensor_dict_candidates


class DLRMInferenceServer:
    """
    Distributed inference server for DLRM models with MLPerf LoadGen integration.

    Implements a producer-consumer pattern with separate threads for:
    - Listening for inference requests (producer)
    - Processing batches and running inference (consumer)

    Supports optional CUDA streams for overlapped data preparation and inference,
    and communicates with LoadGen via MPI or ZMQ for distributed benchmarking.

    Attributes:
        backend: HybridGRBackend instance for model inference.
        query_streaming_sampler: Dataset sampler for loading queries.
        device: CUDA device for inference.
        batch_size: Number of samples per inference batch.
        mode: Operating mode ("performance" or "accuracy").
        use_cuda_streams: Whether to use CUDA streams for overlapped execution.
    """

    def __init__(
        self,
        query_streaming_sampler: StreamingQuerySamplerRef,
        device: torch.cuda.device,
        local_rank: int = 0,
        batch_size: int = 16,
        warmup_steps: int = 5000,
        verbose: int = 0,
        loadgen_rank: int = 0,
        mode: str = "performance",
        use_cuda_streams: bool = False,
    ):
        """
        Initialize the DLRM inference server.

        Args:
            query_streaming_sampler: Dataset sampler for loading queries.
            device: CUDA device for inference.
            local_rank: MPI rank of this worker process.
            batch_size: Number of samples per inference batch.
            warmup_steps: Number of warmup iterations (not used in current implementation).
            verbose: Logging verbosity level (0=INFO, 1=DEBUG, -1=WARNING).
            loadgen_rank: MPI rank of the LoadGen process.
            mode: Operating mode ("performance" or "accuracy").
            use_cuda_streams: Enable overlapped batching/inference with CUDA streams.
        """
        # Core components
        self.backend: HybridGRBackend = None
        self.query_streaming_sampler = query_streaming_sampler
        self.device = device
        self.local_rank = local_rank
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        self.loadgen_rank = loadgen_rank
        self.mode = mode
        self.use_cuda_streams = use_cuda_streams

        # Model configuration storage (set during init_backend)
        self.hstu_config = None
        self.embedding_table_config = None
        self.backend_config = None

        # Producer-consumer queue for inference requests (bounded for backpressure)
        self.request_queue = queue.Queue(maxsize=10)
        torch.cuda.empty_cache()
        torch.cuda.set_device(self.device)

        # Disable garbage collection to reduce latency variance (affects all threads)
        gc.disable()
        self.cudart = torch.cuda.cudart()

        # Threading controls
        self.stop_event = Event()
        self.batching_thread = None

        # ========== CUDA Streams for Overlapped Execution ==========
        # data_stream: Used by producer (listener) for batching/data prep
        # default_stream: Used for inference (cutlass kernels use default_stream internally)
        # This allows overlapping data preparation with inference
        if self.use_cuda_streams:
            self.data_stream = torch.cuda.Stream(device=self.device)
            # Note: We use the default stream for inference because cutlass/CuTe DSL kernels
            # internally use cutlass_torch.default_stream(), not the PyTorch current stream.
            # Using default_stream avoids stream mismatch issues and hangs.
            logger.info(f"[Worker Comm: {self.local_rank}] CUDA streams enabled for overlapped batching/inference (data_stream + default_stream)")
        else:
            self.data_stream = None

        # Configure logger level based on verbosity setting
        if self.verbose >= 1:
            logger.setLevel(logging.DEBUG)
        elif self.verbose == 0:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # Performance tracking
        self.num_batch_processed = 0

    def init_communicator(self, shard_id=0, communicator_config: ZMQRequestSenderShardedConfig = None):
        """
        Initialize inter-process communication for the inference server.

        Sets up communication channels with LoadGen using either MPI or ZMQ (sharded).
        Sharded ZMQ reduces communication fan-in by grouping workers on the same node.

        Args:
            shard_id: Shard identifier (typically node ID). Workers on the same node share a shard
                to reduce communication fan-in from N:1 to N/shards:1.
            communicator_config: Configuration for the communication backend (ZMQ or MPI).
        """
        # Initialize batching latency tracking
        self._batching_latencies = []  # List of (batch_index, latency_ms)

        # Initialize communication backend (ZMQ or MPI)
        if isinstance(communicator_config, ZMQRequestSenderShardedConfig):
            self.request_sender = ZMQRequestSenderSharded(
                is_loadgen=False,
                shard_id=shard_id,
                rank=self.local_rank,
                config=communicator_config
            )
        else:
            # MPI communication path (not yet implemented)
            self.worker_comm = communicator_config.worker_comm
            self.loadgen_comm = communicator_config.loadgen_comm
            self.use_async_mpi = communicator_config.use_async_mpi
            raise NotImplementedError("MPI communication is not yet implemented")

        # Request listener thread controls
        self._request_listener_thread: Thread = None
        self._request_stop_event = Event()
        self._packet_counter = 0

        # Start listening for inference requests
        self._start_request_listener()

    def init_backend(self,
                     hstu_config: DlrmHSTUConfig,
                     embedding_table_config: Dict[str, EmbeddingConfig],
                     backend_config: HybridGRBackendConfig,
                     checkpoint_path: str = "/raid/data/zihaok_1/89/"
                     ):
        """
        Initialize the hybrid GR backend and load model weights.

        Creates the backend instance, initializes it with the model configuration,
        and loads both dense and sparse model components from checkpoints.
        Rank 0 is designated as the main rank for coordinating sparse table loading.

        Args:
            hstu_config: Configuration for the HSTU model architecture.
            embedding_table_config: Configuration for embedding tables.
            backend_config: Configuration for the hybrid GR backend.
            checkpoint_path: Path to the model checkpoint directory.
        """
        # Store configs for later use in multiprocessing
        self.hstu_config = hstu_config
        self.embedding_table_config = embedding_table_config
        self.backend_config = backend_config

        # Initialize backend for this device
        self.backend = HybridGRBackend(model_name="dlrm_hstu", device=self.device)
        self.backend.initialize(hstu_config=hstu_config, embedding_table_config=embedding_table_config, backend_config=backend_config)

        # Load model weights (dense on all ranks, sparse with coordination)
        self.backend.load_model_dense(checkpoint_path=checkpoint_path)
        if self.local_rank == 0:
            self.backend.load_model_sparse(checkpoint_path=checkpoint_path, main_rank=True)
        else:
            self.backend.load_model_sparse(checkpoint_path=checkpoint_path, main_rank=False)

        # Start the batching thread for processing inference requests
        self.batching_thread = Thread(target=self.batching_loop)
        self.batching_thread.start()

    def warmup(self, warmup_steps: int = 100):
        """
        Warm up the inference server with dummy predictions.

        Runs inference on sample data to initialize CUDA kernels and stabilize
        performance before benchmarking. Resets the dataset sampler to the
        starting timestamp after warmup completes.

        Args:
            warmup_steps: Number of warmup inference iterations to perform.
        """
        for i in range(warmup_steps):
            with torch.no_grad():
                # Log progress periodically
                if i % 50 == 0 or i == warmup_steps - 1:
                    logger.info(f"[Worker Comm: {self.local_rank}] Warmup step {i} / {warmup_steps}")

                # Sample and predict
                outputs_ts = self.query_streaming_sampler.get_samples_indices(range(self.batch_size))
                warmup_samples = self.query_streaming_sampler.ds.get_samples_with_ts_updated(outputs_ts)
                _ = self.backend.predict(warmup_samples)

                # Reset when reaching end of dataset
                if self.query_streaming_sampler.ts_processed_cnt >= self.query_streaming_sampler.total_requests:
                    self.query_streaming_sampler.init_sut()

        torch.cuda.empty_cache()
        # Reset the sampler to the start timestamp for benchmarking
        self.query_streaming_sampler.init_sut()

    def enqueue_batch(self, batch_data: Dict):
        """
        Enqueue a batch of inference requests to the processing queue.

        Args:
            batch_data: Dictionary containing batch information and data.
        """
        self.request_queue.put(batch_data)

    def batching_loop(self):
        """
        Consumer thread: Runs inference on default_stream.

        With CUDA streams enabled:
            - Waits on transfer_done_event before reading batch data
            - Runs inference on default_stream (overlaps with next batch prep on data_stream)
            - Note: cutlass/CuTe DSL kernels use default_stream internally, so we use it here
              to avoid stream mismatch issues

        Timeline (overlapped):
            data_stream:     [Prep B0][Prep B1][Prep B2][Prep B3]...
            default_stream:         [Infer B0][Infer B1][Infer B2]...
        """
        # CRITICAL: Set CUDA device context for this thread
        # Each thread in PyTorch needs its own device context
        torch.cuda.set_device(self.device)

        # Ensure garbage collection is disabled in this thread for consistent latency
        gc.disable()
        logger.info(f"[Worker Comm: {self.local_rank}] [Batching Thread] Started batching thread and set CUDA device to {self.device}")

        # Process requests until stop signal and queue is empty
        while not self.stop_event.is_set() or not self.request_queue.empty():
            if self.request_queue.empty():
                time.sleep(0.0001)  # 100μs sleep to avoid busy-waiting
                continue

            data_packet = self.request_queue.get()

            # ========== CUDA Streams Path: Overlapped Execution ==========
            if self.use_cuda_streams:
                # Wait for data prep to complete on data_stream before running inference
                # We use default_stream for inference because cutlass/CuTe DSL kernels
                # internally use cutlass_torch.default_stream(), not PyTorch's current stream
                torch.cuda.current_stream(self.device).wait_event(data_packet.transfer_done_event)

                # Run inference on default_stream (can overlap with next batch prep on data_stream)
                with nvtx.annotate(f"Inference-{self.num_batch_processed}", color="yellow"):
                    if data_packet.is_warmup:
                        predict, labels, weights = self.backend.predict_dummy(data_packet.batch)
                    else:
                        with torch.no_grad():
                            predict, labels, weights = self.backend.predict(data_packet.batch)

                # Synchronize default_stream before sending results
                torch.cuda.current_stream(self.device).synchronize()
            else:
                # ========== Non-Stream Path: Sequential Execution ==========
                if data_packet.is_warmup:
                    predict, labels, weights = self.backend.predict_dummy(data_packet.batch)
                else:
                    with torch.no_grad():
                        predict, labels, weights = self.backend.predict(data_packet.batch)

                # Non-stream path runs on the current stream; sync that stream only.
                torch.cuda.current_stream(self.device).synchronize()

            self.num_batch_processed += 1

            # Send results back to LoadGen
            data_packet.results = (predict, labels, weights)
            result_data_packet = MPIDataPacketDSIndex(
                query_ids=data_packet.query_ids,
                ts_request_pairs=data_packet.ts_request_pairs,
                results=data_packet.results,
            )
            with nvtx.annotate(f"send to loadgen rank", color="orange"):
                self.request_sender.send_to_loadgen(result_data_packet)

        logger.debug(f"[Worker Comm: {self.local_rank}] [Batching Thread] Stopped successfully.")

    def stop_batching(self, timeout: float = 10.0):
        """
        Gracefully stop the batching thread.

        Args:
            timeout: Maximum time to wait for the thread to finish (in seconds)
        """

        logger.debug(f"[Worker Comm: {self.local_rank}] [Batching Thread] Stopping batching thread on {self.device}...")
        self.stop_event.set()  # Signal the thread to stop

        # Stop listener thread if running
        if self._request_listener_thread is not None and self._request_listener_thread.is_alive():
            self._stop_request_listener(timeout=timeout)

        # Wait for the thread to finish
        self.batching_thread.join(timeout=timeout)

        if self.batching_thread.is_alive():
            logger.warning(f"[Worker Comm: {self.local_rank}] [Batching Thread] Warning: Batching thread on {self.device} did not stop within {timeout}s")
        else:
            logger.debug(f"[Worker Comm: {self.local_rank}] [Batching Thread] Batching thread on {self.device} stopped successfully.")

    def _start_request_listener(self, tag: int = None, poll_interval_sec: float = 0.0001):
        """
        Producer thread: Listens for MPI messages and prepares batches on data_stream.

        With CUDA streams enabled:
            - Batching/data prep happens on data_stream
            - Records event when prep is done
            - Consumer (dispatch thread) waits on event before inference on default_stream

        Timeline (overlapped):
            data_stream:     [Prep B0][Prep B1][Prep B2][Prep B3]...
            default_stream:         [Infer B0][Infer B1][Infer B2]...

        Args:
            tag: Message tag to match. If None, listens for any tag.
            poll_interval_sec: Sleep between probe cycles to avoid busy-waiting
        """
        self._request_stop_event.clear()
        self.sample = None

        def _listen_loop():
            # CRITICAL: Set CUDA device context for this thread
            # Each thread in PyTorch needs its own device context
            torch.cuda.set_device(self.device)

            # Ensure garbage collection is disabled in this thread for consistent latency
            gc.disable()

            while not self._request_stop_event.is_set():
                # Check for incoming requests from LoadGen
                if self.request_sender.probe_from_loadgen():
                    with nvtx.annotate(f"receive from loadgen rank", color="blue"):
                        data_packet = self.request_sender.receive_from_loadgen()
                    self._packet_counter += 1

                    # ========== CUDA Streams Path: Prepare Data on data_stream ==========
                    transfer_done_event = None
                    if self.use_cuda_streams:
                        # Create event to signal when data preparation is complete
                        transfer_done_event = torch.cuda.Event()

                        with torch.cuda.stream(self.data_stream):
                            if data_packet.is_warmup:
                                sample = None
                            else:
                                # Batch preparation on data_stream (can overlap with inference)
                                with nvtx.annotate(f"Producer-Prep-batching-{self._packet_counter}", color="blue"):
                                    batch_start = time.perf_counter()
                                    sample = self.query_streaming_sampler.ds.get_samples_with_ts_updated(data_packet.ts_request_pairs)
                                    batch_end = time.perf_counter()

                                    batch_latency_ms = (batch_end - batch_start) * 1000
                                    self._batching_latencies.append((self._packet_counter, batch_latency_ms))

                            # Record event - dispatch thread will wait on this
                            transfer_done_event.record(self.data_stream)
                    else:
                        # ========== Non-Stream Path: Sequential Data Preparation ==========
                        with nvtx.annotate(f"MPI Listener - batching dataset index packet and sending to dispatch thread", color="green"):
                            if data_packet.is_warmup:
                                sample = None
                            else:
                                batch_start = time.perf_counter()
                                sample = self.query_streaming_sampler.ds.get_samples_with_ts_updated(data_packet.ts_request_pairs)
                                batch_end = time.perf_counter()

                                batch_latency_ms = (batch_end - batch_start) * 1000
                                self._batching_latencies.append((self._packet_counter, batch_latency_ms))

                    # Create data packet and enqueue for dispatch thread
                    data_packet_result = MPIDataPacketTSRequest(
                        query_ids=data_packet.query_ids,
                        ts_request_pairs=data_packet.ts_request_pairs,
                        batch=sample,
                        is_warmup=data_packet.is_warmup,
                        transfer_done_event=transfer_done_event
                    )
                    self.enqueue_batch(data_packet_result)
                else:
                    # No requests available, sleep briefly to avoid busy-waiting
                    time.sleep(poll_interval_sec)

            logger.debug(f"[Worker Comm: {self.local_rank}] [Backend Request Listener] Stopped.")

        self._request_listener_thread = Thread(target=_listen_loop, daemon=True)
        self._request_listener_thread.start()

    def _stop_request_listener(self, timeout: float = 10.0):
        """
        Gracefully stop the request listener thread.

        Signals the listener thread to stop, waits for it to finish, and
        shuts down the communication channel.

        Args:
            timeout: Maximum time to wait for the thread to finish (in seconds).
        """
        if self._request_listener_thread is None:
            return

        self._request_stop_event.set()
        self._request_listener_thread.join(timeout=timeout)
        self.request_sender.shutdown()
        logger.info(f"[Worker Comm: {self.local_rank}] [Backend Request Listener] received {self._packet_counter} packets.")

    def dump_latency(self, output_dir: str = "batching_latency"):
        """
        Dump batching latencies to a CSV file.

        Args:
            output_dir: Directory to save the latency file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"backend_batching_latency_{self.local_rank}.csv")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["batch_index", "latency_ms"])
            for batch_index, latency_ms in self._batching_latencies:
                writer.writerow([batch_index, f"{latency_ms:.4f}"])

        logger.info(f"[Worker Comm: {self.local_rank}] Batching latency saved to {output_path} ({len(self._batching_latencies)} batches)")
