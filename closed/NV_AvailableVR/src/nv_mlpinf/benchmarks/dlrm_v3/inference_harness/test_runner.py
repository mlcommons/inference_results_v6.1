from inference_harness.mpi_utils import MPIDataPacketDSIndex, MPIDataPacketTSRequest, ZMQRequestSenderShardedConfig, ZMQRequestSenderSharded
from inference_harness.dataset.streaming_query_sampler import StreamingQuerySamplerRef
import logging
import sys
import os
import time
import threading
from typing import Dict, List, Optional, Tuple
import argparse
import mlperf_loadgen as lg
import numpy as np
import torch
import nvtx
import gc

# Import the multi-threaded QuerySamplesComplete pool
import sys
sys.path.insert(0, "/opt/ffi_utils/build")
import TestPybind

logger = logging.getLogger(__name__)


# ========== Global Thread Pool for QuerySamplesComplete ==========
# Lazy initialization - created on first use
_qsc_pool = None
_QSC_POOL_NUM_THREADS = 10


def get_qsc_pool():
    """
    Get or create the global QuerySamplesComplete thread pool.

    Returns:
        TestPybind.QuerySamplesCompletePool: Thread pool for async query completion.
    """
    global _qsc_pool
    if _qsc_pool is None:
        _qsc_pool = TestPybind.QuerySamplesCompletePool(
            num_threads=_QSC_POOL_NUM_THREADS,
            test_mode=False
        )
    return _qsc_pool


# ========== Buffer Management for Response Buffers ==========
# Prevents use-after-free by keeping buffers alive until thread pool processes them
_accuracy_buffers = []
_ACCURACY_BUFFER_MAX_SIZE = 100  # Maximum buffers to retain before cleanup


def _store_accuracy_buffer(buf):
    """
    Store buffer reference to prevent garbage collection until thread pool processes it.

    In performance/accuracy mode, we pass buffer pointers to a C++ thread pool that
    processes them asynchronously. We must keep Python references alive to prevent
    garbage collection.

    Args:
        buf: NumPy array or buffer to keep alive.
    """
    global _accuracy_buffers
    _accuracy_buffers.append(buf)

    # Cleanup old buffers when we exceed max size (thread pool should have processed them)
    if len(_accuracy_buffers) > _ACCURACY_BUFFER_MAX_SIZE:
        # Keep only the last half to avoid unbounded growth
        _accuracy_buffers = _accuracy_buffers[_ACCURACY_BUFFER_MAX_SIZE // 2:]


def clear_accuracy_buffers():
    """Clear all stored accuracy buffers. Call after benchmarking completes."""
    global _accuracy_buffers
    _accuracy_buffers.clear()


# MLPerf scenario mapping
SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

# MLPerf test mode mapping
MODE_MAP = {
    "performance": lg.TestMode.PerformanceOnly,
    "accuracy": lg.TestMode.AccuracyOnly,
    "find_peak": lg.TestMode.FindPeakPerformance,
}


def parse_user_conf(
    args: argparse.Namespace,
) -> Tuple[lg.TestSettings, lg.LogSettings, str]:
    """
    Parse MLPerf user.conf and construct LoadGen settings and log settings.
    """
    settings = lg.TestSettings()
    settings.scenario = SCENARIO_MAP[args.scenario]
    settings.mode = MODE_MAP[args.mode]

    settings.FromConfig(args.user_conf, "dlrm-v3", args.scenario)
    # settings.min_query_count = streaming_query_sampler.num_queries

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.abspath(".")
    os.makedirs(output_dir, exist_ok=True)

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = output_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = False

    return settings, log_settings


@nvtx.annotate(f"report_loadgen_best_perf", color="yellow")
def report_loadgen_best_perf(
    data_packet: MPIDataPacketDSIndex,
    mt_target_preds: torch.Tensor,
    mt_target_labels: torch.Tensor,
    mt_target_weights: torch.Tensor,
    compute_eval: bool = False,
    ts_idx_list: Optional[List[float]] = None,
    query_idx_list: Optional[List[float]] = None,
):
    """
    Report inference results back to MLPerf LoadGen using multi-threaded completion.

    Uses a C++ thread pool (TestPybind.QuerySamplesCompletePool) to handle
    QuerySamplesComplete calls asynchronously, allowing the main thread to
    continue processing without blocking on LoadGen API calls.

    Args:
        data_packet: Data packet containing query IDs for result correlation.
        mt_target_preds: Model predictions tensor.
        mt_target_labels: Ground truth labels tensor.
        mt_target_weights: Sample weights tensor.
        compute_eval: If True, includes labels and weights for accuracy evaluation;
            if False, only predictions are reported (performance mode).
    """
    pool = get_qsc_pool()

    num_queries = len(data_packet.query_ids)
    if ts_idx_list is None or len(ts_idx_list) != num_queries:
        ts_idx_list = [-1.0] * num_queries
    if query_idx_list is None or len(query_idx_list) != num_queries:
        query_idx_list = [-1.0] * num_queries

    if not compute_eval:
        # Performance mode: include ts_idx/query_idx metadata + predictions
        assert mt_target_preds.is_contiguous(), "mt_target_preds is not contiguous"
        candidate_size = mt_target_preds.size(1) // num_queries
        with nvtx.annotate(f"convert to fp32", color="yellow"):
            all_preds = mt_target_preds[0].contiguous().float().numpy()  # pyre-ignore [61]

        floats_per_query = candidate_size + 2
        bytes_per_query = floats_per_query * 4
        with nvtx.annotate(f"build response buffer", color="purple"):
            response_buffer = np.empty(num_queries * floats_per_query, dtype=np.float32)
            for i in range(num_queries):
                start = candidate_size * i
                end = candidate_size * (i + 1)
                buf_offset = i * floats_per_query
                response_buffer[buf_offset] = float(ts_idx_list[i])
                response_buffer[buf_offset + 1] = float(query_idx_list[i])
                response_buffer[buf_offset + 2: buf_offset + 2 + candidate_size] = all_preds[start:end]

        with nvtx.annotate(f"report back to loadgen", color="yellow"):
            base_ptr = response_buffer.ctypes.data
            pool.enqueue_batch(data_packet.query_ids, base_ptr, bytes_per_query)

        _store_accuracy_buffer(response_buffer)
    else:
        # Accuracy mode: predictions + labels + weights + candidate_size
        candidate_size = mt_target_preds.size(1) // num_queries

        with nvtx.annotate(f"convert to fp32", color="yellow"):
            all_preds = mt_target_preds[0].contiguous().float().numpy()  # pyre-ignore [61]
            all_labels = mt_target_labels[0].contiguous().float().numpy()  # pyre-ignore [16,61]
            all_weights = mt_target_weights[0].contiguous().float().numpy()  # pyre-ignore [61]

        # Each query: ts_idx + query_idx + preds + labels + weights + candidate_size
        floats_per_query = 3 * candidate_size + 3
        bytes_per_query = floats_per_query * 4  # float32 = 4 bytes

        with nvtx.annotate(f"build response buffer", color="purple"):
            # Build contiguous buffer with interleaved data for each query
            response_buffer = np.empty(num_queries * floats_per_query, dtype=np.float32)

            for i in range(num_queries):
                start = candidate_size * i
                end = candidate_size * (i + 1)
                buf_offset = i * floats_per_query

                response_buffer[buf_offset] = float(ts_idx_list[i])
                response_buffer[buf_offset + 1] = float(query_idx_list[i])
                response_buffer[buf_offset + 2: buf_offset + 2 + candidate_size] = all_preds[start:end]
                response_buffer[
                    buf_offset + 2 + candidate_size: buf_offset + 2 + 2 * candidate_size
                ] = all_labels[start:end]
                response_buffer[
                    buf_offset + 2 + 2 * candidate_size: buf_offset + 2 + 3 * candidate_size
                ] = all_weights[start:end]
                response_buffer[buf_offset + 2 + 3 * candidate_size] = float(candidate_size)

        with nvtx.annotate(f"report back to loadgen", color="yellow"):
            base_ptr = response_buffer.ctypes.data
            # Enqueue to thread pool - responses are built in C++
            pool.enqueue_batch(data_packet.query_ids, base_ptr, bytes_per_query)

        # CRITICAL: Store buffer reference to prevent garbage collection
        # The thread pool processes asynchronously, so buffer must stay alive
        _store_accuracy_buffer(response_buffer)


class TestRunner:
    """
    MLPerf LoadGen test runner for distributed DLRM inference benchmarking.

    Manages the LoadGen process, coordinating query issuance to distributed workers
    and collecting results for performance/accuracy evaluation. Supports both Server
    and Offline scenarios with configurable batch sizes and communication backends.

    The runner uses a producer-consumer pattern:
    - Producer: LoadGen issues queries via issue_queries callback
    - Consumer: Results listener thread collects responses from workers

    Attributes:
        streaming_query_sampler: Dataset sampler for loading queries.
        batch_size: Number of samples per batch sent to workers.
        rank: MPI rank of this LoadGen process.
        mode: Operating mode ("performance" or "accuracy").
        scenario: MLPerf scenario ("server" or "offline").
        request_sender: Communication backend for worker coordination.
    """

    def __init__(self,
                 streaming_query_sampler: StreamingQuerySamplerRef,
                 batch_size: int = 128,
                 rank: int = 0,
                 worker_world_size: int = 1,
                 verbose: int = 0,
                 mode: str = "performance",
                 scenario: str = "offline",
                 communicator_config: ZMQRequestSenderShardedConfig = None):
        """
        Initialize the MLPerf LoadGen test runner.

        Args:
            streaming_query_sampler: Dataset sampler for loading queries.
            batch_size: Number of samples per batch sent to workers.
            rank: MPI rank of this LoadGen process.
            worker_world_size: Total number of inference worker processes.
            verbose: Logging verbosity level (0=INFO, 1=DEBUG, -1=WARNING).
            mode: Operating mode ("performance" or "accuracy").
            scenario: MLPerf scenario ("server" or "offline").
            communicator_config: Configuration for inter-process communication.
        """
        # Disable garbage collection for consistent latency
        gc.disable()

        # Core configuration
        self.streaming_query_sampler = streaming_query_sampler
        self.batch_size = batch_size
        self.rank = rank
        self.mode = mode
        self.scenario = scenario
        self._worker_ranks = list(range(worker_world_size))
        self._num_workers = worker_world_size

        # Initialize communication backend
        if isinstance(communicator_config, ZMQRequestSenderShardedConfig):
            logger.info(f"[Loadgen Rank: {rank}] Loadgen hostname: {communicator_config.loadgen_hostname}, num_shards: {communicator_config.num_shards}")
            # Sharded ZMQ: Creates num_shards socket pairs for better scalability
            # Workers connect to ports based on their shard_id for reduced fan-in
            self.request_sender = ZMQRequestSenderSharded(
                is_loadgen=True,
                shard_id=0,  # LoadGen doesn't use shard_id
                rank=rank,
                config=communicator_config
            )
        else:
            # MPI communication path (not yet implemented)
            self.worker_world_size = worker_world_size
            self.use_async_mpi = communicator_config.use_async_mpi
            pass

        # Debugging and testing utilities
        self._dummy_response_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
        self.debug_accuracy = False
        self._accuracy_dump_file = None

        # Results collection thread and query samples complete pool
        self.query_samples_complete_pool = TestPybind.QuerySamplesCompletePool(
            num_threads=_QSC_POOL_NUM_THREADS,
            test_mode=False
        )
        self.result_stop_event = threading.Event()
        self.results_thread = threading.Thread(target=self._results_listener, daemon=True)
        self.results_thread.start()

        # Performance tracking counters
        self.out_batch_counter = 0  # Number of batches sent to workers
        self.in_batch_counter = 0   # Number of batches received from workers
        self.num_issue_queries_called = 0  # Number of times issue_queries was called

        # Latency testing mode (for debugging/profiling, not MLPerf benchmarking)
        self.test_latency_mode = False
        self._latency_send_times: Dict[int, float] = {}  # request_idx -> send timestamp
        self._latency_results: list = []  # list of (request_idx, latency_ms)

        # Current batch accumulation
        self.current_query_ids = []
        self.current_qsl_ids = []

        # Configure logging verbosity
        self.verbose = verbose
        if self.verbose >= 1:
            logger.setLevel(logging.DEBUG)
        elif self.verbose == 0:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

    def setup_loadgen(self, args: argparse.Namespace):
        """
        Set up MLPerf LoadGen components: SUT, QSL, and settings.

        Constructs the System Under Test (SUT) with query issuance callbacks,
        the Query Sample Library (QSL) with dataset access, and configures
        test settings from user.conf and command-line arguments.

        Args:
            args: Command-line arguments containing scenario, mode, and configuration paths.

        Returns:
            tuple: (sut, qsl, settings, log_settings) for use with LoadGen.StartTest.
        """
        logger.info(f"[Loadgen Rank: {self.rank}] Setting up LoadGen...")
        self.test_latency_mode = False

        self.mode = args.mode

        sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

        count = self.streaming_query_sampler.get_item_count()
        print("QSL has count: ", count)
        qsl = lg.ConstructQSL(
            count,
            count,
            self.streaming_query_sampler.load_query_samples,
            self.streaming_query_sampler.unload_query_samples,
        )

        return sut, qsl

    def issue_queries(self, query_samples):
        """
        MLPerf LoadGen callback to issue queries to inference workers.

        Routes query issuance to scenario-specific handlers (Server or Offline).
        This method is called by LoadGen whenever queries need to be processed.

        Args:
            query_samples: List of QuerySample objects from LoadGen.

        Raises:
            ValueError: If an unsupported scenario is specified.
        """
        if self.scenario.lower() == "server":
            self.issue_queries_server(query_samples)
        elif self.scenario.lower() == "offline":
            self.issue_queries_offline(query_samples)
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}")

    def issue_queries_server(self, query_samples):
        """
        Issue queries to workers in Server scenario.

        In Server mode, queries arrive dynamically and are batched until
        batch_size is reached, then sent to workers via round-robin.

        Args:
            query_samples: List of QuerySample objects from LoadGen.
        """
        with nvtx.annotate(f"issue_queries", color="blue"):
            self.num_issue_queries_called += 1

            # Accumulate queries into current batch
            for sample in query_samples:
                self.current_query_ids.append(sample.id)
                self.current_qsl_ids.append(sample.index)

            # Send batch when full
            if len(self.current_query_ids) >= self.batch_size:
                # Get timestamp-request_id pairs from the query sampler
                outputs_ts = self.streaming_query_sampler.get_samples_indices(self.current_query_ids)

                # Create data packet with timestamp-request pairs
                data_packet_ts_request = MPIDataPacketTSRequest(
                    query_ids=self.current_query_ids.copy(),
                    ts_request_pairs=outputs_ts,  # List of (ts, request_id) tuples
                    is_warmup=False
                )

                # Round-robin distribution to workers
                target_rank = self._worker_ranks[self.out_batch_counter % self._num_workers]
                self.out_batch_counter += 1

                logger.debug(f"[Loadgen Rank: {self.rank}] sending ts_request packet to worker rank {target_rank}, out_batch_counter: {self.out_batch_counter}, num_pairs: {len(outputs_ts)}")

                # Send to worker
                self.request_sender.send_to_worker(data_packet_ts_request)

                # Reset batch accumulation
                self.current_query_ids = []
                self.current_qsl_ids = []

    def issue_queries_offline(self, query_samples):
        """
        Issue queries to workers in Offline scenario.

        In Offline mode, all queries are issued at once. They are batched
        and distributed to workers via round-robin as batches are filled.

        Args:
            query_samples: List of QuerySample objects from LoadGen.
        """
        logger.info(f"[Loadgen Rank: {self.rank}] [Offline Mode] query_samples length: {len(query_samples)}")
        with nvtx.annotate(f"issue_queries", color="blue"):
            self.num_issue_queries_called += 1

            # Process each sample, sending batches as they fill up
            for sample in query_samples:
                self.current_query_ids.append(sample.id)
                self.current_qsl_ids.append(sample.index)

                if len(self.current_query_ids) >= self.batch_size:
                    # Get timestamp-request_id pairs from the query sampler
                    outputs_ts = self.streaming_query_sampler.get_samples_indices(self.current_query_ids)

                    # Create data packet with timestamp-request pairs
                    data_packet_ts_request = MPIDataPacketTSRequest(
                        query_ids=self.current_query_ids.copy(),
                        ts_request_pairs=outputs_ts,  # List of (ts, request_id) tuples
                        is_warmup=False
                    )

                    # Round-robin distribution to workers
                    target_rank = self._worker_ranks[self.out_batch_counter % self._num_workers]
                    self.out_batch_counter += 1

                    logger.debug(f"[Loadgen Rank: {self.rank}] sending ts_request packet to worker rank {target_rank}, out_batch_counter: {self.out_batch_counter}, num_pairs: {len(outputs_ts)}")

                    # Send to worker
                    self.request_sender.send_to_worker(data_packet_ts_request)

                    # Reset batch accumulation
                    self.current_query_ids = []
                    self.current_qsl_ids = []

    def flush_queries(self):
        """
        MLPerf LoadGen callback to flush any remaining queries in the batch buffer.

        Called by LoadGen at the end of a test to ensure all queries are processed,
        even if the final batch is not full. Sends partial batches to workers.
        """
        logger.info(f"[Loadgen Rank: {self.rank}] Flushing queries...")
        if len(self.current_query_ids) > 0:
            # Get timestamp-request pairs for remaining queries
            outputs_ts = self.streaming_query_sampler.get_samples_indices(self.current_query_ids)

            # Create data packet for partial batch
            data_packet_ts_request = MPIDataPacketTSRequest(
                query_ids=self.current_query_ids.copy(),
                ts_request_pairs=outputs_ts,
                is_warmup=False
            )

            # Send to next worker in round-robin
            target_rank = self._worker_ranks[self.out_batch_counter % self._num_workers]
            self.out_batch_counter += 1
            logger.info(f"[Loadgen Rank: {self.rank}] Flushing {len(outputs_ts)} remaining queries to worker {target_rank}")
            self.request_sender.send_to_worker(data_packet_ts_request)

            # Clear batch buffer
            self.current_query_ids = []
            self.current_qsl_ids = []

        # Brief wait to allow final processing to complete
        time.sleep(0.5)
        logger.info(f"[Loadgen Rank: {self.rank}] All queries flushed")

    def benchmark_zmq_latency(self, num_requests: int = 50000):
        """
        Run a latency profiling test (for debugging, not MLPerf benchmarking).

        Sends warmup requests to workers and measures round-trip latency,
        collecting percentile statistics for performance analysis.

        Args:
            num_requests: Number of test requests to send.
        """
        self.test_latency_mode = True
        self._latency_send_times.clear()
        self._latency_results.clear()

        logger.info(f"[Loadgen Rank: {self.rank}] Starting latency test with {num_requests} requests")
        for i in range(num_requests):
            # fake dataset index for benchmark latency
            query_ids = list(range(i * self.batch_size, (i + 1) * self.batch_size))
            outputs_ts = [(j, j) for j in range(self.batch_size)]
            data_packet_ts_request = MPIDataPacketTSRequest(
                query_ids=query_ids,
                ts_request_pairs=outputs_ts,
                is_warmup=True
            )
            # Record send time
            self._latency_send_times[i] = time.perf_counter()
            self.request_sender.send_to_worker(data_packet_ts_request)
            time.sleep(0.0001)

        # Wait for all responses
        timeout = 10  # seconds
        start_wait = time.time()
        while len(self._latency_results) < num_requests:
            if time.time() - start_wait > timeout:
                logger.warning(f"Timeout waiting for responses. Got {len(self._latency_results)}/{num_requests}")
                break
            time.sleep(0.01)

        self._report_latency()

        self.test_latency_mode = False

    def _results_listener(self):
        """
        Background thread that listens for inference results from workers.

        Continuously polls for results from workers and reports them back to
        LoadGen via QuerySamplesComplete. Runs until result_stop_event is set.
        """
        # Ensure garbage collection is disabled for consistent latency
        gc.disable()
        logger.info(f"[Loadgen Rank: {self.rank}] Results listener started")

        while not self.result_stop_event.is_set():
            # Check for incoming results from workers
            if self.request_sender.probe_from_worker():
                data_packet = self.request_sender.receive_from_worker()
                if self.test_latency_mode:
                    self._record_test_latency(data_packet)
                    continue

                self.in_batch_counter += 1
                if self.in_batch_counter % 500 == 0:
                    logger.info(f"[Loadgen Rank: {self.rank}] Received {self.in_batch_counter} batches from workers. data shape: {data_packet.results[0].shape}")

                # Extract results
                preds = data_packet.results[0]
                labels = data_packet.results[1]
                weights = data_packet.results[2]
                ts_idx_list, query_idx_list = self._build_query_metadata(data_packet)

                # Normal mode: Report results back to LoadGen
                report_loadgen_best_perf(
                    data_packet,
                    preds,
                    labels,
                    weights,
                    compute_eval=True if self.mode == "accuracy" else False,
                    ts_idx_list=ts_idx_list,
                    query_idx_list=query_idx_list,
                )
                # Debug mode: Dump tensors to file for accuracy verification
                if self.debug_accuracy:
                    self._record_accuracy(preds, labels, weights)

            else:
                # No results available, sleep briefly to avoid busy-waiting
                time.sleep(0.001)  # 1ms sleep

        logger.info(f"[Loadgen Rank: {self.rank}] Results listener stopped")

        # Verify all batches were received
        assert self.in_batch_counter == self.out_batch_counter, \
            f"Batch count mismatch: received {self.in_batch_counter}, sent {self.out_batch_counter}"

    def _record_test_latency(self, data_packet):
        # Latency profiling mode: Compute round-trip latency
        recv_time = time.perf_counter()
        first_qid = data_packet.query_ids[0]
        request_idx = first_qid // self.batch_size  # Derive request index from query_id pattern
        if request_idx in self._latency_send_times:
            latency_ms = (recv_time - self._latency_send_times[request_idx]) * 1000
            self._latency_results.append((request_idx, latency_ms))

    def _build_query_metadata(
        self, data_packet: MPIDataPacketDSIndex
    ) -> Tuple[List[float], List[float]]:
        num_queries = len(data_packet.query_ids)
        pairs = data_packet.ts_request_pairs
        if not pairs or len(pairs) != num_queries:
            return [-1.0] * num_queries, [-1.0] * num_queries
        start_ts = self.streaming_query_sampler.start_ts
        ts_idx_list = [float(ts - start_ts) for ts, _ in pairs]
        query_idx_list = [float(req_id) for _, req_id in pairs]
        return ts_idx_list, query_idx_list

    def _report_latency(self):
        latencies = [lat for _, lat in self._latency_results]
        logger.info(f"[Latency Test Results] n={len(latencies)}")
        logger.info(f"  Min:    {min(latencies):.3f} ms")
        logger.info(f"  Max:    {max(latencies):.3f} ms")
        logger.info(f"  Avg:    {sum(latencies) / len(latencies):.3f} ms")
        sorted_lat = sorted(latencies)
        p50_idx = int(len(sorted_lat) * 0.5)
        p75_idx = int(len(sorted_lat) * 0.75)
        p95_idx = int(len(sorted_lat) * 0.95)
        p98_idx = int(len(sorted_lat) * 0.98)
        p99_idx = int(len(sorted_lat) * 0.99)
        p999_idx = int(len(sorted_lat) * 0.999)
        logger.info(f"  P50:    {sorted_lat[p50_idx]:.3f} ms")
        logger.info(f"  P75:    {sorted_lat[p75_idx]:.3f} ms")
        logger.info(f"  P95:    {sorted_lat[min(p95_idx, len(sorted_lat) - 1)]:.3f} ms")
        logger.info(f"  P98:    {sorted_lat[min(p98_idx, len(sorted_lat) - 1)]:.3f} ms")
        logger.info(f"  P99:    {sorted_lat[min(p99_idx, len(sorted_lat) - 1)]:.3f} ms")
        logger.info(f"  P99.9:  {sorted_lat[min(p999_idx, len(sorted_lat) - 1)]:.3f} ms")

    def _record_accuracy(self, preds, labels, weights):
        if self._accuracy_dump_file is None:
            self._accuracy_dump_file = open("tensor_dump_mpi_recv.txt", "a")
        f = self._accuracy_dump_file
        f.write(f"iteration: {self.in_batch_counter - 1} ----------\n\n")

        # Output predictions
        f.write(f"iteration: {self.in_batch_counter - 1} mt_target_preds,\n\n")
        f.write(f"{preds.cpu()[0].tolist()[0:128]}\n\n")
        f.write(f"iteration: {self.in_batch_counter - 1} mt_target_preds shape: {preds.cpu()[0].shape}\n\n")

        # Output labels
        f.write(f"iteration: {self.in_batch_counter - 1} mt_target_labels,\n\n")
        f.write(f"{labels.cpu()[0].tolist()[0:128]}\n\n")

        # Output weights
        f.write(f"iteration: {self.in_batch_counter - 1} mt_target_weights,\n\n")
        f.write(f"{weights.cpu()[0].tolist()[0:128]}\n\n\n")

    def shutdown(self):
        """
        Gracefully shutdown the TestRunner and its background threads.

        Stops the results listener thread, waits for it to complete, and
        closes communication channels with workers.
        """
        logger.info(f"[Loadgen Rank: {self.rank}] Shutting down TestRunner...")

        # Signal the results listener thread to stop
        self.result_stop_event.set()

        # Wait for thread to finish (with timeout to avoid hanging)
        if self.results_thread.is_alive():
            logger.info(f"[Loadgen Rank: {self.rank}] Waiting for results listener thread to stop...")
            self.results_thread.join(timeout=2.0)
            if self.results_thread.is_alive():
                logger.warning(f"[Loadgen Rank: {self.rank}] Results listener thread did not stop within timeout")

        # Close communication channels (after thread has stopped)
        self.request_sender.shutdown(num_workers=self._num_workers)

        if self._accuracy_dump_file is not None:
            self._accuracy_dump_file.close()
            self._accuracy_dump_file = None

        logger.info(f"[Loadgen Rank: {self.rank}] TestRunner shutdown complete.")
