
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch
from generative_recommenders.dlrm_v3.datasets.dataset import Samples
import nvtx
from enum import Enum
import zmq
import socket
import struct
import os
import weakref
import gc
import logging

logger = logging.getLogger(__name__)


@dataclass
class MPIDataPacketDSIndex:
    """
    Data packet for dataset index-based requests (legacy format).

    Used for communication between LoadGen and workers when using
    dataset indices. Contains query metadata, batch data, and results.

    Attributes:
        query_ids: LoadGen query IDs for result correlation.
        request_to_send: List of dataset indices to fetch.
        current_ts: Current timestamp in the dataset.
        repeat: Repeat counter for dataset iteration.
        ts_request_pairs: List of (timestamp, request_id) tuples for metadata.
        batch: Sample batch data.
        user_id: User ID tensor (if applicable).
        results: Tuple of (predictions, labels, weights) tensors.
        is_warmup: Whether this is a warmup request.
    """
    query_ids: List[int] = None
    request_to_send: List[int] = None
    current_ts: int = None
    repeat: int = None
    ts_request_pairs: Optional[List[Tuple[int, int]]] = None
    batch: Samples = None
    user_id: torch.Tensor = None
    results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None
    is_warmup: bool = False


@dataclass
class MPIDataPacketTSRequest:
    """
    New data packet format for requests with timestamp-request_id pairs.

    Each request is a list of N tuples: [(ts1, req_id1), (ts2, req_id2), ...]
    where N is the batch size.

    Attributes:
        query_ids: List of LoadGen query IDs for correlation
        ts_request_pairs: List of (timestamp, request_id) tuples
        is_warmup: Whether this is a warmup request
    """
    query_ids: List[int] = None
    ts_request_pairs: List[Tuple[int, int]] = None  # [(ts, request_id), ...]
    batch: Samples = None
    user_id: torch.Tensor = None
    results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None
    is_warmup: bool = False
    transfer_done_event: torch.cuda.Event = None  # Signals when data prep is complete


class CommunicatorType(Enum):
    """
    Enum for communication backend types.

    MPI: MPI-based communication (not yet implemented).
    ZMQ: ZeroMQ-based communication (sharded for scalability).
    """
    MPI = "mpi"
    ZMQ = "zmq"


# Default port configuration for ZMQ communication
BASED_REQUEST_PORT = 5000
BASED_RESULT_PORT = 5100


@dataclass
class ZMQRequestSenderShardedConfig:
    """
    Configuration for sharded ZMQ communication.

    Attributes:
        loadgen_hostname: Hostname of the LoadGen node.
        num_shards: Number of socket pair shards (typically equals num_nodes).
        batch_size: Batch size for inference requests.
        num_preds: Number of predictions per sample.
        mode: Operating mode ("performance" or "accuracy").
        gpus_per_node: Number of GPUs per node (for shard calculation).
        base_request_port: Starting port for request sockets.
        base_result_port: Starting port for result sockets.
    """
    loadgen_hostname: str
    num_shards: int
    batch_size: int
    num_preds: int
    mode: str
    gpus_per_node: int
    base_request_port: int = BASED_REQUEST_PORT
    base_result_port: int = BASED_RESULT_PORT


class ZMQRequestSenderSharded:
    """
    Sharded ZMQ-based request sender for scalable multi-node communication.

    Instead of 1 PUSH -> N PULL (fan-out) and N PUSH -> 1 PULL (fan-in bottleneck),
    this creates multiple socket pairs:
    - LoadGen: N_shards PUSH sockets + N_shards PULL sockets
    - Workers: Each worker connects to its assigned shard based on node_id

    This reduces fan-in from 32:1 to 4:1 (with 4 GPUs per node, 8 shards for 8 nodes).

    Architecture (8 nodes, 4 GPUs each = 32 workers):
    - LoadGen binds to ports: 5555-5562 (request), 5656-5663 (result)
    - Node 0 workers connect to port 5555/5656
    - Node 1 workers connect to port 5556/5657
    - ... etc

    Benefits:
    - Reduces socket contention (4 workers per socket instead of 32)
    - Better TCP connection utilization
    - Parallel polling on LoadGen side
    - Round-robin request distribution across shards
    """

    def __init__(
        self,
        is_loadgen: bool = False,
        shard_id: int = 0,    # Which shard this worker belongs to (typically = node_id)
        rank: int = 0,
        config: ZMQRequestSenderShardedConfig = None,
    ):
        """
        Initialize sharded ZMQ sender.

        Args:
            is_loadgen: True if this is the LoadGen process
            loadgen_hostname: Hostname of LoadGen (required for workers)
            base_request_port: Starting port for request sockets
            base_result_port: Starting port for result sockets
            num_shards: Number of socket pairs (typically = num_nodes)
            shard_id: Which shard this worker belongs to (workers only)
            batch_size: Batch size for pre-allocated buffers
            num_preds: Number of predictions per sample
            mode: "performance" or "accuracy"
            rank: Global rank of this process
            gpus_per_node: GPUs per node for shard calculation
        """
        self.is_loadgen = is_loadgen
        self.rank = rank
        self.shard_id = shard_id
        self.local_hostname = socket.gethostname()

        self.loadgen_hostname = config.loadgen_hostname
        self.batch_size = config.batch_size
        self.num_preds = config.num_preds
        self.mode = config.mode
        self.num_shards = config.num_shards
        self.gpus_per_node = config.gpus_per_node
        self.base_request_port = config.base_request_port
        self.base_result_port = config.base_result_port

        # ZMQ context with multiple I/O threads for TCP
        self.context = zmq.Context(io_threads=max(4, config.num_shards))

        # Round-robin counter for LoadGen request distribution
        self._rr_counter = 0

        if is_loadgen:
            self._setup_loadgen_sharded(config.base_request_port, config.base_result_port, config.num_shards)
        else:
            if config.loadgen_hostname is None:
                raise ValueError("Workers must specify loadgen_hostname")
            self._setup_worker_sharded(config.loadgen_hostname, config.base_request_port, config.base_result_port, shard_id)

    def _setup_loadgen_sharded(self, base_request_port: int, base_result_port: int, num_shards: int):
        """
        Set up LoadGen with multiple socket pairs for sharded communication.

        Creates one PUSH socket (for requests) and one PULL socket (for results)
        per shard. This enables parallel communication with reduced fan-in.

        Args:
            base_request_port: Starting port for request sockets.
            base_result_port: Starting port for result sockets.
            num_shards: Number of socket pair shards to create.
        """
        self.request_sockets = []  # List of PUSH sockets (one per shard)
        self.result_sockets = []   # List of PULL sockets (one per shard)

        # Create one socket pair per shard
        for i in range(num_shards):
            logger.debug(f"[LoadGen Rank: {self.rank}] Setting up shard {i} with request port {base_request_port + i} and result port {base_result_port + i}")
            request_port = base_request_port + i
            result_port = base_result_port + i

            # PUSH socket for sending requests to shard i
            req_socket = self.context.socket(zmq.PUSH)
            req_socket.setsockopt(zmq.SNDHWM, 100_000)
            req_socket.setsockopt(zmq.LINGER, 0)
            req_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            req_socket.bind(f"tcp://0.0.0.0:{request_port}")
            self.request_sockets.append(req_socket)

            # PULL socket for receiving results from shard i
            res_socket = self.context.socket(zmq.PULL)
            res_socket.setsockopt(zmq.RCVHWM, 100_000)
            res_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            res_socket.bind(f"tcp://0.0.0.0:{result_port}")
            self.result_sockets.append(res_socket)

        # Poller for all result sockets
        self.poller = zmq.Poller()
        for sock in self.result_sockets:
            self.poller.register(sock, zmq.POLLIN)

        logger.info(f"[LoadGen Rank: {self.rank}] Listening on {self.local_hostname}:"
                    f"{base_request_port}-{base_request_port + num_shards - 1}/"
                    f"{base_result_port}-{base_result_port + num_shards - 1} ({num_shards} shards)")

    def _setup_worker_sharded(self, loadgen_hostname: str, base_request_port: int, base_result_port: int, shard_id: int):
        """
        Set up worker connection to its assigned shard on LoadGen.

        Workers on the same node share a shard_id, connecting to the same
        socket pair to reduce communication fan-in.

        Args:
            loadgen_hostname: Hostname of the LoadGen node.
            base_request_port: Starting port for request sockets.
            base_result_port: Starting port for result sockets.
            shard_id: Shard identifier for this worker (typically node_id).
        """
        request_port = base_request_port + shard_id
        result_port = base_result_port + shard_id

        # PULL socket: Receive requests from LoadGen (shard-specific)
        self.request_socket = self.context.socket(zmq.PULL)
        self.request_socket.setsockopt(zmq.RCVHWM, 100_000)
        self.request_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.request_socket.connect(f"tcp://{loadgen_hostname}:{request_port}")

        # PUSH socket: Send results to LoadGen (shard-specific)
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.setsockopt(zmq.SNDHWM, 100_000)
        self.result_socket.setsockopt(zmq.LINGER, 1000)
        self.result_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.result_socket.connect(f"tcp://{loadgen_hostname}:{result_port}")

        # Poller for non-blocking checks
        self.poller = zmq.Poller()
        self.poller.register(self.request_socket, zmq.POLLIN)

        logger.info(f"[Worker Comm: {self.rank}] {self.local_hostname} connected to "
                    f"{loadgen_hostname}:{request_port}/{result_port} (shard {shard_id})")

    # ==================== LOADGEN METHODS ====================

    def send_to_worker(self, data: MPIDataPacketTSRequest) -> None:
        """
        LoadGen sends ts_request to workers using round-robin across shards.

        Uses the new timestamp-request_id pair format.

        Args:
            data: MPIDataPacketTSRequest with query_ids and ts_request_pairs
            dest_rank: Destination rank (unused in sharded ZMQ, kept for API compatibility)
            src_rank: Source rank (unused in sharded ZMQ, kept for API compatibility)
        """
        msg = self._serialize_request(data)

        # Round-robin across shards
        shard_idx = self._rr_counter % self.num_shards
        self._rr_counter += 1

        self.request_sockets[shard_idx].send(msg)

    def send_to_worker_targeted(self, data: MPIDataPacketTSRequest, target_shard: int) -> None:
        """
        LoadGen sends ts_request to a specific shard (for locality-aware scheduling).

        Args:
            data: MPIDataPacketTSRequest with query_ids and ts_request_pairs
            target_shard: Target shard index
        """
        msg = self._serialize_request(data)
        self.request_sockets[target_shard].send(msg)

    def probe_from_worker(self) -> bool:
        """
        Check if any result is available from any worker shard (non-blocking).

        Returns:
            bool: True if at least one result is available, False otherwise.
        """
        events = dict(self.poller.poll(timeout=0))
        return len(events) > 0

    def receive_from_worker(self) -> MPIDataPacketDSIndex:
        """
        LoadGen receives result from any worker (any shard).

        Uses fair polling to avoid starvation.
        """
        with nvtx.annotate(f"poll for events", color="green"):
            # Poll all result sockets
            events = dict(self.poller.poll(timeout=0))
        with nvtx.annotate(f"check for events", color="green"):
            for sock in self.result_sockets:
                if sock in events:
                    msg = sock.recv(zmq.NOBLOCK)
                    return self._deserialize_result(msg, self.mode)

        # Fallback: blocking recv on first socket (shouldn't reach here normally)
        msg = self.result_sockets[0].recv()
        return self._deserialize_result(msg, self.mode)

    # ==================== WORKER METHODS ====================

    def probe_from_loadgen(self) -> bool:
        """
        Check if a request is available from LoadGen (non-blocking).

        Returns:
            bool: True if a request is available, False otherwise.
        """
        events = dict(self.poller.poll(timeout=0))
        return self.request_socket in events

    def receive_from_loadgen(self) -> MPIDataPacketTSRequest:
        """
        Worker receives ts_request from LoadGen.

        Uses the new timestamp-request_id pair format.

        Args:
            src: Source rank (unused in ZMQ, kept for API compatibility)
            status: Status object (unused in ZMQ, kept for API compatibility)

        Returns:
            MPIDataPacketTSRequest or None if shutdown signal received
        """
        msg = self.request_socket.recv()

        # Check for shutdown signal
        if msg == b'__SHUTDOWN__':
            return None

        return self._deserialize_request(msg)

    def send_to_loadgen(self, data: MPIDataPacketDSIndex) -> None:
        """
        Send inference results back to LoadGen.

        Args:
            data: Data packet containing query IDs and inference results.
        """
        msg = self._serialize_result(data, self.mode)
        self.result_socket.send(msg)

    # ==================== SERIALIZATION (same as original) ====================

    @staticmethod
    def _serialize_request(data: MPIDataPacketTSRequest) -> bytes:
        """
        Serialize request with timestamp-request_id pairs.

        Format:
        - Header: [num_query_ids (4B), num_pairs (4B), is_warmup (1B)]
        - query_ids array: [qid1, qid2, ...] (int64)
        - ts_request_pairs flattened: [ts1, req1, ts2, req2, ...] (int64)

        Args:
            data: MPIDataPacketTSRequest with query_ids and ts_request_pairs

        Returns:
            Serialized bytes
        """
        query_ids = np.array(data.query_ids, dtype=np.int64)
        num_pairs = len(data.ts_request_pairs)
        is_warmup = 1 if data.is_warmup else 0

        # Flatten the list of tuples into a single array [ts1, req1, ts2, req2, ...]
        flattened_pairs = np.empty(num_pairs * 2, dtype=np.int64)
        for i, (ts, req_id) in enumerate(data.ts_request_pairs):
            flattened_pairs[i * 2] = ts
            flattened_pairs[i * 2 + 1] = req_id

        parts = [
            struct.pack('!IIB', len(query_ids), num_pairs, is_warmup),
            query_ids.tobytes(),
            flattened_pairs.tobytes()
        ]
        return b''.join(parts)

    @staticmethod
    def _deserialize_request(msg: bytes) -> MPIDataPacketTSRequest:
        """
        Deserialize request with timestamp-request_id pairs.

        Format matches _serialize_ts_request:
        - Header: [num_query_ids (4B), num_pairs (4B), is_warmup (1B)]
        - query_ids array: [qid1, qid2, ...] (int64)
        - ts_request_pairs flattened: [ts1, req1, ts2, req2, ...] (int64)

        Args:
            msg: Serialized bytes

        Returns:
            MPIDataPacketTSRequest with query_ids and ts_request_pairs
        """
        # Parse header
        num_query_ids, num_pairs, is_warmup = struct.unpack_from('!IIB', msg, 0)
        offset = 9  # 4 + 4 + 1 bytes

        # Parse query_ids
        query_ids = np.frombuffer(msg, dtype=np.int64, offset=offset, count=num_query_ids)
        offset += num_query_ids * 8

        # Parse flattened ts_request_pairs
        flattened_pairs = np.frombuffer(msg, dtype=np.int64, offset=offset, count=num_pairs * 2)

        # Reconstruct list of tuples
        ts_request_pairs = [(int(flattened_pairs[i * 2]), int(flattened_pairs[i * 2 + 1]))
                            for i in range(num_pairs)]

        return MPIDataPacketTSRequest(
            query_ids=query_ids.tolist(),
            ts_request_pairs=ts_request_pairs,
            is_warmup=bool(is_warmup)
        )

    @staticmethod
    def _serialize_result(data: MPIDataPacketDSIndex, mode: str = "performance") -> bytes:
        """Serialize result (query_ids + tensor).

        Args:
            data: MPIDataPacketDSIndex with query_ids and results tuple (preds, labels, weights)
            mode: "performance" or "accuracy". In accuracy mode, labels and weights are included.
        """
        query_ids = np.array(data.query_ids, dtype=np.int64)
        ts_request_pairs = data.ts_request_pairs or []
        num_pairs = len(ts_request_pairs)

        results_tensor = data.results[0].detach().cpu()
        results_arr = results_tensor.view(torch.uint16).numpy()
        results_arr = np.ascontiguousarray(results_arr)

        if mode == "accuracy":
            # labels and weights are float32
            labels_tensor = data.results[1].detach().cpu()
            labels_arr = labels_tensor.numpy().astype(np.float32)
            labels_arr = np.ascontiguousarray(labels_arr)
            weights_tensor = data.results[2].detach().cpu()
            weights_arr = weights_tensor.numpy().astype(np.float32)
            weights_arr = np.ascontiguousarray(weights_arr)
        else:
            labels_arr = None
            weights_arr = None
        # Header: num_query_ids, dim0, dim1, num_pairs (assume 2D shape)
        flattened_pairs = np.empty(num_pairs * 2, dtype=np.int64)
        for i, (ts, req_id) in enumerate(ts_request_pairs):
            flattened_pairs[i * 2] = ts
            flattened_pairs[i * 2 + 1] = req_id
        parts = [
            struct.pack(
                '!IIII',
                len(query_ids),
                results_arr.shape[0],
                results_arr.shape[1],
                num_pairs,
            ),
            query_ids.tobytes(),
            flattened_pairs.tobytes(),
            results_arr.tobytes(),
        ]

        if mode == "accuracy":
            # Append labels and weights (same 2D shape as results)
            parts.append(labels_arr.tobytes())
            parts.append(weights_arr.tobytes())

        return b''.join(parts)

    @staticmethod
    def _deserialize_result(msg: bytes, mode: str = "performance") -> MPIDataPacketDSIndex:
        """Deserialize result.

        Args:
            msg: Serialized bytes from _serialize_result
            mode: "performance" or "accuracy". Must match the mode used during serialization.
        """
        offset = 0
        # Header: num_query_ids, dim0, dim1, num_pairs (2D shape), 16 bytes total
        num_ids, dim0, dim1, num_pairs = struct.unpack_from('!IIII', msg, offset)
        offset += 16

        query_ids = np.frombuffer(msg, dtype=np.int64, offset=offset, count=num_ids)
        offset += num_ids * 8  # 8 bytes per integer

        ts_request_pairs: Optional[List[Tuple[int, int]]] = None
        if num_pairs > 0:
            flattened_pairs = np.frombuffer(
                msg, dtype=np.int64, offset=offset, count=num_pairs * 2
            )
            offset += num_pairs * 16  # 2 * int64 per pair
            ts_request_pairs = [
                (int(flattened_pairs[i * 2]), int(flattened_pairs[i * 2 + 1]))
                for i in range(num_pairs)
            ]

        # result is bfloat16, 2 bytes per value
        tensor_size = dim0 * dim1
        results_arr = np.frombuffer(msg, dtype=np.uint16, offset=offset, count=tensor_size)
        results_arr = results_arr.reshape((dim0, dim1))
        results_tensor = torch.from_numpy(results_arr.copy()).view(torch.bfloat16)
        offset += tensor_size * 2  # uint16 is 2 bytes

        # label and weights are fp32
        labels_tensor = None
        weights_tensor = None
        if mode == "accuracy":
            # Deserialize labels (float32, same shape as results)
            labels_arr = np.frombuffer(msg, dtype=np.float32, offset=offset, count=tensor_size)
            labels_arr = labels_arr.reshape((dim0, dim1))
            labels_tensor = torch.from_numpy(labels_arr.copy())
            offset += tensor_size * 4  # float32 is 4 bytes

            # Deserialize weights (float32, same shape as results)
            weights_arr = np.frombuffer(msg, dtype=np.float32, offset=offset, count=tensor_size)
            weights_arr = weights_arr.reshape((dim0, dim1))
            weights_tensor = torch.from_numpy(weights_arr.copy())

        return MPIDataPacketDSIndex(
            query_ids=query_ids.tolist(),
            ts_request_pairs=ts_request_pairs,
            results=(results_tensor, labels_tensor, weights_tensor),
        )

    # ==================== SHUTDOWN ====================

    def shutdown(self, num_workers: int = 72):
        """
        Gracefully shutdown ZMQ communication channels.

        LoadGen: Sends shutdown signals to all workers across all shards,
                 then closes all sockets.
        Workers: Closes request and result sockets.

        Args:
            num_workers: Total number of worker processes (for LoadGen only).
        """
        if self.is_loadgen:
            # Send shutdown signal to each shard
            workers_per_shard = num_workers // self.num_shards
            for shard_idx, req_socket in enumerate(self.request_sockets):
                # Send shutdown messages (extra for safety)
                for _ in range(workers_per_shard + 1):
                    try:
                        req_socket.send(b'__SHUTDOWN__', zmq.DONTWAIT)
                    except zmq.Again:
                        pass

            # Close all sockets
            for sock in self.request_sockets:
                sock.close()
            for sock in self.result_sockets:
                sock.close()
        else:
            # Worker: Close sockets
            self.request_socket.close()
            self.result_socket.close()

        # Terminate ZMQ context
        self.context.term()


# Environment variable to enable garbage collection profiling.
# Set to "1" to enable recording of garbage collection events during profiling.
PROFILE_RECORD_GC_ENV_VAR_NAME = "RECORD_GC"


class _GCNvtxHandle:
    """Handle object for GC NVTX watcher to keep it alive."""


# Singleton for the GC NVTX watcher handle.
_gc_watcher_handle: Optional[_GCNvtxHandle] = None


def _setup_gc_nvtx_profiling() -> Optional[_GCNvtxHandle]:
    """
    Set up NVTX range markers for Python garbage collection events (singleton).
    This helps in profiling to visualize when GC occurs during execution.

    This function is called automatically at module import time. The environment
    variable TLLM_PROFILE_RECORD_GC must be set before importing this module.

    This is an internal function and should not be called directly by users.

    Returns:
        _GCNvtxHandle or None: A handle object that keeps the GC callback alive,
                               or None if GC profiling is not enabled.
    """
    global _gc_watcher_handle

    # Return existing handle if already initialized
    if _gc_watcher_handle is not None:
        return _gc_watcher_handle

    enabled = os.environ.get(PROFILE_RECORD_GC_ENV_VAR_NAME, None)
    if not enabled:
        return None

    def gc_callback(phase, info):
        if phase == "start":
            # Use range_push/range_pop (more reliable than range_start/range_end)
            print(f"GC started gen{info['generation']}")
            torch.cuda.nvtx.range_push(f"Python GC gen{info['generation']}")
        elif phase == "stop":
            print(f"GC stopped gen{info['generation']}")
            torch.cuda.nvtx.range_pop()

    gc.callbacks.append(gc_callback)

    def gc_cleanup(callback):
        try:
            gc.callbacks.remove(callback)
        except ValueError:
            pass

    handle = _GCNvtxHandle()
    weakref.finalize(handle, gc_cleanup, gc_callback)

    _gc_watcher_handle = handle
    return handle


# Initialize GC NVTX profiling singleton at module import time
_setup_gc_nvtx_profiling()
