"""
Streaming query sampler for MLPerf LoadGen benchmarking.

Provides query sampling strategies for different MLPerf scenarios (Server, Offline)
with support for repeated dataset iterations and timestamp-based query indexing.
"""

import logging
import random
import threading
import time
from typing import List, Optional

import numpy as np

from generative_recommenders.dlrm_v3.datasets.dataset import Samples
from inference_harness.dataset.mlperf_streaming_qsl import DLRMv3StreamingMLPerfDataset
from inference_harness.dataset.util.timer import timer


logger: logging.Logger = logging.getLogger(__name__)


def get_num_queries(
    input_size: Optional[int],
    one_pass_size: int,
    scenario_name: str,
    offline_target_qps: int,
    target_duration: float,
) -> int:
    """
    Determine the number of queries to run based on scenario and settings.

    Args:
        input_size: User-specified query count (None to use defaults).
        one_pass_size: Size of one complete pass through the dataset.
        scenario_name: MLPerf scenario name ('Server' or 'Offline').
        offline_target_qps: Target QPS for offline scenario.
        target_duration: Target duration in milliseconds.

    Returns:
        Number of queries to execute in the benchmark run.
    """
    if scenario_name == "Offline":
        # consistent with https://github.com/mlcommons/inference/blob/8999c4d686f6e4a180da14597c97063fce7c9f33/loadgen/test_settings_internal.cc#L147
        return int(1.1 * target_duration / 1000 * offline_target_qps)
    else:
        if input_size is None:
            return one_pass_size
        return input_size


class StreamingQuerySamplerRef:
    """
    Sampler for streaming dataset
    The execution order is determined by `StreamingQuerySampler.run_order`, not by the QSL or input query ID.
    This ensures that queries are executed according to their timestamp constraints.
    """

    def __init__(
        self,
        ds: DLRMv3StreamingMLPerfDataset,
        dataset_percentage: float,
        scenario_name: str,
        offline_target_qps: int,
        target_duration: float,
        input_queries: Optional[int] = None,
        compute_eval: bool = False,
    ) -> None:
        self.ds: DLRMv3StreamingMLPerfDataset = ds
        self.ds.is_inference = True
        self.inference_ts: int = self.ds.total_ts - self.ds.train_ts
        self.start_ts: int = self.ds.train_ts
        self.dataset_percentage: float = dataset_percentage
        self.num_unique_requests: List[int] = self.get_num_unique_requests(
            warmup_ratio=1.0
        )

        self.num_unique_requests_cumsum: List[int] = np.cumsum(
            self.num_unique_requests
        ).tolist()
        self.total_requests: int = sum(self.num_unique_requests)
        self.run_order: List[List[int]] = self.build_random_exec_order()
        self.ts_idx: int = 0
        self.ts_processed_cnt: int = 0
        self.last_loaded: float = -1.0
        num_queries: int = get_num_queries(
            input_size=input_queries,
            one_pass_size=self.total_requests,
            scenario_name=scenario_name,
            offline_target_qps=offline_target_qps,
            target_duration=target_duration,
        )
        logger.info(f"StreamingQuerySampler constructred to handle {num_queries} queries")
        self.num_queries: int = num_queries
        self.num_repeats: int = (
            max(1, num_queries // self.total_requests) if not compute_eval else 1
        )
        self.remaining_queries: int = (
            num_queries % self.total_requests if not compute_eval else 0
        )
        self._lock = threading.Lock()

    def get_num_unique_requests(self, warmup_ratio: float) -> List[int]:
        """
        Calculate number of unique requests per timestamp.

        Args:
            warmup_ratio: Fraction of users to include in warmup.

        Returns:
            List of request counts per timestamp.
        """
        num_unique_requests = [
            int(
                self.ds.ts_to_users_cumsum[t][-1]
                * self.dataset_percentage
                * warmup_ratio
            )
            for t in range(self.start_ts, self.start_ts + self.inference_ts)
        ]
        return num_unique_requests

    def build_random_exec_order(self) -> List[List[int]]:
        """
        Build randomized execution order for each timestamp.

        Returns:
            List of shuffled index lists, one per timestamp.
        """
        order = []
        for req_size in self.num_unique_requests:
            within_ts_order = list(range(req_size))
            random.shuffle(within_ts_order)
            order.append(within_ts_order)
        return order

    def init_sut(self) -> None:
        """Initialize System Under Test state for a new benchmark run."""
        self.ts_idx = 0
        self.ts_processed_cnt = 0
        self.ds.set_ts(self.start_ts)

    def load_query_samples_warmup(self, query_ids: List[Optional[int]]) -> None:
        """
        Load query samples into memory for the benchmark.

        Args:
            query_ids: List of query identifiers to load.
        """
        length = len(query_ids)
        ts_idx: int = 0
        while self.num_unique_requests_cumsum[ts_idx] < length:
            ts_idx += 1
        for i in range(0, ts_idx):
            self.ds.set_ts(i + self.start_ts)
            self.ds.load_query_samples(self.run_order[i])
        self.ds.set_ts(ts_idx + self.start_ts)
        delta_length = (
            length
            if ts_idx == 0
            else length - self.num_unique_requests_cumsum[ts_idx - 1]
        )
        self.ds.load_query_samples(self.run_order[ts_idx][:delta_length])
        self.init_sut()
        self.last_loaded = time.time()

    def load_query_samples(self, query_ids: List[Optional[int]]) -> None:
        pass

    @timer
    def load_query_samples_multi_processing(self, query_ids: List[Optional[int]]) -> None:
        """
        Load query samples into memory using multiprocessing for faster loading.

        This method loads samples across multiple timestamps using the dataset's
        multiprocessing loader for improved performance.

        Args:
            query_ids: List of query identifiers to load.
        """
        length = len(query_ids)
        ts_idx: int = 0
        while self.num_unique_requests_cumsum[ts_idx] < length:
            ts_idx += 1
            assert ts_idx < len(self.num_unique_requests_cumsum), \
                "query ids length is longer than the total number of requests"

        # Load first N-1 timestamps
        for i in range(0, ts_idx):
            self.ds.set_ts(i + self.start_ts)
            result_at_ts = self.ds.load_query_samples_multi_processing(self.run_order[i])
            self.ds.items_in_memory[i + self.start_ts] = result_at_ts

        # Load last timestamp (potentially partial)
        self.ds.set_ts(ts_idx + self.start_ts)
        delta_length = (
            length if ts_idx == 0 else length - self.num_unique_requests_cumsum[ts_idx - 1]
        )
        result_at_ts = self.ds.load_query_samples_multi_processing(
            self.run_order[ts_idx][:delta_length]
        )
        self.ds.items_in_memory[ts_idx + self.start_ts] = result_at_ts

        self.init_sut()
        self.last_loaded = time.time()

    # @timer
    def load_query_samples_preprocessed(self, preprocessed_dir: str) -> None:
        """
        Load samples from preprocessed numpy arrays (fastest method).

        This bypasses CSV parsing and multiprocessing entirely. All tensors
        are allocated in the main process, ensuring NUMA locality.

        Args:
            preprocessed_dir: Directory containing serialized numpy arrays
                              (created by DLRMv3StreamingMLPerfDataset.serialize_ds)
        """
        self.ds.load_preprocessed_dataset(preprocessed_dir)
        self.init_sut()
        self.last_loaded = time.time()

    def unload_query_samples(self, sample_list: List[int]) -> None:
        """
        Unload query samples from memory.

        Args:
            sample_list: List of sample identifiers to unload.
        """
        self.ds.unload_query_samples(sample_list)

    def get_samples(self, id_list: List[int]) -> List[Samples]:
        """
        Get samples for a batch of queries, handling timestamp boundaries.

        Args:
            id_list: List of query identifiers.

        Returns:
            List of Samples objects, potentially spanning multiple timestamps.
        """
        batch_size: int = len(id_list)
        with self._lock:
            curr_ts_idx: int = self.ts_idx
            curr_ts_unique_requests: int = self.num_unique_requests[curr_ts_idx]
            curr_ts_queries: int = curr_ts_unique_requests * self.num_repeats
            if curr_ts_idx == self.inference_ts - 1:
                curr_ts_queries += self.remaining_queries
            begin_query_idx: int = self.ts_processed_cnt
            end_query_idx: int = min(begin_query_idx + batch_size, curr_ts_queries)
            begin_request_idx: int = begin_query_idx % curr_ts_unique_requests
            end_request_idx: int = end_query_idx % curr_ts_unique_requests
            if begin_query_idx + batch_size >= curr_ts_queries:
                self.ts_idx += 1
                self.ts_processed_cnt = begin_query_idx + batch_size - curr_ts_queries
            else:
                self.ts_processed_cnt = begin_query_idx + batch_size
        # requests of current ts
        outputs: List[Samples] = []
        if end_request_idx > begin_request_idx:
            output: Samples = self.ds.get_samples_with_ts(
                self.run_order[curr_ts_idx][begin_request_idx:end_request_idx],
                curr_ts_idx + self.start_ts,
            )
            outputs.append(output)
        else:
            if begin_request_idx < curr_ts_unique_requests:
                output: Samples = self.ds.get_samples_with_ts(
                    self.run_order[curr_ts_idx][begin_request_idx:],
                    curr_ts_idx + self.start_ts,
                )
                outputs.append(output)
            if end_request_idx > 0:
                output = self.ds.get_samples_with_ts(
                    self.run_order[curr_ts_idx][0:end_request_idx],
                    curr_ts_idx + self.start_ts,
                )
                outputs.append(output)
        # requests of next ts
        if begin_query_idx + batch_size > curr_ts_queries:
            output: Samples = self.ds.get_samples_with_ts(
                self.run_order[curr_ts_idx + 1][
                    : begin_query_idx + batch_size - curr_ts_queries
                ],
                curr_ts_idx + 1 + self.start_ts,
            )
            outputs.append(output)
        return outputs

    # for ZMQ send

    def get_samples_indices(self, id_list: List[int]) -> List[Samples]:
        """
        Get samples index for a batch of queries, handling timestamp boundaries.
        """
        batch_size: int = len(id_list)
        with self._lock:
            curr_ts_idx: int = self.ts_idx
            curr_ts_unique_requests: int = self.num_unique_requests[curr_ts_idx]
            curr_ts_queries: int = curr_ts_unique_requests * self.num_repeats
            if curr_ts_idx == self.inference_ts - 1:
                curr_ts_queries += self.remaining_queries
            begin_query_idx: int = self.ts_processed_cnt
            end_query_idx: int = min(begin_query_idx + batch_size, curr_ts_queries)
            begin_request_idx: int = begin_query_idx % curr_ts_unique_requests
            end_request_idx: int = end_query_idx % curr_ts_unique_requests
            if begin_query_idx + batch_size >= curr_ts_queries:
                self.ts_idx += 1
                self.ts_processed_cnt = begin_query_idx + batch_size - curr_ts_queries
            else:
                self.ts_processed_cnt = begin_query_idx + batch_size
        # requests of current ts
        outputs_ts: List[int] = []
        if end_request_idx > begin_request_idx:
            for i in range(begin_request_idx, end_request_idx):
                outputs_ts.append((curr_ts_idx + self.start_ts, i))
        else:
            if begin_request_idx < curr_ts_unique_requests:
                for i in range(begin_request_idx, curr_ts_unique_requests):
                    outputs_ts.append((curr_ts_idx + self.start_ts, i))
            if end_request_idx > 0:
                for i in range(0, end_request_idx):
                    outputs_ts.append((curr_ts_idx + self.start_ts, i))
        # requests of next ts
        if begin_query_idx + batch_size > curr_ts_queries:
            for i in range(0, begin_query_idx + batch_size - curr_ts_queries):
                outputs_ts.append((curr_ts_idx + 1 + self.start_ts, i))
        return outputs_ts

    def get_item_count(self) -> int:
        """
        Get total number of items in the dataset.

        Returns:
            Total request count across all timestamps.
        """
        return self.total_requests
