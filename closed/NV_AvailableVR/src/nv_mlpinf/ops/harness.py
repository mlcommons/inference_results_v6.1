# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard library imports
import contextlib
import dataclasses as dcls
import importlib
import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import subprocess

# Third-party imports
import nvmitten.json_utils as json
import nvmitten.nvidia.smi as NvSMI
from nvmitten.configurator import bind, autoconfigure, HelpInfo
from nvmitten.pipeline import Operation

# pylint: disable=c-extension-no-member
import mlperf_loadgen as lg

# Local imports
from ..common.constants import Scenario, AuditTest
from ..common.mlcommons.accuracy_checker import check_accuracy
from ..common.mlcommons.compliance import get_audit_verifier
from ..scripts.result_display import populate_accuracy_fields, print_session_results
from ..common.mlcommons.lg_logs import LoadgenLogReader, result_key
from ..common.mlcommons.loadgen import QUERY_METRIC_CONSTRAINTS, LogOutputSettings, LogSettings
from ..common.mlcommons.runner import ScopedQSL, ScopedSUT
from ..common.systems.system_list import DETECTED_SYSTEM
from ..common.workload import Workload
from ..fields import general as general_fields
from ..fields import harness as harness_fields
from ..fields import loadgen as lg_fields
from ..fields import meta as metafields

from .loadgen import LoadgenConfFilesOp


@autoconfigure
@bind(harness_fields.vboost_slider, "value")
class Vboost:
    """Context manager for controlling GPU voltage boost settings.

    This class provides a context manager interface for setting and resetting
    GPU voltage boost settings. It is only supported on Hopper architecture GPUs.

    Attributes:
        value (int): The voltage boost value to set.
        is_supported (bool): Whether voltage boost is supported on the current system.
    """

    def __init__(self, value: int = 0):
        """Initialize the Vboost context manager.

        Args:
            value (int, optional): The voltage boost value to set. Defaults to 0.
        """
        self.value = value
        self.is_supported = any(tag in DETECTED_SYSTEM.extras["tags"] for tag in ("is_hopper", "is_blackwell"))

    def __enter__(self):
        """Set the voltage boost value when entering the context.

        Returns:
            Any: The result of setting the voltage boost, if supported.
        """
        if self.is_supported:
            logging.debug("Setting vboost to %d", self.value)
            try:
                NvSMI.set_vboost(self.value)
            except subprocess.CalledProcessError as e:
                logging.info("WARNING: Failed to set vboost slider. Skipping...")

    def __exit__(self, *args):
        """Reset the voltage boost to 0 when exiting the context."""
        if self.is_supported:
            try:
                NvSMI.set_vboost(0)
            except subprocess.CalledProcessError as e:
                logging.info("WARNING: Failed to reset vboost slider.")


@autoconfigure
@bind(Workload.FIELD)
@bind(general_fields.verbose)
@bind(general_fields.verbose_nvtx)
@bind(lg_fields.test_mode)
class BenchmarkHarnessOp(Operation):
    """Base class for benchmark harness operations.

    This class provides the base functionality for running MLPerf benchmarks,
    including power monitoring, environment setup, and logging.

    Attributes:
        verbose (bool): Whether to enable verbose output.
        verbose_nvtx (bool): Whether to enable NVTX profiling.
        test_mode (str): The test mode to run (e.g., "PerformanceOnly").
        _env_vars (dict): Environment variables for the benchmark.
    """

    @classmethod
    def immediate_dependencies(cls):
        """Get the immediate dependencies of this operation.

        Returns:
            set: Set of operation classes that this operation depends on.
        """
        return {LoadgenConfFilesOp}

    @classmethod
    def output_keys(cls):
        """Get the output keys produced by this operation.

        Returns:
            list: List of output keys.
        """
        return ["log_dir", "result_metadata"]

    def __init__(self,
                 *args,
                 workload: Optional[Workload] = None,
                 verbose: bool = False,
                 verbose_nvtx: bool = False,
                 test_mode: str = "PerformanceOnly",
                 **kwargs):
        """Initialize the benchmark harness operation.

        Args:
            verbose (bool, optional): Whether to enable verbose output. Defaults to False.
            verbose_nvtx (bool, optional): Whether to enable NVTX profiling. Defaults to False.
            test_mode (str, optional): The test mode to run. Defaults to "PerformanceOnly".
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)

        if workload is None:
            raise ValueError("Workload is required for BenchmarkHarnessOp")
        self.wl = workload

        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx
        self.test_mode = test_mode

        self._env_vars = os.environ.copy()

    def prepend_ld_preload(self, so_path):
        """Prepend a shared library to LD_PRELOAD.

        Args:
            so_path (str): Path to the shared library to preload.
        """
        if "LD_PRELOAD" in self._env_vars:
            self._env_vars["LD_PRELOAD"] = ":".join([so_path, self._env_vars["LD_PRELOAD"]])
        else:
            self._env_vars["LD_PRELOAD"] = so_path

        logging.debug("Updated LD_PRELOAD: %s", self._env_vars["LD_PRELOAD"])


    def load_run_results(self):
        """Load and process the results from the benchmark run.

        Returns:
            dict: Dictionary containing the result metadata and log directory.
        """
        log_reader = LoadgenLogReader(self.wl)
        qmc = QUERY_METRIC_CONSTRAINTS[self.wl.scenario]
        rk = result_key(self.wl.benchmark, self.wl.scenario)
        loadgen_query_keys = ["result_validity",
                              rk,
                              "early_stopping_met",
                              qmc.name,
                              "effective_min_duration_ms"]
        # Append QPS for LLMs in Offline
        if self.wl.benchmark.is_llm and self.wl.scenario == Scenario.Offline:
            loadgen_query_keys.append("result_samples_per_second")
        results = log_reader.get_keys(*loadgen_query_keys)

        qmc_measured = float(results[qmc.name])
        satisfies_query_constraint = (qmc_measured >= qmc.val)
        perf_value, perf_metric = log_reader.result_summary(strict_match=False)
        results.update({"system_name": self.wl.submission_system,
                        "base_log_dir": str(self.wl.base_log_dir.absolute()),
                        "detected_system": DETECTED_SYSTEM.summary_description(),
                        "workload_setting_code": self.wl.setting.short,
                        "benchmark_short": self.wl.benchmark.valstr,
                        "benchmark_full": self.wl.submission_benchmark,
                        "scenario": self.wl.scenario.valstr,
                        "avg_power": log_reader.avg_power(),
                        "test_mode": self.test_mode,
                        "scenario_key": rk,
                        "satisfies_query_constraint": satisfies_query_constraint,
                        "true_result_value": perf_value,
                        "true_result_metric": perf_metric})

        # Extra stats for Server and Interactive scenarios
        if self.wl.scenario in (Scenario.Server, Scenario.Interactive) and self.test_mode == "PerformanceOnly":
            serv_lat = log_reader.get_keys("requested_server_ttft_latency",
                                           "result_first_token_99.00_percentile_latency_ns",
                                           "requested_server_tpot_latency",
                                           "result_time_per_output_token_99.00_percentile_ns",
                                           "requested_server_target_latency_ns",
                                           "result_99.00_percentile_latency_ns")

            if serv_lat["requested_server_ttft_latency"]:
                ttft_99 = float(serv_lat["result_first_token_99.00_percentile_latency_ns"])
                ttft_target = float(serv_lat["requested_server_ttft_latency"])
                tpot_99 = float(serv_lat["result_time_per_output_token_99.00_percentile_ns"])
                tpot_target = float(serv_lat["requested_server_tpot_latency"])

                results["latency_usage_ttft"] = ttft_99 / ttft_target
                results["latency_usage_tpot"] = tpot_99 / tpot_target
            else:
                lat_99 = float(serv_lat["result_99.00_percentile_latency_ns"])
                lat_target = float(serv_lat["requested_server_target_latency_ns"])

                results["latency_usage_raw"] = lat_99 / lat_target
        return {"result_metadata": results,
                "log_dir": self.wl.log_dir}


HelpInfo.add_configurator_dependency(BenchmarkHarnessOp, LoadgenLogReader)


class PyHarnessOp(BenchmarkHarnessOp, ScopedSUT):
    """Harness for executing MLPerf benchmarks with Python code.

    This class extends BenchmarkHarnessOp to support running benchmarks using
    Python code, with support for tensor and pipeline parallelism.
    """

    @classmethod
    def immediate_dependencies(cls):
        """Get the immediate dependencies of this operation.

        Returns:
            set: Set of operation classes that this operation depends on.
        """
        return {LoadgenConfFilesOp}

    @classmethod
    def output_keys(cls):
        """Get the output keys produced by this operation.

        Returns:
            list: List of output keys.
        """
        return ["log_dir", "result_metadata"]

    def __init__(self,
                 qsl_cls: Type[ScopedQSL],
                 *args,
                 total_sample_count: Optional[int] = None,
                 **kwargs):
        """Initialize the PyHarnessOp.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.qsl_cls = qsl_cls
        self.total_sample_count = total_sample_count

        self._qsl_inst = None

    def issue_queries(self, query_samples: List[lg.QuerySample]):
        """Issue queries to the SUT.

        Args:
            query_samples: List of query samples to issue.
        """
        raise NotImplementedError("issue_queries is not implemented for PyHarnessOp")

    def flush_queries(self):
        """Flush queries from the SUT.
        """
        raise NotImplementedError("flush_queries is not implemented for PyHarnessOp")

    @contextlib.contextmanager
    def wrap_lg_test(self, scratch_space, dependency_outputs):
        """Context wrapped around lg.StartTestWithLogSettings. Users of this class should override this method to
        perform any setup or teardown necessary for the SUT before and after the test.

        self._qsl_inst will be set to the ScopedQSL instance before the context is entered.

        Yields:
            None: No value should be yielded for this context.
        """
        yield None

    def run(self, scratch_space, dependency_outputs):
        """Run the benchmark using Python code.

        Args:
            scratch_space: The scratch space for temporary files.
            dependency_outputs: Outputs from dependency operations.

        Returns:
            dict: Dictionary containing the result metadata and log directory.
        """
        user_conf = dependency_outputs[LoadgenConfFilesOp]["user_conf"]
        if self.total_sample_count is None:
            total_sample_count = user_conf.performance_sample_count
        else:
            total_sample_count = self.total_sample_count

        lg_settings = dependency_outputs[LoadgenConfFilesOp]["lg_settings"]
        test_settings = lg_settings.to_lg_obj()

        log_settings = LogSettings(LogOutputSettings(self.wl.log_dir)).to_lg_obj()
        self._qsl_inst = self.qsl_cls(total_sample_count, user_conf.performance_sample_count)
        with self._qsl_inst as qsl, \
                self as sut, \
                Vboost():

            with self.wrap_lg_test(scratch_space, dependency_outputs):
                lg.StartTestWithLogSettings(sut, qsl, test_settings, log_settings)
        self._qsl_inst = None
        return self.load_run_results()


HelpInfo.add_configurator_dependency(PyHarnessOp, LogSettings)



@autoconfigure
@bind(Workload.FIELD)
@bind(harness_fields.audit_test)
class ResultSummaryOp(Operation):
    """Operation for generating result summaries from benchmark runs.

    This class handles the creation of metadata JSON files and accuracy checking
    for benchmark results.
    """

    @classmethod
    def immediate_dependencies(cls):
        """Get the immediate dependencies of this operation.

        Returns:
            set: Set of operation classes that this operation depends on.
        """
        return {BenchmarkHarnessOp}

    def __init__(self,
                 *args,
                 workload: Optional[Workload] = None,
                 audit_test: Optional[AuditTest] = None,
                 **kwargs):
        """Initialize the ResultSummaryOp.
        """
        super().__init__(*args, **kwargs)

        if workload is None:
            raise ValueError("Workload is required for ResultSummaryOp")
        self.wl = workload

        self.audit_test = audit_test

    def create_metadata_json(self,
                             log_dir: Path,
                             result_data: Dict[str, Any],
                             append: bool = False):
        """Create or update a metadata JSON file with benchmark results.

        Args:
            log_dir (Path): Directory to store the metadata file.
            result_data (Dict[str, Any]): Dictionary of result data to write.
            append (bool, optional): Whether to append to existing metadata. Defaults to False.
        """
        summary_file = log_dir / "metadata.json"
        if append and summary_file.exists():
            with summary_file.open(mode='r') as f:
                md = json.load(f)
                md.update(result_data)
        else:
            md = result_data

        with summary_file.open(mode="w") as f:
            json.dump(md, f, indent=4, sort_keys=True)

    def run(self, scratch_space, dependency_outputs):
        """Run the result summary operation.

        Args:
            scratch_space: The scratch space for temporary files.
            dependency_outputs: Outputs from dependency operations.

        Returns:
            dict: Dictionary containing the result metadata.
        """
        result_data = dependency_outputs[BenchmarkHarnessOp]["result_metadata"]

        if result_data.get("test_mode") == "AccuracyOnly":
            acc_results = check_accuracy(self.wl)
            populate_accuracy_fields(result_data, acc_results)

        if self.audit_test is not None:
            verifier = get_audit_verifier(self.audit_test)()
            audit_result = verifier.run()
            result_data["audit_result"] = audit_result
            result_data["audit_test"] = self.audit_test.valstr
            result_data["audit_success"] = (audit_result.split('_')[1].upper() == "PASS")

        self.create_metadata_json(self.wl.log_dir, result_data)
        print_session_results(self.wl.base_log_dir)
