#!/usr/bin/env python3
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


__doc__ = """NVIDIA's MLPerf Inference Benchmark submission code. NVIDIA's implementation runs in 2 phases.

The first phase is 'engine generation', which builds a TensorRT Engine using TensorRT, a Deep Learning Inference
performance optimization SDK by NVIDIA. This only applies to NVIDIA accelerator-based workloads.

The second phase is a 'harness run', which launches the generated TensorRT engine in a server-like harness that
accepts input from LoadGen (MLPerf Inference's official Load Generator), runs the inference with the engine, and reports
the output back to LoadGen.

More about the MLPerf Inference Benchmark and NVIDIA's submission implementation can be found in the README.md for this
project.
"""
from nv_mlpinf import G_BENCHMARK_MODULES
import argparse
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import atexit
import sys
from typing import List, Optional, Tuple

from nvmitten.configurator import (
    Configuration,
    ConfigurationIndex,
    Field,
    HelpInfo,
    autoconfigure,
    bind,
)
from nvmitten.importer import ScopedImporter
from nvmitten.pipeline import Pipeline, ScratchSpace
from nvmitten.system.system import System

import time

import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.gen_engines as builder_fields
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.fields.meta as metafields
import nv_mlpinf.fields.general as general_fields
import nv_mlpinf.fields.loadgen as lg_fields
import nv_mlpinf.ops as Ops

from nv_mlpinf.common import logging
from nv_mlpinf.common.systems.system_list import DETECTED_SYSTEM, apply_system_name_override
from nv_mlpinf.common.workload import Workload
from nv_mlpinf.common.mlcommons.compliance import get_audit_verifier, set_audit_conf
from nv_mlpinf.llmlib.config import HarnessConfig
from nv_mlpinf.llmlib.launch_server import RunTrtllmServeOp
from nv_mlpinf.scripts.result_display import print_session_results



def conf_import_base(benchmark: C.Benchmark, scenario: C.Scenario, system_id: str) -> str:
    return f"{benchmark.substitute_hyphen_for_underscore}/{system_id}/{benchmark.default_serving_framework.valstr}/{scenario.valstr}"

def import_benchmark_module(config_dir: Path, config_index: ConfigurationIndex, benchmark: C.Benchmark, scenario: C.Scenario, action: C.Action, system_id: str):
    imp_path = "server" if action == C.Action.RunLLMServer else "harness"
    import_base = config_dir / conf_import_base(benchmark, scenario, system_id)
    p = import_base / f"{imp_path}.py"

    if not p.exists():
        raise FileNotFoundError(
            f"No config found for benchmark='{benchmark.valstr}' scenario='{scenario.valstr}'.\n"
            f"  Expected config: {p}\n"
            f"  Add a config file under config_dir={config_dir}."
        )

    logging.info(f"Loading configs from {p}")

    with ScopedImporter([import_base] + sys.path):
        config_index.load_module(imp_path, prefix=[system_id, benchmark, scenario])
    return sys.modules.get(imp_path)



@autoconfigure
@bind(metafields.action)
@bind(metafields.benchmarks)
@bind(metafields.scenarios)
@bind(metafields.accuracy_target)
@bind(metafields.power_setting)
@bind(general_fields.config_dir)
@bind(harness_fields.audit_test)
@bind(general_fields.show_help, "show_help")
@bind(general_fields.verbose)
@bind(general_fields.verbose_nvsmi)
@bind(general_fields.log_dir)
@bind(lg_fields.test_mode)
class MainRunner:
    def __init__(self,
                 system: System,
                 action: C.Action = None,
                 benchmarks: List[C.Benchmark] = None,
                 scenarios: List[C.Scenario] = None,
                 accuracy_target: C.AccuracyTarget = C.AccuracyTarget(0.99),
                 power_setting: C.PowerSetting = C.PowerSetting.MaxP,
                 show_help: bool = False,
                 config_dir: os.PathLike = paths.PROJECT_BASE_DIR / "configs",
                 audit_test: Optional[C.AuditTest] = None,
                 verbose: bool = False,
                 verbose_nvsmi: bool = False,
                 log_dir: os.PathLike = paths.BUILD_DIR / "logs" / "default",
                 test_mode: str = 'PerformanceOnly'):
        assert action is not None, "No action specified"
        if action not in (C.Action.ShowPaths, C.Action.DisplayResults):
            assert benchmarks is not None, "No benchmarks specified"
            assert scenarios is not None, "No scenarios specified"

        self.system = system
        self.system_id = system.extras["id"]
        self.action = action
        self.benchmarks = benchmarks
        self.scenarios = scenarios

        self.accuracy_target = accuracy_target
        self.power_setting = power_setting

        self.config_dir = config_dir
        self.audit_test = audit_test
        self.test_mode = test_mode

        assert not (audit_test is not None and test_mode == 'AccuracyOnly'), \
            f"test_mode=AccuracyOnly cannot be combined with audit_test={audit_test.valstr if audit_test else None}. Audit tests must use PerformanceOnly mode."

        self.show_help = show_help
        self.verbose = verbose
        self.verbose_nvsmi = verbose_nvsmi
        self.log_dir = log_dir
        self.nvidia_smi_process = None
        self.nvidia_smi_csv_file = None
        self.config_index = ConfigurationIndex()
        if self.action in (C.Action.ShowPaths, C.Action.DisplayResults):
            return
        self._configs = {}
        for benchmark in self.benchmarks:
            for scenario in self.scenarios:
                module = import_benchmark_module(self.config_dir, self.config_index, benchmark, scenario, self.action, self.system_id)
                workload_setting = C.WorkloadSetting(harness_type=benchmark.default_harness_type,
                                                     accuracy_target=self.accuracy_target,
                                                     power_setting=self.power_setting)
                keyspace = [self.system_id, benchmark, scenario, workload_setting]
                config = self.config_index.get(keyspace)
                if config is None:
                    raise ValueError(f"Config not found for current system. Please check if the config file is correct.")
                if (accuracy_overrides := getattr(module, 'ACCURACY_OVERRIDES', None)) and self.test_mode == 'AccuracyOnly':
                    config = self._accuracy_override(config, accuracy_overrides, workload_setting)
                elif (compliance_overrides := getattr(module, 'COMPLIANCE_OVERRIDES', None)) and self.audit_test is not None:
                    config = self._compliance_override(config, compliance_overrides, workload_setting)
                wl = Workload.from_fields(benchmark,
                                          scenario,
                                          system=self.system,
                                          setting=workload_setting)
                config[Workload.FIELD] = wl
                config[general_fields.log_dir] = self.log_dir
                config[general_fields.data_dir] = paths.DATA_DIR
                config[general_fields.preprocessed_data_dir] = paths.PREPROCESSED_DATA_DIR
                if self.action == C.Action.GenerateEngines:
                    config[builder_fields.force_build_engines] = True
                
                
                self._configs[(benchmark, scenario)] = config

    def _start_nvidia_smi_monitoring(self):
        """Start nvidia-smi monitoring if verbose_nvsmi is enabled."""
        if not self.verbose_nvsmi:
            return

        # Get polling interval from environment variable (default: 200ms)
        polling_interval_ms = int(os.environ.get('NVSMI_REFRESH_RATE', 200))

        # Import nvidia_smi_csv_keys from the GPU fields script
        scripts_path = paths.PROJECT_BASE_DIR / "scripts" / "perf_monitor"
        sys.path.insert(0, str(scripts_path))
        try:
            from nvsmi_gpu_fields import nvidia_smi_csv_keys
            all_fields = list(nvidia_smi_csv_keys.keys())
        except ImportError:
            logging.warning("Could not import nvidia-smi GPU fields script, using default fields")
            all_fields = ["timestamp", "pci.bus_id", "power.draw", "utilization.gpu", "temperature.gpu"]
        finally:
            if str(scripts_path) in sys.path:
                sys.path.remove(str(scripts_path))

        # Create log directory if it doesn't exist
        log_dir_path = Path(self.log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Prepare nvidia-smi command
        csv_file = log_dir_path / "nvidia_smi_monitor.csv"
        self.nvidia_smi_csv_file = csv_file  # Store for later plotting
        cmd = [
            "nvidia-smi",
            "--format=csv",
            f"--loop-ms={polling_interval_ms}",
            f"--filename={csv_file}",
            f"--query-gpu={','.join(all_fields)}"
        ]

        try:
            logging.info(f"Starting nvidia-smi monitoring with {polling_interval_ms}ms polling interval, output: {csv_file}")
            self.nvidia_smi_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            atexit.register(self._stop_nvidia_smi_monitoring)
        except Exception as e:
            logging.warning(f"Failed to start nvidia-smi monitoring: {e}")
            self.nvidia_smi_process = None

    def _stop_nvidia_smi_monitoring(self):
        """Stop nvidia-smi monitoring process and generate plots."""
        if self.nvidia_smi_process is not None:
            try:
                logging.info("Stopping nvidia-smi monitoring...")
                self.nvidia_smi_process.terminate()
                # Wait a bit for graceful termination
                try:
                    timeout = int(os.getenv('NVSMI_KILL_TIMEOUT', 5))
                    self.nvidia_smi_process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logging.warning("nvidia-smi process did not terminate gracefully, killing it")
                    self.nvidia_smi_process.kill()
                self.nvidia_smi_process = None
            except Exception as e:
                logging.warning(f"Error stopping nvidia-smi monitoring: {e}")

        # Generate plots if CSV file was created
        if self.verbose_nvsmi and self.nvidia_smi_csv_file is not None:
            self._generate_nvsmi_plots()

    def _generate_nvsmi_plots(self):
        """Generate plots from nvidia-smi CSV data."""
        if not self.nvidia_smi_csv_file.exists():
            logging.warning(f"nvidia-smi CSV file not found: {self.nvidia_smi_csv_file}")
            return

        # Give nvidia-smi time to flush data
        time.sleep(2)

        # Check if CSV file has content
        try:
            file_size = self.nvidia_smi_csv_file.stat().st_size
            if file_size == 0:
                logging.warning(f"nvidia-smi CSV file is empty: {self.nvidia_smi_csv_file}")
                return

            # Validate CSV has data rows (not just headers)
            with open(self.nvidia_smi_csv_file, 'r') as f:
                lines = f.readlines()

            if len(lines) < 2:
                logging.warning(f"nvidia-smi CSV file has insufficient data (only {len(lines)} lines): {self.nvidia_smi_csv_file}")
                return

            # Check for GPU identifier columns
            header_line = lines[0].strip().lower()
            has_gpu_identifier = any(col in header_line for col in ['uuid', 'index'])
            if not has_gpu_identifier:
                logging.warning(f"nvidia-smi CSV file missing GPU identifier columns. Header: {lines[0][:200]}")
                return

            logging.info(f"nvidia-smi CSV file has {len(lines)} lines, proceeding with plot generation")

        except Exception as e:
            logging.warning(f"Error validating nvidia-smi CSV file: {e}")
            return

        try:
            # Install plotting requirements first
            logging.info("Installing plotting requirements...")
            plot_requirements = paths.PROJECT_BASE_DIR / "scripts" / "plot" / "requirements.txt"
            install_cmd = f"python3 -m pip install -q -r {plot_requirements}"
            subprocess.run(install_cmd, shell=True, check=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Prepare output directory
            log_dir_path = Path(self.log_dir)
            output_dir = log_dir_path / "nvsmi_plots"

            # Run plotting script
            logging.info(f"Generating nvidia-smi plots from {self.nvidia_smi_csv_file}...")
            plot_script = paths.PROJECT_BASE_DIR / "scripts" / "plot" / "nvsmi_csv.py"
            plot_cmd = [
                "python3",
                str(plot_script),
                str(self.nvidia_smi_csv_file),
                "-o", str(output_dir),
            ]

            result = subprocess.run(plot_cmd, check=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)

            logging.info(f"nvidia-smi plots generated successfully in {output_dir}")
            if result.stdout:
                logging.debug(f"Plot generation output:\n{result.stdout}")

        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to generate nvidia-smi plots: {e}")
            if e.stderr:
                logging.warning(f"Error output: {e.stderr}")
        except Exception as e:
            logging.warning(f"Error generating nvidia-smi plots: {e}")


    @staticmethod
    def _deep_merge(base: dict, overrides: dict) -> dict:
        """Deep merge overrides into base dict.

        For nested dicts, recursively merge. For other types, override replaces base.
        """
        result = base.copy()
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = MainRunner._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_fields(self, config: Configuration, overrides: dict) -> Configuration:
        """Merge override fields into config and sanitize field types."""
        for field, value in overrides.items():
            if field in config and isinstance(config[field], dict) and isinstance(value, dict):
                config[field] = self._deep_merge(config[field], value)
                logging.info(f"Applying override (deep merge): {field.name}")
            else:
                config[field] = value
                logging.info(f"Applying override: {field.name} = {value}")
        for k, v in config.items():
            assert isinstance(k, Field), f"Invalid Configuration key {k} is not a Mitten Field object"
            if isinstance(v, str) and (k.from_string and k.from_string is not str):
                logging.debug(f"Configuration - Parsing string for field {k.name}")
                config[k] = k.from_string(v)
        return config

    def _accuracy_override(self, config: Configuration, accuracy_overrides: dict,
                           workload_setting: C.WorkloadSetting) -> Configuration:
        """Apply ACCURACY_OVERRIDES fields."""
        return self._apply_fields(config, accuracy_overrides.get(workload_setting, {}))

    def _compliance_override(self, config: Configuration, compliance_overrides: dict,
                             workload_setting: C.WorkloadSetting) -> Configuration:
        """Apply COMPLIANCE_OVERRIDES fields."""
        overrides = compliance_overrides.get(self.audit_test, {}).get(workload_setting, {})
        return self._apply_fields(config, overrides)

    def _run_workload(self, benchmark: C.Benchmark, scenario: C.Scenario):
        """Run a specific workload for a given benchmark and scenario.

        This method sets up the workload configuration, creates a pipeline, and executes it
        with the appropriate power context.

        Args:
            benchmark (C.Benchmark): The benchmark to run
            scenario (C.Scenario): The scenario to run the benchmark under
        """
        # Override log directory for audit tests
        if self.audit_test is not None:
            # Check if we even need to run the audit test in the first place
            verifier = get_audit_verifier(self.audit_test)
            if benchmark in verifier.exclude_list:
                logging.info(f"Skipping audit test {self.audit_test.valstr} for {benchmark.valstr} {scenario.valstr} as it is not needed for submission.")
                return

            _audit_log_dir = paths.BUILD_DIR / "compliance_logs" / self.audit_test.valstr
            _audit_log_dir.mkdir(parents=True, exist_ok=True)
            os.environ["LOG_DIR"] = str(_audit_log_dir)

        # self.audit_test == None will cleanup any audit configs before harness starts
        set_audit_conf(self.audit_test, benchmark)

        config = self._configs[(benchmark, scenario)]

        with config.autoapply():
            m = G_BENCHMARK_MODULES[benchmark]
            m.load()

            if self.action == C.Action.RunLLMServer:
                ops = [RunTrtllmServeOp]
            elif self.action == C.Action.RunHarness:
                ops = [v for k, v in m.custom_op_impls.items()]
            else:
                raise ValueError(f"Unsupported action: {self.action.valstr}")

            if self.show_help:
                print(HelpInfo.build_help_string(ops))
                sys.exit(0)

            scratch_space = ScratchSpace(paths.BUILD_DIR)
            pipeline = Pipeline(scratch_space, ops, dict())

            # Start nvidia-smi monitoring if verbose_nvsmi is enabled
            self._start_nvidia_smi_monitoring()
            pipeline.run()



    def _action_show_paths(self):
        if paths._CONFIG_PATH is None:
            cfg_label = "(none — using hardcoded defaults)"
        elif not paths._CONFIG_PATH.exists():
            cfg_label = f"{paths._CONFIG_PATH} (not found, using defaults)"
        elif "NV_MLPINF_PATHS_CONFIG" in os.environ:
            cfg_label = f"{paths._CONFIG_PATH} (via NV_MLPINF_PATHS_CONFIG)"
        else:
            cfg_label = f"{paths._CONFIG_PATH} (default location)"
        print(f"nv_mlpinf path configuration (config: {cfg_label})\n")

        system_id = self.system_id
        if any(a == '--system_name' or a.startswith('--system_name=') for a in sys.argv):
            system_source = "arg: --system_name"
        elif "SYSTEM_NAME" in os.environ:
            system_source = "env: SYSTEM_NAME"
        else:
            system_source = "auto-detected"
        print(f"  {'system_name':<26} = {system_id:<45} [{system_source}]")
        print()

        entries = [
            ("project_base_dir",       paths.PROJECT_BASE_DIR),
            ("build_dir",              paths.BUILD_DIR),
            ("mlperf_scratch_path",    paths.MLPERF_SCRATCH_PATH),
            ("model_dir",              paths.MODEL_DIR),
            ("data_dir",               paths.DATA_DIR),
            ("preprocessed_data_dir",  paths.PREPROCESSED_DATA_DIR),
            ("trtllm_dir",             paths.TRTLLM_DIR),
            ("mlcommons_inf_repo",     paths.MLCOMMONS_INF_REPO),
            ("results_submission_dir", paths.RESULTS_SUBMISSION_DIR),
            ("results_staging_dir",    paths.RESULTS_STAGING_DIR),
        ]
        for key, val in entries:
            source = paths._path_sources.get(key, "unknown")
            print(f"  {key:<26} = {str(val):<45} [{source}]")
        sys.exit(0)

    def _action_display_results(self):
        log_dir = Path(self.log_dir)
        all_pass = print_session_results(log_dir)
        if not all_pass:
            raise SystemExit("Accuracy tests failed.")

    def run_all(self):
        """Run all configured workloads.

        This method iterates through all configured benchmarks and scenarios,
        running each workload in sequence.
        """
        if self.action == C.Action.ShowPaths:
            self._action_show_paths()
            return
        if self.action == C.Action.DisplayResults:
            self._action_display_results()
            return
        for benchmark in self.benchmarks:
            for scenario in self.scenarios:
                self._run_workload(benchmark, scenario)


def main():
    # Support positional action syntax: `nv-mlpinf run_harness --benchmarks ...`
    # Rewrite to `nv-mlpinf --action run_harness --benchmarks ...` for nvmitten's argparse.
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        sys.argv.insert(1, '--action')

    # DETECTED_SYSTEM is built at import time by nvmitten hardware detection.
    # If --system_name is given, override its ID before MainRunner is instantiated.
    # Using parse_known_args so nvmitten's own argparse handles all other flags.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--system_name', default=None)
    _pre_args, _ = _pre.parse_known_args()
    if _pre_args.system_name:
        apply_system_name_override(_pre_args.system_name)

    mp.set_start_method("spawn")

    Ops.MPS().disable()
    if "id" not in DETECTED_SYSTEM.extras:
        logging.info(f"Detected system did not match any known systems. Exiting. {DETECTED_SYSTEM}")
    else:
        logging.info(f"Detected system ID: {DETECTED_SYSTEM.extras['id']}")
        with Configuration().autoapply():  # Create empty Configuration to invoke autoconfigure
            runner = MainRunner(DETECTED_SYSTEM)
        runner.run_all()


if __name__ == "__main__":
    main()
