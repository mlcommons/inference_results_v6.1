# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from enum import Enum
import pathlib

from nvmitten.configurator import Field

from nv_mlpinf.common.constants import AuditTest


__doc__ = """Harness control flags

Settings for benchmark harness runs.
"""


test_run = Field(
    "test_run",
    description="If set, will set min_duration to 1 minute (60000ms). For Offline and Server, min_query_count is set to 1.",
    from_string=bool)

glog_verbosity = Field(
    "glog_verbosity",
    description="Enable verbose output",
    from_string=int)



class MPIMode(Enum):
    """Enum for supported MPI modes"""
    LEADER = "leader"
    LEGACY = "legacy"

    @classmethod
    def from_string(cls, s: str) -> 'MPIMode':
        """Parse MPI mode from string to MPIMode enum.

        Args:
            s (str): String representation of MPI mode (e.g., 'leader' or 'legacy')

        Returns:
            MPIMode: The corresponding MPIMode enum value

        Raises:
            ValueError: If the string doesn't match any MPIMode value
        """
        for mode in cls:
            if mode.value == s.lower():
                return mode
        raise ValueError(f"Invalid mpi_mode '{s}'. Must be one of: {', '.join([m.value for m in cls])}")


mpi_mode = Field(
    "mpi_mode",
    description=f"MPI mode for multi-process launches (choices: {', '.join([m.value for m in MPIMode])}). "
                "Controls how the harness detects and handles MPI task initialization.",
    from_string=MPIMode.from_string)

config_id = Field(
    "config_id",
    description="Configuration ID for atomic configs. Only used with --mpi_mode=leader. "
                "Selects a specific config variant from ATOMIC_EXPORTS. Defaults to 'default'.",
    from_string=str)

profiler = Field(
    "profiler",
    description="[INTERNAL ONLY] Select profiler to use.",
    argparse_opts={
        "choices": ["nsys", "nvprof", "ncu", "pic-c"]
    })

audit_test = Field(
    "audit_test",
    description="The audit test to run, if an audit-related action is chosed.",
    from_string=AuditTest.get_match)

no_audit_verify = Field(
    "no_audit_verify",
    description=("If set, skip the verification step for the audit harness. Ignored if not "
                 "running audit harness."),
    from_string=bool)

vboost_slider = Field(
    "vboost_slider",
    description=("Control clock-propagation ratios between GPC-XBAR. "
                 "Look at `nvidia-smi boost-slider --vboost`."),
    from_string=int)

tensor_path = Field(
    "tensor_path",
    description="Path to preprocessed samples in .npy format",
    from_string=str)  # TODO: tensor_path can be a comma-separated list. Handle this later.

warmup_duration = Field(
    "warmup_duration",
    description="Minimum duration to perform warmup for (s)",
    from_string=float)

use_graphs = Field(
    "use_graphs",
    description="Enable CUDA graphs.",
    from_string=bool)

workers_per_core = Field(
    "workers_per_core",
    description="Number of endpoint worker processes per core (default: 2).",
    from_string=int)

max_concurrency = Field(
    "max_concurrency",
    description="Maximum number of concurrent in-flight requests per endpoint core. "
                "Use -1 for unlimited concurrency. Default is unlimited.",
    from_string=int)


class CoreType(Enum):
    """Enum for supported core types"""
    TRTLLM_EXECUTOR = "trtllm_executor"
    TRTLLM_ENDPOINT = "trtllm_endpoint"
    DISAGG_FRONTEND = "disagg_frontend"
    DISAGG_PREFILL = "disagg_prefill"
    DISAGG_DECODE = "disagg_decode"
    DYNAMO_ENDPOINT = "dynamo_endpoint"  # Harness-only mode for pre-deployed Dynamo clusters
    DUMMY = "dummy"

    @classmethod
    def from_string(cls, s: str) -> 'CoreType':
        """Parse core type from string to CoreType enum.

        Args:
            s (str): String representation of core type (e.g., 'trtllm_executor')

        Returns:
            CoreType: The corresponding CoreType enum value

        Raises:
            ValueError: If the string doesn't match any CoreType value
        """
        for core_type in cls:
            if core_type.value == s:
                return core_type
        raise ValueError(f"Invalid core type: {s}. Valid options are: {', '.join([ct.value for ct in cls])}")


core_type = Field(
    "core_type",
    description=f"Type of core to use (choices: {', '.join([ct.value for ct in CoreType])})",
    from_string=CoreType.from_string)


class DisaggBenchMode(Enum):
    """Benchmark mode for disaggregated serving"""
    FULL = "full"  # Full request (prefill + decode)
    PREFILL_ONLY = "prefill_only"  # Only prefill, max_output_len=1
    DECODE_ONLY = "decode_only"  # Only decode timing (not yet implemented)

    @classmethod
    def from_string(cls, s: str) -> 'DisaggBenchMode':
        """Parse disagg bench mode from string to DisaggBenchMode enum.

        Args:
            s (str): String representation of bench mode (e.g., 'full', 'prefill_only', 'decode_only')

        Returns:
            DisaggBenchMode: The corresponding DisaggBenchMode enum value

        Raises:
            ValueError: If the string doesn't match any DisaggBenchMode value
        """
        for mode in cls:
            if mode.value == s.lower():
                return mode
        raise ValueError(f"Invalid disagg_bench_mode: {s}. Valid options are: {', '.join([m.value for m in cls])}")


disagg_bench_mode = Field(
    "disagg_bench_mode",
    description=f"Disaggregated benchmark mode (choices: {', '.join([m.value for m in DisaggBenchMode])}). "
                "Used with disagg core types to benchmark specific phases. "
                "'prefill_only' sets max_output_len=1 to measure only prefill performance. "
                "'decode_only' measures only decode phase timing (not yet implemented).",
    from_string=DisaggBenchMode.from_string)


# Dangerous Flag, reduce TTFT because of the additional overhead of the metric capture process in LLM server
enable_metrics = Field(
    "enable_metrics",
    description="Enable metrics capture during harness execution (default: False). "
                "Set to False to disable background metric capture.",
    from_string=bool)
