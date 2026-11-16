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


from nv_mlpinf.common.systems.system_list import DETECTED_SYSTEM
from nvmitten.configurator import bind, autoconfigure, Field
from nvmitten.system import System
from pathlib import Path
from typing import ClassVar, List

import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.general as general_fields
import nv_mlpinf.fields.meta as metafields
import os


@autoconfigure
@bind(general_fields.log_dir)
class Workload:
    """
    Represents a workload configuration for MLPerf inference benchmarks.

    This class manages the configuration and settings for running MLPerf inference workloads,
    including benchmark type, scenario, system settings, and device types.

    Attributes:
        FIELD (ClassVar[Field]): Configuration field for workload injection.
        benchmark (C.Benchmark): The benchmark type to run.
        scenario (C.Scenario): The inference scenario to use.
        system (System): The system configuration.
        setting (C.WorkloadSetting): Workload-specific settings.
        device_types (List[str]): List of device types to use (e.g., ["gpu", "dla"]).
        log_dir (Path): Directory for storing workload logs.
    """

    # Convenience Field so that Workload can be injected into the Configuration by MainRunner
    FIELD: ClassVar[Field] = Field("workload",
                                   disallow_default=True,
                                   disallow_argparse=True)

    def __init__(self,
                 benchmark: C.Benchmark,
                 scenario: C.Scenario,
                 system: System = DETECTED_SYSTEM,
                 setting: C.WorkloadSetting = C.WorkloadSetting(),
                 device_types: List[str] = None, # No DLA submission in recent MLPerf Inference
                 log_dir: os.PathLike = paths.BUILD_DIR / "logs" / "default"):
        """
        Initialize a Workload instance.

        Args:
            benchmark (C.Benchmark): The benchmark type to run.
            scenario (C.Scenario): The inference scenario to use.
            system (System, optional): The system configuration. Defaults to DETECTED_SYSTEM.
            setting (C.WorkloadSetting, optional): Workload-specific settings. Defaults to C.WorkloadSetting().
            device_types (List[str], optional): List of device types to use. Defaults to ["gpu"] or ["gpu", "dla"] for SoC.
            log_dir (os.PathLike, optional): Directory for storing workload logs. Defaults to BUILD_DIR/logs/default.
        """
        self.benchmark = benchmark
        self.scenario = scenario
        self.system = system
        self.setting = setting

        if device_types is None or device_types == ["all"]:
            self.device_types = ["gpu"]
            if "is_soc" in self.system.extras["tags"]:
                self.device_types.append("dla")
        else:
            self.device_types = device_types

        self.base_log_dir = Path(log_dir)
        self.log_dir = self.base_log_dir / self.submission_system / self.submission_benchmark / scenario.valstr
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.audit_test01_fallback_mode = False

    def __eq__(self, other):
        """
        Check if two Workload instances are equal.

        Args:

            other (Workload): The other Workload instance to compare with.

        Returns:
            bool: True if the Workloads are equal, False otherwise.
        """
        if not isinstance(other, Workload):
            return NotImplemented
        return (self.benchmark == other.benchmark and
                self.scenario == other.scenario and
                self.system == other.system and
                self.setting == other.setting and
                self.device_types == other.device_types and
                self.log_dir == other.log_dir)

    def __str__(self):
        """
        Return a string representation of the Workload.

        Returns:
            str: String representation in the format "Workload(benchmark, scenario, setting, log_dir)"
        """
        return f"Workload({self.benchmark}, {self.scenario}, {self.setting.short}, {self.log_dir})"

    @property
    def submission_benchmark(self) -> str:
        """
        Get the submission benchmark name based on benchmark and accuracy target.

        Returns:
            str: The submission benchmark name.
        """
        return C.submission_benchmark_name(self.benchmark, self.setting.accuracy_target)

    @property
    def submission_system(self) -> str:
        """
        Get the submission system name based on system ID and power settings.

        Returns:
            str: The submission system name in the format "system_id_TRT[_MaxQ]".
        """
        parts = [self.system.extras["id"], "TRT"]
        if self.setting.power_setting == C.PowerSetting.MaxQ:
            parts.append("MaxQ")
        return '_'.join(parts)


