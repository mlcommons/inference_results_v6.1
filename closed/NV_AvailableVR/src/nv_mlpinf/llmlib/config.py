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

"""Configuration classes for LLM harness components."""

from __future__ import annotations
from nv_mlpinf import G_BENCHMARK_MODULES
import dataclasses
from enum import Enum
import json
import logging
import os
from pathlib import Path
import yaml
import re
from typing import Any, Dict, List, Optional, Type

import nv_mlpinf.common.constants as C
from nv_mlpinf.common.workload import Workload
from nv_mlpinf.common.systems.system_list import DETECTED_SYSTEM
from nv_mlpinf.fields import harness as harness_fields
from nv_mlpinf.fields.harness import MPIMode
from nv_mlpinf.fields import general as gen_fields
from nv_mlpinf.fields import loadgen as lg_fields
from nvmitten.configurator import autoconfigure, bind
from nvmitten.json_utils import JSONable
from nvmitten.nvidia.accelerator import GPU

from . import fields as llm_fields
from .utils import get_yaml_string


def ignore_extra_kwargs(cls):
    """Decorator that allows dataclasses to ignore extra kwargs during initialization."""
    # First apply dataclass if not already applied
    if not hasattr(cls, '__dataclass_fields__'):
        cls = dataclasses.dataclass(cls)

    # Store the original __init__
    original_init = cls.__init__

    # Create a new __init__ that filters kwargs
    def __init__(self, **kwargs):
        # Get valid field names from the dataclass
        field_names = set(cls.__dataclass_fields__.keys())

        # Filter kwargs to only include known fields
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in field_names}
        ignored_kwargs = {k: v for k, v in kwargs.items() if k not in field_names}

        if ignored_kwargs:
            logging.debug(f"ignore_extra_kwargs: {cls.__name__} ignoring kwargs: {sorted(ignored_kwargs.keys())}")

        # Call original __init__ with filtered kwargs
        original_init(self, **filtered_kwargs)

    # Replace the __init__ method
    cls.__init__ = __init__

    return cls


@dataclasses.dataclass
class JSONSliceable(JSONable):
    """TRTLLM config.json files can contain arbitrary fields that vary depending on the model. Some
    fields are common and are used by our LLM Harness. Subclasses of TRTLLMConfig can specify which
    fields should be parsed from the config.json file.
    """

    _name_to_class = {}

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        JSONSliceable._name_to_class[cls.__name__] = cls

    @classmethod
    def type_from_name(cls, name: str) -> Type[JSONSliceable]:
        try:
            return JSONSliceable._name_to_class[name]
        except:
            return None

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> JSONSliceable:
        args = {}

        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue

            # We can't directly check f.type here, since the `from __future__ import annotations`
            # changes the type of the fields to be a string with the class name instead of the
            # actual type.
            _type = f.type
            if not isinstance(_type, type):
                _type = JSONSliceable.type_from_name(_type)

            if isinstance(_type, type) and issubclass(_type, JSONSliceable):
                args[f.name] = _type.from_json(d[f.name])
            else:
                args[f.name] = d[f.name]
        return cls(**args)

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@ignore_extra_kwargs
@dataclasses.dataclass
class GenerationConfig(JSONSliceable):
    eos_token_id: int = 2
    bos_token_id: int = 1
    max_output_len: int = 1024
    min_output_len: int = 1
    name: str = "llama"
    runtime_beam_width: int = 1
    streaming: bool = True
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 0.001
    use_stop_tokens: bool = False
    skip_special_tokens: bool = True

    @classmethod
    def from_file(cls, path: os.PathLike) -> GenerationConfig:
        """
        Load GenerationConfig from a JSON file.

        Args:
            path (os.PathLike): The path to the JSON file containing the generation configuration.

        Returns:
            GenerationConfig: The loaded GenerationConfig object.
        """
        with Path(path).open() as f:
            return cls.from_json(json.load(f)['generation_config'])


@autoconfigure
@bind(harness_fields.core_type, "_core_type")
@bind(llm_fields.server_instance_size)
@bind(llm_fields.enable_ttft_latency_tracker)
@bind(llm_fields.show_steady_state_progress)
@bind(llm_fields.traffic_distribution_policy)
@bind(llm_fields.llm_gen_config_path, "gen_config_path")
@bind(lg_fields.test_mode)
@bind(gen_fields.log_dir)
@bind(llm_fields.capture_server_logs_dir)
@bind(Workload.FIELD, "workload")
@bind(harness_fields.disagg_bench_mode)
@bind(llm_fields.readiness_timeout)
@ignore_extra_kwargs
@dataclasses.dataclass
class HarnessConfig:
    # we make core_type a property to allow lazy import
    _core_type: harness_fields.CoreType = None
    server_instance_size: Optional[int] = None
    enable_ttft_latency_tracker: bool = False
    show_steady_state_progress: bool = False
    gen_config_path: str = None
    test_mode: str = "PerformanceOnly"
    workload: Workload = None
    traffic_distribution_policy: str = None  # auto-assign based on workload
    log_dir: str = None
    capture_server_logs_dir: str = None
    disagg_bench_mode: harness_fields.DisaggBenchMode = None
    readiness_timeout: int = 300

    gen_config: GenerationConfig = dataclasses.field(default_factory=GenerationConfig)
    random_seed: int = 0

    def __post_init__(self):
        if self.gen_config_path is not None:
            gen_config_path = Path(self.gen_config_path)
            if not gen_config_path.exists():
                raise FileNotFoundError(
                    f"Generation config file not found: {gen_config_path}. "
                    "Unset `llm_gen_config_path` to use default generation config."
                )
            self.gen_config = GenerationConfig.from_file(gen_config_path)

        # adjust streaming based on scenario
        self.gen_config.streaming &= (self.workload.scenario != C.Scenario.Offline)

        # Handle disaggregated benchmark modes
        if self.disagg_bench_mode == harness_fields.DisaggBenchMode.PREFILL_ONLY:
            # For prefill-only benchmarking, set max_output_len=1 to skip decode
            logging.info("Disagg bench mode: prefill_only - setting max_output_len=1")
            self.gen_config.max_output_len = 1
        elif self.disagg_bench_mode == harness_fields.DisaggBenchMode.DECODE_ONLY:
            raise NotImplementedError("Disagg bench mode 'decode_only' is not yet supported")

        if self.capture_server_logs_dir is not None:
            self.capture_server_logs_dir = Path(self.capture_server_logs_dir)

        # override assign traffic distribution policy based on workload
        if self.traffic_distribution_policy is None:
            # TODO(vir): advanced load-balancing
            match self.workload.scenario:
                case C.Scenario.Server: self.traffic_distribution_policy = "load_balancing"
                case C.Scenario.Offline: self.traffic_distribution_policy = "round_robin"
                case _: self.traffic_distribution_policy = "round_robin"

    def get_instance_size(self) -> int:
        """ Get the size for given instance of this LLM. """
        return int(self.server_instance_size) if self.server_instance_size is not None else 1

    @property
    def core_type(self):
        """ FIXME(vir): WAR Lazy import the Workload DEFAULT CORE TYPE """
        if self._core_type is None:
            # use default core type if not specified
            self._core_type = G_BENCHMARK_MODULES[self.workload.benchmark].load().DEFAULT_CORE_TYPE

        return self._core_type




# Backend-Specific Harness Configurations

class CheckpointType(Enum):
    """Enum for supported checkpoint types"""
    TRTLLM = "TRTLLM"
    HF = "HuggingFace"



def _parse_instance_size_from_yaml(yaml_path: Path) -> int:
    """Derive the GPU instance size from a trtllm-serve YAML config.

    If attention DP is enabled or MOE expert parallel size > 1, the instance size
    equals moe_expert_parallel_size.  Otherwise it is TP * PP.
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"trtllm_yml_override YAML has invalid format (expected a dict, got {type(cfg).__name__}): {yaml_path}")

    tp = cfg.get('tensor_parallel_size', 1)
    pp = cfg.get('pipeline_parallel_size', 1)
    ep = cfg.get('moe_expert_parallel_size', 1)

    if ep > 1:
        return int(ep)
    return int(tp * pp)


@autoconfigure
@bind(harness_fields.core_type, "_core_type")
@bind(gen_fields.log_dir)
@bind(llm_fields.capture_server_logs_dir)
@bind(Workload.FIELD, "workload")
@bind(llm_fields.trtllm_server_urls, "trtllm_endpoint_urls")
@bind(llm_fields.trtllm_yml_override, "trtllm_yml_override")
@bind(llm_fields.server_use_hf_tokenizer, "server_use_hf_tokenizer")
@bind(harness_fields.mpi_mode)
@ignore_extra_kwargs
@dataclasses.dataclass
class TrtllmEndpointServerConfig:
    """Minimal config for launching trtllm-serve endpoints only.

    This config intentionally avoids harness-only fields (for example
    llm_gen_config_path and max_concurrency) so run_llm_server can use a
    backend-only config file.
    """
    _core_type: Optional[harness_fields.CoreType] = None
    log_dir: str = None
    capture_server_logs_dir: Optional[Path] = None
    workload: Workload = None
    trtllm_endpoint_urls: Optional[List[str]] = None
    endpoint_url: str = "0.0.0.0:30000"
    mpi_mode: MPIMode = MPIMode.LEGACY
    trtllm_yml_override: Optional[Path] = None
    server_use_hf_tokenizer: bool = False
    global_size: Optional[int] = None # leader mode: Slurm NTasks 
    server_instance_size: Optional[int] = None # needed for single-NODE orchestrator mode only

    def __post_init__(self):
        if self.capture_server_logs_dir is not None:
            self.capture_server_logs_dir = Path(self.capture_server_logs_dir)

        if not self.trtllm_yml_override or not str(self.trtllm_yml_override).strip():
            raise ValueError("trtllm_yml_override must be specified for TrtllmEndpointServerConfig")
        self.trtllm_yml_override = Path(self.trtllm_yml_override)
        if not self.trtllm_yml_override.exists():
            raise FileNotFoundError(f"trtllm_yml_override file not found: {self.trtllm_yml_override}")

        # Always set global_size in leader mode and require explicit endpoints.
        if self.mpi_mode == MPIMode.LEADER:
            if global_size := os.environ.get('SLURM_NTASKS', None):
                self.global_size = int(global_size)
            if self.trtllm_endpoint_urls is None:
                raise ValueError("Endpoint URLs must be provided in leader mode")

        if self.trtllm_endpoint_urls is None:
            num_local_gpus = len(DETECTED_SYSTEM.accelerators[GPU])
            num_dp_ranks = num_local_gpus // self.get_instance_size()
            self.trtllm_endpoint_urls = [
                self._get_endpoint_url(dp_index)
                for dp_index in range(num_dp_ranks)
            ]

    def _get_endpoint_url(self, dp_index: int = 0) -> str:
        base_port = 30000
        port = base_port + dp_index
        local_node_name = "0.0.0.0"
        return f"{local_node_name}:{port}"

    def get_instance_size(self) -> int:
        return _parse_instance_size_from_yaml(self.trtllm_yml_override)

    @property
    def core_type(self):
        if self._core_type is None:
            self._core_type = G_BENCHMARK_MODULES[self.workload.benchmark].load(('DEFAULT_CORE_TYPE',)).DEFAULT_CORE_TYPE
        return self._core_type

@autoconfigure
@bind(llm_fields.trtllm_server_urls, "trtllm_endpoint_urls")
@bind(llm_fields.harness_use_hf_tokenizer, "harness_use_hf_tokenizer")
@bind(harness_fields.enable_metrics, "enable_metrics")
@bind(harness_fields.workers_per_core)
@bind(harness_fields.max_concurrency)
@bind(harness_fields.mpi_mode)
@ignore_extra_kwargs
@dataclasses.dataclass
class TrtllmEndpointConfig(HarnessConfig):
    """Configuration for TrtllmEndpointCore"""
    CHECKPOINT_T = CheckpointType.HF

    trtllm_endpoint_urls: Optional[List[str]] = None
    endpoint_url: str = "0.0.0.0:30000"
    mpi_mode: MPIMode = MPIMode.LEGACY
    harness_use_hf_tokenizer: bool = False
    global_size: Optional[int] = None
    workers_per_core: int = 2
    max_concurrency: int = -1
    enable_metrics: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Always merge default runtime flags (needed by harness even with YAML override)
        self.runtime_flags = {
            'http_backend': 'custom_http',  # force custom_http backend for best performance.
        }

        # Support migration from runtime_flags -> harness_fields while preserving backward compatibility.
        self.max_concurrency = int(self.max_concurrency)
        self.workers_per_core = int(self.workers_per_core)

        # Always set global_size in leader mode (needed by harness)
        if self.mpi_mode == MPIMode.LEADER:
            if global_size := os.environ.get('SLURM_NTASKS', None):
                self.global_size = int(global_size)
            if self.trtllm_endpoint_urls is None:
                raise ValueError("Endpoint URLs must be provided in leader mode")

        # NOTE(vir):
        # we have full world visibility in trtllm-serve only in single-NODE orchestrator mode
        # so we are able to spawn all erquired local DP ranks using subprocess
        if self.trtllm_endpoint_urls is None and self.__class__ is TrtllmEndpointConfig:
            num_local_gpus = len(DETECTED_SYSTEM.accelerators[GPU])
            assert self.server_instance_size is not None, "server_instance_size must be specified for TrtllmEndpointConfig in single-NODE orchestrator mode"
            assert self.server_instance_size > 0, f"server_instance_size must be positive, got {self.server_instance_size}"
            worker_size = self.get_instance_size()
            num_dp_ranks = num_local_gpus // worker_size
            self.trtllm_endpoint_urls = [
                self._get_endpoint_url(dp_index)
                for dp_index in range(num_dp_ranks)
            ]

    def _get_endpoint_url(self, dp_index: int = 0) -> str:
        base_port = 30000
        port = base_port + dp_index
        local_node_name = "0.0.0.0"
        return f"{local_node_name}:{port}"
    
    def get_model_repo(self):
        return G_BENCHMARK_MODULES[self.workload.benchmark].load().HF_MODEL_REPO



@autoconfigure
@bind(llm_fields.trtllm_server_urls, "trtllm_endpoint_urls")
@ignore_extra_kwargs
@dataclasses.dataclass
class DynamoEndpointConfig(HarnessConfig):
    """Minimal configuration for running harness against pre-deployed Dynamo clusters.

    This config skips all build/runtime flag loading from system configs.
    It only requires endpoint URLs and uses sensible defaults for everything else.

    Usage:
        make run_harness RUN_ARGS="--core_type=dynamo_endpoint --trtllm_server_urls=host:port"
    """

    trtllm_endpoint_urls: Optional[List[str]] = None
    endpoint_url: str = "0.0.0.0:8000"
    runtime_flags: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # Minimal runtime flags for HTTP harness
    DEFAULT_RUNTIME_FLAGS = {
        'workers_per_core': 2,
        'http_backend': 'custom_http',
        'max_concurrency': 5120,
    }

    def __post_init__(self):
        super().__post_init__()

        # Use minimal runtime flags - don't load from system configs
        self.runtime_flags = DynamoEndpointConfig.DEFAULT_RUNTIME_FLAGS | self.runtime_flags

        # Ensure endpoint URLs are set
        if self.trtllm_endpoint_urls:
            self.endpoint_url = self.trtllm_endpoint_urls[0]
        elif self.endpoint_url:
            self.trtllm_endpoint_urls = [self.endpoint_url]
        else:
            raise ValueError("DynamoEndpointConfig requires --trtllm_server_urls")

        logging.info(f"DynamoEndpointConfig initialized with endpoints: {self.trtllm_endpoint_urls}")

    def get_model_repo(self):
        """Get model repository info from benchmark module."""
        return G_BENCHMARK_MODULES[self.workload.benchmark].load(("HF_MODEL_REPO",)).HF_MODEL_REPO
