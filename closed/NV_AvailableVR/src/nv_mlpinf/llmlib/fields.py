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

from typing import Any, Dict, List

from nvmitten.configurator import Field
from nvmitten.configurator.fields import AutoConfStrategy

import pathlib
from nv_mlpinf.common.paths import PROJECT_BASE_DIR


__doc__ = """LLM Harness Fields

Used to control components of the LLM Harness and the TRTLLM backend.
"""

capture_server_logs_dir = Field(
    "capture_server_logs_dir",
    description="Path where server-logs, and harness logs will be placed in",
    from_string=pathlib.Path)

trtllm_lib_path = Field(
    "trtllm_lib_path",
    description="Path to TensorRT-LLM repo root",
    from_string=pathlib.Path)




server_instance_size = Field(
    "server_instance_size",
    description="Number of GPUs per server instance for endpoint launches. Must be set in harness.py for docker single node benchmark workflow.",
    from_string=int)

# Fields used by whisper config files (configs/whisper/*/TRTLLM/*/harness.py)
tensor_parallelism = Field(
    "tensor_parallelism",
    description="Tensor parallelism degree.",
    from_string=int)

pipeline_parallelism = Field(
    "pipeline_parallelism",
    description="Pipeline parallelism degree.",
    from_string=int)

trtllm_runtime_flags = Field(
    "trtllm_runtime_flags",
    description="TRT-LLM runtime flags (dict).")

quantizer_outdir = Field(
    "llm_quantizer_outdir",
    description="Path to store the quantized checkpoints for TRTLLM",
    from_string=pathlib.Path)

quantizer_lib_path_override = Field(
    "quantizer_lib_path_override",
    description="Path to the TRTLLM library to use for quantization",
    from_string=pathlib.Path)

llm_gen_config_path = Field(
    "llm_gen_config_path",
    description="The path to the json files storing the generation configs.",
    from_string=pathlib.Path)

use_token_latencies = Field(
    "use_token_latencies",
    description="If enabled, uses token latencies.",
    from_string=bool)

disable_progress_display = Field(
    "disable_progress_display",
    description="Disable LLMHarness progress display rendering (default=False).",
    from_string=bool)

enable_sort = Field(
    "enable_sort",
    description="(Placeholder, not functional) Enable sorting of requests by token length (default=True).",
    from_string=bool)

enable_ttft_latency_tracker = Field(
    "enable_ttft_latency_tracker",
    description="Enable latency tracker for TTFT(default=False).",
    from_string=bool)

harness_use_hf_tokenizer = Field(
    "harness_use_hf_tokenizer",
    description="Use HuggingFace tokenizer on harness/client instead of local checkpoint tokenizer.",
    from_string=bool)

server_use_hf_tokenizer = Field(
    "server_use_hf_tokenizer",
    description="Use HuggingFace tokenizer on server instead of local checkpoint tokenizer.",
    from_string=bool)



trtllm_server_urls = Field(
    "trtllm_server_urls",
    description="TRTLLM server URLs (default=0.0.0.0:30000)",
    from_string=lambda s: s.split(','))

server_in_foreground = Field(
    "server_in_foreground",
    description="Run the server process in foreground instead of child zombie procs",
    from_string=bool)
    
show_steady_state_progress = Field(
    "show_steady_state_progress",
    description="Show steady state information in progress bar",
    from_string=bool)

traffic_distribution_policy = Field(
    "traffic_distribution_policy",
    description="Traffic distribution policy for load balancing across cores. Options: 'round_robin', 'load_balancing', 'isl_load_balancing' (default: auto-determined based on scenario)",
    from_string=str)

warmup_iterations = Field(
    "warmup_iterations",
    description="Number of warmup iterations before actual benchmark. (default: auto-determined)",
    from_string=int)

readiness_timeout = Field(
    "readiness_timeout",
    description="Maximum seconds to wait for all cores to become healthy during warmup (default: 300)",
    from_string=int)

trtllm_yml_override = Field(
    "trtllm_yml_override",
    description="Path to a YAML file to use instead of generating config from trtllm_*_flags. "
                "Works with trtllm_endpoint, disagg_prefill, disagg_decode core types.",
    from_string=pathlib.Path)

env_yml_override = Field(
    "env_yml_override",
    description="Path to a YAML file containing environment variables to pass to workers. "
                "The file should contain key-value pairs, e.g., {TRTLLM_MOE_ENABLE_ALLTOALL: '1'}. "
                "Used with --core_type=disagg_prefill/disagg_decode.",
    from_string=pathlib.Path)

nsys_options = Field(
    "nsys_options",
    description="YML file containing nsys options",
    from_string=pathlib.Path)

# Dynamo disaggregated serving fields
dynamo_frontend_host = Field(
    "dynamo_frontend_host",
    description="Hostname/IP of Dynamo master node. Workers use this to connect via etcd (port 2379) and NATS (port 4222). "
                "Used with --core_type=disagg_prefill/disagg_decode.",
    from_string=str)

dynamo_frontend_port = Field(
    "dynamo_frontend_port",
    description="Base HTTP port for frontends (default: 8000). Each frontend uses base_port + rank.",
    from_string=int)

dynamo_router_mode = Field(
    "dynamo_router_mode",
    description="Router mode for the frontend: 'round-robin', 'kv', or 'random' (default: round-robin).",
    from_string=str)

dynamo_kv_overlap_weight = Field(
    "dynamo_kv_overlap_weight",
    description="KV overlap score weight for KV router mode (0.0-1.0, default: 1.0).",
    from_string=float)

dynamo_router_replica_sync = Field(
    "dynamo_router_replica_sync",
    description="Enable router replica synchronization for multiple frontends (default: False).",
    from_string=bool)

dynamo_frontend_secondary = Field(
    "dynamo_frontend_secondary",
    description="Mark this frontend as secondary (doesn't start NATS/etcd, connects to primary). "
                "Used when launching multiple frontends as separate srun commands.",
    from_string=bool)

dynamo_cluster = Field(
    "dynamo_cluster",
    description="""Dynamo cluster configuration for disaggregated serving.

Used by launch_disagg_cluster.py to orchestrate a complete disaggregated cluster:
- Frontend (NATS, etcd, router) on first node (uses system from config path)
- Prefill workers on first K nodes
- Decode workers on next M nodes

The frontend system is inferred from the config path: configs/{system}/scenario/benchmark.py

Top-level keys:
    num_prefill_workers (int): Number of prefill worker instances
    num_decode_workers (int): Number of decode worker instances
    gpus_per_node (int): GPUs per node in the cluster (default: 4)

Nested 'prefill'/'decode' dict:
    system (str): System name for workers (e.g., 'GB200-NVL72_GB200-186GB_aarch64x4')
    config_id (str): Config ID to load from worker system's ATOMIC_EXPORTS (default: 'dynamo_prefill'/'dynamo_decode')
    trtllm_yml_override (str): Path to YAML config file (alternative to config_id)

Worker config is specified via EITHER config_id OR trtllm_yml_override:
- config_id: Worker loads from its system's ATOMIC_EXPORTS[config_id]
- trtllm_yml_override: Worker uses --trtllm_yml_override=/path/to/config.yaml

Example with config_id (workers load from their system configs):
    llm_fields.dynamo_cluster: {
        'num_prefill_workers': 1,
        'num_decode_workers': 1,
        'gpus_per_node': 4,
        'prefill': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x4',
            'config_id': 'dynamo_prefill',
        },
        'decode': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x16',
            'config_id': 'dynamo_decode',
        },
    },

Example with trtllm_yml_override (workers use explicit YAML files):
    llm_fields.dynamo_cluster: {
        'num_prefill_workers': 1,
        'num_decode_workers': 1,
        'gpus_per_node': 4,
        'prefill': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x4',
            'trtllm_yml_override': PROJECT_BASE_DIR / 'scripts/configs/prefill_config.yaml',
        },
        'decode': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x16',
            'trtllm_yml_override': PROJECT_BASE_DIR / 'scripts/configs/decode_config.yaml',
        },
    },
""",
    autoconf_strategy=AutoConfStrategy.DictUpdate)
