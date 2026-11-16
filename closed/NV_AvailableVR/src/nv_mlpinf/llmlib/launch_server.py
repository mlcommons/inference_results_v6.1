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

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import yaml

from nv_mlpinf import G_BENCHMARK_MODULES
from nv_mlpinf.common import logging
from nv_mlpinf.common.constants import Benchmark
from nv_mlpinf.common.paths import TRTLLM_DIR, BUILD_DIR, PROJECT_BASE_DIR
from nv_mlpinf.common.systems.system_list import DETECTED_SYSTEM
from nv_mlpinf.common.workload import Workload
import nv_mlpinf.fields.general as general_fields
from nv_mlpinf.fields import harness as harness_fields
from nv_mlpinf.fields import loadgen as loadgen_fields
from nv_mlpinf.fields.harness import MPIMode
from nv_mlpinf.llmlib.builder import TRTLLMBuilderOp, TRTLLMQuantizerOp
import nv_mlpinf.llmlib.fields as llm_fields
from nvmitten.configurator import autoconfigure, bind
from nvmitten.nvidia.accelerator import GPU
from nvmitten.pipeline import Operation

from .config import TrtllmEndpointConfig, TrtllmEndpointServerConfig



def _load_env_from_yaml(env_yml_override: str | None) -> dict[str, str]:
    """Load custom environment variables from a YAML file.

    Args:
        env_yml_override: Path to the YAML file containing env var overrides, or None.

    Returns:
        A dict of {str: str} environment variables. Empty dict if no file or invalid.
    """
    if not env_yml_override:
        return {}
    env_path = Path(env_yml_override)
    if not env_path.exists():
        logging.warning(f"env_yml_override file not found: {env_path}")
        return {}
    with open(env_path, 'r') as f:
        custom_env = yaml.safe_load(f) or {}
    if not isinstance(custom_env, dict):
        logging.warning(f"env_yml_override must contain a dict, got: {type(custom_env)}")
        return {}
    custom_env = {str(k): str(v) for k, v in custom_env.items()}
    logging.info(f"Applied env from {env_path}:")
    for env_k, env_v in custom_env.items():
        logging.info(f"Setting env {env_k} = {env_v}")
    return custom_env


def setup_tiktoken_for_gpt_oss(benchmark: Benchmark) -> None:
    """
    Temporary workaround, TODO: @shobhitv to remove later
    Download tiktoken encodings for gpt-oss-120b benchmark.

    WAR for openai_harmony trying to download vocab files at runtime from the network.
    Some environments block the download, so we pre-download the tiktoken files.
    """
    if benchmark != Benchmark.GPT_OSS_120B:
        return

    tiktoken_dir = BUILD_DIR / "gpt-oss-tiktoken"
    tiktoken_dir.mkdir(parents=True, exist_ok=True)

    tiktoken_files = {
        "o200k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "cl100k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    }

    for fname, url in tiktoken_files.items():
        fpath = tiktoken_dir / fname
        if fpath.exists():
            logging.info(f"Tiktoken file already present: {fpath}")
            continue
        logging.info(f"Downloading {fname} to {fpath}...")
        try:
            urllib.request.urlretrieve(url, fpath)
            logging.info(f"Downloaded {fname}")
        except Exception as e:
            logging.warning(f"Failed to download {fname}: {e}")

    # Set env vars so trtllm-serve can find the tiktoken files
    os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_dir)
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(tiktoken_dir)
    logging.info(f"Set TIKTOKEN_CACHE_DIR and TIKTOKEN_ENCODINGS_BASE to {tiktoken_dir}")

@autoconfigure
@bind(llm_fields.server_in_foreground)
@bind(llm_fields.nsys_options, "nsys_config_file")
@bind(general_fields.log_dir)
@bind(Workload.FIELD, "workload")
@bind(llm_fields.trtllm_yml_override)
@bind(llm_fields.env_yml_override)
@bind(loadgen_fields.test_mode)
class RunTrtllmServeOp(Operation):
    """ Operation to run trtllm-serve endpoint(s) using trtllm-serve cli """

    def __init__(self,
                 workload: Workload,
                 server_in_foreground: bool = False,
                 nsys_config_file: Path = None,
                 log_dir: Path = None,
                 trtllm_yml_override: Path = None,
                 env_yml_override: Path = None,
                 test_mode: str = None):
        super().__init__()
        self.log_dir = log_dir
        self.wl = workload
        self.blocking = server_in_foreground
        self.nsys_options = None
        self.nsys_cmd_parts = None
        self.trtllm_yml_override = trtllm_yml_override
        self.env_yml_override = env_yml_override
        self.test_mode = test_mode
        if nsys_config_file is not None:
            with open(nsys_config_file, 'r') as f:
                nsys_options = yaml.safe_load(f)
            self.nsys_cmd_parts = [
                f"{nsys_options['nsys_path']}", "profile",
            ]
            self.nsys_cmd_parts = self.nsys_cmd_parts + nsys_options['extra_flags']
            self.nsys_options = nsys_options

        # Merge user flags with defaults
        self.harness_config = TrtllmEndpointServerConfig()

        if self.harness_config.capture_server_logs_dir is not None:
            self.log_dir = self.harness_config.capture_server_logs_dir
        else:
            self.log_dir = self.harness_config.log_dir

        # Get GPU devices
        gpus = DETECTED_SYSTEM.accelerators[GPU]
        self.devices = [gpu.gpu_index for gpu in gpus]

    def run(self, scratch_space, dependency_outputs):
        # WAR: Setup tiktoken for gpt-oss-120b (openai_harmony needs these files)
        setup_tiktoken_for_gpt_oss(self.wl.benchmark)

        # 1. Get checkpoint path from benchmark module
        target_path = Path(G_BENCHMARK_MODULES[self.wl.benchmark].load().MODEL_CHECKPOINT_PATH)
        assert target_path.exists(), f"Checkpoint path {target_path} does not exist."

        # 2. Determine tokenizer path
        if self.harness_config.server_use_hf_tokenizer:
            logging.info("Using HuggingFace tokenizer")
            self.tokenizer_path = None
        else:
            self.tokenizer_path = target_path


        # 3. Calculate number of trtllm-serve commands to launch on this node
        gpus_per_server = self.harness_config.get_instance_size()
        launch_endpoints = self.harness_config.trtllm_endpoint_urls

        # 4. Determine config YAML path - always in log_dir
        extra_config_path = Path(self.log_dir) / "trtllm_serve_extra_conf.yaml"

        if self.trtllm_yml_override:
            override_source = Path(self.trtllm_yml_override)
            assert override_source.exists(), f"trtllm_yml_override file not found: {override_source}"
            shutil.copy2(override_source, extra_config_path)
            logging.info(f"Copied override YAML: \nfrom: {override_source} \nto: {extra_config_path}")
        else:
            raise ValueError("A trtllm_yml_override YAML file is required. Pass --trtllm_yml_override=/path/to/config.yaml")
        # Read and log YAML contents (works for both override and generated)
        with extra_config_path.open('r') as f:
            yaml_contents = f.read()
        source = f"override from {self.trtllm_yml_override}" if self.trtllm_yml_override else "generated"
        logging.info(f"Extra Config YAML Contents ({source}):\n{yaml_contents}")

        # 4. Launch trtllm-serve processes
        server_processes = []
        mpi_rank = int(os.getenv('SLURM_PROCID', 0))
        custom_env = _load_env_from_yaml(self.env_yml_override)

        for index in range(len(launch_endpoints)):
            nsys_cmd_parts_current = None
            if self.nsys_cmd_parts is not None:
                dp_rank = os.getenv("DP_RANK", index)
                nsys_cmd_parts_current = self.nsys_cmd_parts + [
                    f"--output={self.log_dir}/{self.nsys_options['profile_name']}-dp{dp_rank}-rank{mpi_rank}",
                ]
            endpoint_url = launch_endpoints[index]
            endpoint_port = endpoint_url.split(':')[-1]
            env = os.environ.copy()
            env.update(custom_env)

            cmd = []
            if self.harness_config.mpi_mode == MPIMode.LEADER:
                # Assert DP=1 in leader mode by checking num_gpus from system name, SLURM world size, and TP*PP are equal
                system_name = DETECTED_SYSTEM.extras["id"]
                num_gpus_from_system = int(system_name.split('x')[-1])
                assert num_gpus_from_system == self.harness_config.global_size, \
                    (
                        f"DP must be 1 in leader mode: num_gpus_from_system={num_gpus_from_system}, "
                        f"global_size={self.harness_config.global_size}.\n"
                        f"System name: {system_name}\n"
                        "Please ensure SYSTEM_NAME matches the total GPU count and SLURM_NTASKS equals the instance size."
                    )

                cmd = ['trtllm-llmapi-launch', 'trtllm-serve']
                local_rank_id = os.getenv('SLURM_LOCALID', '0')
                gpu_ids = [local_rank_id]
                # In leader mode, each rank sees all GPUs via NVIDIA_VISIBLE_DEVICES.
                # nsys profiling requires CUDA_VISIBLE_DEVICES to be set so each process monitors only its own GPU.
                if self.nsys_options:
                    env['CUDA_VISIBLE_DEVICES'] = local_rank_id

            else:
                # Legacy mode: Calculate GPU assignment
                cmd = ['trtllm-serve']
                start_gpu = index * gpus_per_server
                gpu_ids = list(range(start_gpu, start_gpu + gpus_per_server))
                env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

                # Clean environment: remove SLURM variables to avoid MPI initialization issues in legacy mode
                slurm_vars_removed = []
                for key in list(env.keys()):
                    if key.startswith('SLURM_'):
                        del env[key]
                        slurm_vars_removed.append(key)
                if slurm_vars_removed and index == 0:  # Log only once for the first endpoint
                    logging.info(f"Removed {len(slurm_vars_removed)} SLURM environment variables for running pseudo-MPI programs within single task srun")

            cmd.extend([
                str(target_path),
                '--host', '0.0.0.0',
                '--port', str(endpoint_port),
                '--extra_llm_api_options', str(extra_config_path.absolute())
            ])

            if self.tokenizer_path is not None:
                cmd.extend(['--tokenizer', str(self.tokenizer_path)])

            if self.harness_config.mpi_mode == MPIMode.LEADER:
                # In leader mode, include MPI rank to avoid cluttering a single file
                mpi_rank = int(os.getenv('SLURM_PROCID', 0))
                dp_rank = os.getenv('DP_RANK', None)
                if dp_rank is not None:
                    dp_rank = int(dp_rank)
                    log_file = self.log_dir / f'trtllm_serve_dp{dp_rank}_rank{mpi_rank}.log'
                else:
                    log_file = self.log_dir / f'trtllm_serve_dp{index}_rank{mpi_rank}.log'
            else:
                log_file = self.log_dir / f'trtllm_serve_{index}.log'
            if self.nsys_options and nsys_cmd_parts_current:
                assert "TLLM_PROFILE_START_STOP" in env, "TLLM_PROFILE_START_STOP must be set in the environment when using nsys"
                cmd = nsys_cmd_parts_current + cmd

            with open(log_file, 'w') as f:
                f.write(f"Launch ENV:\n{env}\n\n")
                f.write(f"Launch CMD:\n{' '.join(cmd)}\n\n")
                # Write YAML contents to log (read from file - works for both override and generated)
                with open(extra_config_path, 'r') as yaml_f:
                    yaml_content = yaml_f.read()
                source = f"override from {self.trtllm_yml_override}" if self.trtllm_yml_override else "generated"
                f.write(f"Extra Config ({source}):\n{yaml_content}\n\n")
                server_processes.append(subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT
                ))

            logging.info(f"Launched {endpoint_url}")
            logging.info(f"  CMD: {' '.join(cmd)}")
            logging.info(f"  GPU devices: {gpu_ids}")
            logging.info(f"  Log file: {log_file}")

        if self.blocking:
            for process in server_processes:
                process.wait()

        return {"trtllm_endpoint_urls": launch_endpoints}

    @classmethod
    def output_keys(cls):
        return ["trtllm_endpoint_urls"]

    @classmethod
    def immediate_dependencies(cls):
        return None




@autoconfigure
@bind(llm_fields.server_in_foreground)
@bind(general_fields.log_dir)
@bind(llm_fields.dynamo_frontend_port)
@bind(llm_fields.dynamo_router_mode)
@bind(llm_fields.dynamo_kv_overlap_weight)
@bind(llm_fields.dynamo_router_replica_sync)
@bind(llm_fields.dynamo_frontend_host)
@bind(llm_fields.dynamo_frontend_secondary)
class RunDisaggFrontendOp(Operation):
    """Operation to launch disaggregated serving frontend (NATS, etcd, router).

    This operation starts the required infrastructure services for disaggregated serving:
    - NATS server (port 4222) for messaging (primary only)
    - etcd (port 2379) for service discovery (primary only)
    - Dynamo frontend (configurable port) for request routing

    For multiple frontends launched via srun --ntasks=N:
    - Primary frontend (rank 0): starts NATS + etcd + router
    - Secondary frontends (rank > 0): only start router, connect to primary's NATS/etcd

    Rank is auto-detected from SLURM_PROCID environment variable.
    Port is computed as: base_port + rank

    Used with: --core_type=disagg_frontend
    """

    # Standard ports
    ETCD_PORT = 2379
    NATS_PORT = 4222
    DEFAULT_FRONTEND_PORT = 8000

    def __init__(self,
                 server_in_foreground: bool = True,
                 log_dir: Path = None,
                 dynamo_frontend_port: int = None,
                 dynamo_router_mode: str = None,
                 dynamo_kv_overlap_weight: float = None,
                 dynamo_router_replica_sync: bool = False,
                 dynamo_frontend_host: str = None,
                 dynamo_frontend_secondary: bool = False):
        super().__init__()
        self.blocking = server_in_foreground
        self.log_dir = log_dir
        self.base_frontend_port = dynamo_frontend_port or self.DEFAULT_FRONTEND_PORT
        self.router_mode = dynamo_router_mode or "round-robin"
        self.kv_overlap_weight = dynamo_kv_overlap_weight
        self.router_replica_sync = dynamo_router_replica_sync
        self.primary_host = dynamo_frontend_host  # For secondary frontends to find primary
        self.is_secondary = dynamo_frontend_secondary  # Explicit secondary frontend flag

    def run(self, scratch_space, dependency_outputs):
        # Detect rank from SLURM environment (must read before cleaning env vars)
        global_rank = int(os.environ.get('SLURM_PROCID', 0))
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        num_tasks = int(os.environ.get('SLURM_NTASKS', 1))
        # Primary if global rank 0 AND not explicitly marked as secondary
        # (needed when launching separate srun commands per port)
        is_primary = (global_rank == 0) and not self.is_secondary

        # Compute port from LOCAL rank: base_port + local_rank
        # This ensures each node uses the same port in distributed mode (ntasks-per-node=1)
        # while stacked mode (multiple frontends on same node) gets unique ports
        frontend_port = self.base_frontend_port + local_rank

        # Set up log directory (per-rank if multiple frontends)
        log_dir = Path(self.log_dir)
        if num_tasks > 1:
            log_dir = log_dir / f'fe_{global_rank}'
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"[Rank {global_rank}/{num_tasks}] Frontend starting on port {frontend_port}")

        processes = []

        if is_primary:
            # Primary frontend (rank 0): start NATS and etcd
            logging.info(f"[Rank {global_rank}] Starting PRIMARY frontend (NATS + etcd + router)")

            # Start NATS server
            nats_log = log_dir / "nats_server.log"
            nats_cmd = ["nats-server", "-js"]
            logging.info(f"Starting NATS server on port {self.NATS_PORT}")
            logging.info(f"  CMD: {' '.join(nats_cmd)}")
            with open(nats_log, 'w') as f:
                nats_proc = subprocess.Popen(
                    nats_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
                processes.append(nats_proc)
            logging.info(f"NATS server started, PID: {nats_proc.pid}, log: {nats_log}")

            # Start etcd
            etcd_log = log_dir / "etcd.log"
            etcd_data_dir = log_dir / "etcd_data"
            etcd_data_dir.mkdir(parents=True, exist_ok=True)
            etcd_cmd = [
                "etcd",
                "--listen-client-urls", f"http://0.0.0.0:{self.ETCD_PORT}",
                "--advertise-client-urls", f"http://0.0.0.0:{self.ETCD_PORT}",
                "--data-dir", str(etcd_data_dir)
            ]
            logging.info(f"Starting etcd on port {self.ETCD_PORT}")
            logging.info(f"  CMD: {' '.join(etcd_cmd)}")
            with open(etcd_log, 'w') as f:
                etcd_proc = subprocess.Popen(
                    etcd_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
                processes.append(etcd_proc)
            logging.info(f"etcd started, PID: {etcd_proc.pid}, log: {etcd_log}")

            # Wait for NATS and etcd to initialize
            time.sleep(3)
        else:
            # Secondary frontend (global rank > 0): connect to primary's NATS/etcd
            assert self.primary_host, \
                "dynamo_frontend_host must be specified for multi-frontend launch"
            logging.info(f"[Rank {global_rank}] Starting SECONDARY frontend (router only)")
            logging.info(f"  Connecting to primary NATS/etcd at {self.primary_host}")
            # Wait a bit for primary to start NATS/etcd
            time.sleep(5)

        # Start Dynamo frontend router
        frontend_log = log_dir / "disagg_frontend.log"
        frontend_cmd = [
            "python3", "-m", "dynamo.frontend",
            "--http-port", str(frontend_port),
            "--router-mode", self.router_mode
        ]

        # Add optional router arguments
        if self.kv_overlap_weight is not None:
            frontend_cmd.extend(["--kv-overlap-score-weight", str(self.kv_overlap_weight)])
        if self.router_replica_sync:
            frontend_cmd.append("--router-replica-sync")

        # Clean environment: remove SLURM variables to avoid MPI initialization issues in dynamo.frontend
        clean_env = os.environ.copy()
        slurm_vars_removed = []
        for key in list(clean_env.keys()):
            if key.startswith('SLURM_'):
                del clean_env[key]
                slurm_vars_removed.append(key)
        if slurm_vars_removed:
            logging.info(f"Removed {len(slurm_vars_removed)} SLURM environment variables for running pseudo-MPI programs within single task srun")

        # For secondary frontends, set environment to connect to primary's NATS/etcd
        if not is_primary:
            clean_env["ETCD_ENDPOINTS"] = f"{self.primary_host}:{self.ETCD_PORT}"
            clean_env["NATS_SERVER"] = f"nats://{self.primary_host}:{self.NATS_PORT}"

        frontend_type = "secondary" if not is_primary else "primary"
        logging.info(f"Starting {frontend_type} frontend on port {frontend_port}")
        logging.info(f"  Router mode: {self.router_mode}")
        if self.kv_overlap_weight is not None:
            logging.info(f"  KV overlap weight: {self.kv_overlap_weight}")
        if self.router_replica_sync:
            logging.info(f"  Router replica sync: enabled")
        logging.info(f"  CMD: {' '.join(frontend_cmd)}")

        with open(frontend_log, 'w') as f:
            frontend_proc = subprocess.Popen(
                frontend_cmd,
                env=clean_env,
                stdout=f,
                stderr=subprocess.STDOUT
            )
            processes.append(frontend_proc)
        logging.info(f"Frontend started, PID: {frontend_proc.pid}, log: {frontend_log}")

        frontend_url = f"localhost:{frontend_port}"
        logging.info(f"Disaggregated serving frontend started. URL: {frontend_url}")
        if is_primary:
            logging.info(f"Workers should connect with: --dynamo_frontend_host=<this_node_hostname>")

        if self.blocking:
            try:
                for proc in processes:
                    proc.wait()
            except KeyboardInterrupt:
                logging.info("Stopping frontend services...")
                for proc in processes:
                    proc.terminate()

        return {"frontend_url": frontend_url}

    @classmethod
    def output_keys(cls):
        return ["frontend_url"]

    @classmethod
    def immediate_dependencies(cls):
        return set()


@autoconfigure
@bind(llm_fields.server_in_foreground)
@bind(general_fields.log_dir)
@bind(Workload.FIELD, "workload")
@bind(llm_fields.dynamo_frontend_host)
@bind(harness_fields.mpi_mode)
@bind(llm_fields.trtllm_yml_override)
@bind(llm_fields.env_yml_override)
@bind(loadgen_fields.test_mode)
class RunDisaggPrefillOp(Operation):
    """Operation to launch disaggregated prefill (context) worker.

    This operation starts a TRT-LLM prefill worker that handles prompt processing
    and registers with the disaggregated serving frontend via NATS/etcd.

    IMPORTANT: Must be launched in MPI leader mode via:
        srun --ntasks=<tensor_parallelism> --nodes=<num_nodes> make run_llm_server ...

    Used with: --core_type=disagg_prefill
    """

    # Standard ports
    ETCD_PORT = 2379
    NATS_PORT = 4222

    def __init__(self,
                 workload: Workload,
                 server_in_foreground: bool = True,
                 log_dir: Path = None,
                 dynamo_frontend_host: str = None,
                 mpi_mode: MPIMode = MPIMode.LEGACY,
                 trtllm_yml_override: Path = None,
                 env_yml_override: Path = None,
                 test_mode: str = None):
        super().__init__()
        self.blocking = server_in_foreground
        self.log_dir = log_dir
        self.wl = workload
        self.dynamo_frontend_host = dynamo_frontend_host
        self.mpi_mode = mpi_mode
        self.trtllm_yml_override = trtllm_yml_override
        self.env_yml_override = env_yml_override
        self.test_mode = test_mode
        # Use TrtllmExtraYAMLConfig for YAML generation (if no override)
        self.harness_config = TrtllmExtraYAMLConfig() if not trtllm_yml_override else None

    @property
    def etcd_endpoints(self) -> str:
        """Get etcd endpoint URL derived from dynamo_frontend_host."""
        return f"{self.dynamo_frontend_host}:{self.ETCD_PORT}"

    @property
    def nats_server(self) -> str:
        """Get NATS server URL derived from dynamo_frontend_host."""
        return f"nats://{self.dynamo_frontend_host}:{self.NATS_PORT}"

    def run(self, scratch_space, dependency_outputs):
        assert self.dynamo_frontend_host, \
            "dynamo_frontend_host must be specified for disaggregated prefill workers"

        # Prefill workers MUST be launched in leader mode with srun
        assert self.mpi_mode == MPIMode.LEADER, (
            "Disaggregated prefill workers must be launched in MPI leader mode.\n"
            "Please launch with: srun --ntasks=<tensor_parallelism> --nodes=<num_nodes> make run_llm_server ..."
        )

        # Get model path from benchmark module
        model_path = Path(G_BENCHMARK_MODULES[self.wl.benchmark].load().MODEL_CHECKPOINT_PATH)
        assert model_path.exists(), f"Model path {model_path} does not exist"

        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Prefill worker - always uses "prefill" disaggregation mode
        disagg_mode = "prefill"
        worker_name = "prefill"

        # Determine config YAML path - always in log_dir
        config_yaml_path = log_dir / f"disagg_{worker_name}_config.yaml"

        # Use override YAML if provided (copy to log_dir), otherwise generate from config
        if self.trtllm_yml_override:
            override_source = Path(self.trtllm_yml_override)
            assert override_source.exists(), f"trtllm_yml_override file not found: {override_source}"
            shutil.copy2(override_source, config_yaml_path)
            logging.info(f"Copied override YAML: \nfrom: {override_source} \nto: {config_yaml_path}")
            # Apply accuracy overrides if test_mode=AccuracyOnly and _accuracy.yml exists
            _apply_yml_accuracy_override(override_source, config_yaml_path, self.test_mode)
        else:
            # Generate worker config YAML using TrtllmExtraYAMLConfig
            config_yaml_content = self.harness_config.extra_config_yaml
            with open(config_yaml_path, 'w') as f:
                f.write(config_yaml_content)
            logging.info(f"Generated {worker_name} worker config at {config_yaml_path}")

        # Get served model name from benchmark module
        model_repo = G_BENCHMARK_MODULES[self.wl.benchmark].load(("HF_MODEL_REPO",)).HF_MODEL_REPO
        served_model_name, _ = list(model_repo.items())[0]

        # Set environment for worker
        # Reference: https://github.com/ai-dynamo/dynamo/blob/main/examples/basics/multinode/trtllm/start_trtllm_worker.sh
        env = os.environ.copy()
        env.update({
            'ETCD_ENDPOINTS': self.etcd_endpoints,
            'NATS_SERVER': self.nats_server,
            'TRTLLM_SERVER_DISABLE_GC': '1',
            'TRTLLM_WORKER_DISABLE_GC': '1',
            'TLLM_LOG_LEVEL': 'INFO',
        })

        custom_env = _load_env_from_yaml(self.env_yml_override)
        env.update(custom_env)

        # In leader mode, each MPI rank handles one GPU via SLURM_LOCALID
        mpi_rank = int(os.getenv('SLURM_PROCID', 0))
        worker_log = log_dir / f"disagg_{worker_name}_worker_rank{mpi_rank}.log"

        # Build command using trtllm-llmapi-launch
        cmd = [
            "trtllm-llmapi-launch",
            "python3", "-m", "dynamo.trtllm",
            "--model-path", str(model_path),
            "--served-model-name", served_model_name,
            "--extra-engine-args", str(config_yaml_path),
            "--disaggregation-mode", disagg_mode,
        ]

        logging.info(f"Starting disaggregated prefill worker (MPI rank {mpi_rank})")
        logging.info(f"  CMD: {' '.join(cmd)}")
        logging.info(f"  ETCD_ENDPOINTS: {env['ETCD_ENDPOINTS']}")
        logging.info(f"  NATS_SERVER: {env['NATS_SERVER']}")
        logging.info(f"  MODEL_PATH: {model_path}")
        logging.info(f"  SERVED_MODEL_NAME: {served_model_name}")
        logging.info(f"  CONFIG: {config_yaml_path}")
        logging.info(f"  DISAGGREGATION_MODE: {disagg_mode}")

        # Launch worker via subprocess
        with open(worker_log, 'w') as f:
            f.write(f"MPI Rank: {mpi_rank}\n")
            f.write(f"Launch CMD: {' '.join(cmd)}\n\n")
            f.write(f"Environment:\n")
            for k, v in env.items():
                if k.startswith(('ETCD', 'NATS', 'TRTLLM', 'TLLM', 'SLURM', 'CUDA')):
                    f.write(f"  {k}={v}\n")
            # Log custom env vars from env_yml_override
            if custom_env:
                f.write(f"\nCustom env from {self.env_yml_override}:\n")
                for k, v in custom_env.items():
                    f.write(f"  {k}={v}\n")
            # Write YAML config from file (works for both override and generated)
            with open(config_yaml_path, 'r') as yaml_f:
                yaml_content = yaml_f.read()
            source = f"override from {self.trtllm_yml_override}" if self.trtllm_yml_override else "generated"
            f.write(f"\nConfig YAML ({source}):\n{yaml_content}\n\n")
            f.flush()

            worker_proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT
            )

        logging.info(f"Disaggregated prefill worker started, PID: {worker_proc.pid}, log: {worker_log}")

        if self.blocking:
            try:
                worker_proc.wait()
            except KeyboardInterrupt:
                logging.info("Stopping prefill worker...")
                worker_proc.terminate()

        return {"worker_pid": worker_proc.pid}

    @classmethod
    def output_keys(cls):
        return ["worker_pid"]

    @classmethod
    def immediate_dependencies(cls):
        return None


@autoconfigure
@bind(llm_fields.server_in_foreground)
@bind(general_fields.log_dir)
@bind(Workload.FIELD, "workload")
@bind(llm_fields.dynamo_frontend_host)
@bind(harness_fields.mpi_mode)
@bind(llm_fields.trtllm_yml_override)
@bind(llm_fields.env_yml_override)
@bind(loadgen_fields.test_mode)
class RunDisaggDecodeOp(Operation):
    """Operation to launch disaggregated decode (generation) worker.

    This operation starts a TRT-LLM decode worker that handles token generation
    and registers with the disaggregated serving frontend via NATS/etcd.

    IMPORTANT: Must be launched in MPI leader mode via:
        srun --ntasks=<tensor_parallelism> --nodes=<num_nodes> make run_llm_server ...

    Used with: --core_type=disagg_decode
    """

    # Standard ports
    ETCD_PORT = 2379
    NATS_PORT = 4222

    def __init__(self,
                 workload: Workload,
                 server_in_foreground: bool = True,
                 log_dir: Path = None,
                 dynamo_frontend_host: str = None,
                 mpi_mode: MPIMode = MPIMode.LEGACY,
                 trtllm_yml_override: Path = None,
                 env_yml_override: Path = None,
                 test_mode: str = None):
        super().__init__()
        self.blocking = server_in_foreground
        self.log_dir = log_dir
        self.wl = workload
        self.dynamo_frontend_host = dynamo_frontend_host
        self.mpi_mode = mpi_mode
        self.trtllm_yml_override = trtllm_yml_override
        self.env_yml_override = env_yml_override
        self.test_mode = test_mode
        # Use TrtllmExtraYAMLConfig for YAML generation (if no override)
        self.harness_config = TrtllmExtraYAMLConfig() if not trtllm_yml_override else None

    @property
    def etcd_endpoints(self) -> str:
        """Get etcd endpoint URL derived from dynamo_frontend_host."""
        return f"{self.dynamo_frontend_host}:{self.ETCD_PORT}"

    @property
    def nats_server(self) -> str:
        """Get NATS server URL derived from dynamo_frontend_host."""
        return f"nats://{self.dynamo_frontend_host}:{self.NATS_PORT}"

    def run(self, scratch_space, dependency_outputs):
        assert self.dynamo_frontend_host, \
            "dynamo_frontend_host must be specified for disaggregated decode workers"

        # Decode workers MUST be launched in leader mode with srun
        assert self.mpi_mode == MPIMode.LEADER, (
            "Disaggregated decode workers must be launched in MPI leader mode.\n"
            "Please launch with: srun --ntasks=<tensor_parallelism> --nodes=<num_nodes> make run_llm_server ..."
        )

        # Get model path from benchmark module
        model_path = Path(G_BENCHMARK_MODULES[self.wl.benchmark].load().MODEL_CHECKPOINT_PATH)
        assert model_path.exists(), f"Model path {model_path} does not exist"

        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Decode worker - always uses "decode" disaggregation mode
        disagg_mode = "decode"
        worker_name = "decode"

        # Determine config YAML path - always in log_dir
        config_yaml_path = log_dir / f"disagg_{worker_name}_config.yaml"

        # Use override YAML if provided (copy to log_dir), otherwise generate from config
        if self.trtllm_yml_override:
            override_source = Path(self.trtllm_yml_override)
            assert override_source.exists(), f"trtllm_yml_override file not found: {override_source}"
            shutil.copy2(override_source, config_yaml_path)
            logging.info(f"Copied override YAML: \nfrom: {override_source} \nto: {config_yaml_path}")
            # Apply accuracy overrides if test_mode=AccuracyOnly and _accuracy.yml exists
            _apply_yml_accuracy_override(override_source, config_yaml_path, self.test_mode)
        else:
            # Generate worker config YAML using TrtllmExtraYAMLConfig
            config_yaml_content = self.harness_config.extra_config_yaml
            with open(config_yaml_path, 'w') as f:
                f.write(config_yaml_content)
            logging.info(f"Generated {worker_name} worker config at {config_yaml_path}")

        # Get served model name from benchmark module
        model_repo = G_BENCHMARK_MODULES[self.wl.benchmark].load(("HF_MODEL_REPO",)).HF_MODEL_REPO
        served_model_name, _ = list(model_repo.items())[0]

        # Set environment for worker
        # Reference: https://github.com/ai-dynamo/dynamo/blob/main/examples/basics/multinode/trtllm/start_trtllm_worker.sh
        env = os.environ.copy()
        env.update({
            'ETCD_ENDPOINTS': self.etcd_endpoints,
            'NATS_SERVER': self.nats_server,
            'TRTLLM_SERVER_DISABLE_GC': '1',
            'TRTLLM_WORKER_DISABLE_GC': '1',
            'TLLM_LOG_LEVEL': 'INFO',
        })

        custom_env = _load_env_from_yaml(self.env_yml_override)
        env.update(custom_env)

        # In leader mode, each MPI rank handles one GPU via SLURM_LOCALID
        mpi_rank = int(os.getenv('SLURM_PROCID', 0))
        worker_log = log_dir / f"disagg_{worker_name}_worker_rank{mpi_rank}.log"

        # Build command using trtllm-llmapi-launch
        cmd = [
            "trtllm-llmapi-launch",
            "python3", "-m", "dynamo.trtllm",
            "--model-path", str(model_path),
            "--served-model-name", served_model_name,
            "--extra-engine-args", str(config_yaml_path),
            "--disaggregation-mode", disagg_mode,
        ]

        logging.info(f"Starting disaggregated decode worker (MPI rank {mpi_rank})")
        logging.info(f"  CMD: {' '.join(cmd)}")
        logging.info(f"  ETCD_ENDPOINTS: {env['ETCD_ENDPOINTS']}")
        logging.info(f"  NATS_SERVER: {env['NATS_SERVER']}")
        logging.info(f"  MODEL_PATH: {model_path}")
        logging.info(f"  SERVED_MODEL_NAME: {served_model_name}")
        logging.info(f"  CONFIG: {config_yaml_path}")
        logging.info(f"  DISAGGREGATION_MODE: {disagg_mode}")

        # Launch worker via subprocess
        with open(worker_log, 'w') as f:
            f.write(f"MPI Rank: {mpi_rank}\n")
            f.write(f"Launch CMD: {' '.join(cmd)}\n\n")
            f.write(f"Environment:\n")
            for k, v in env.items():
                if k.startswith(('ETCD', 'NATS', 'TRTLLM', 'TLLM', 'SLURM', 'CUDA')):
                    f.write(f"  {k}={v}\n")
            # Log custom env vars from env_yml_override
            if custom_env:
                f.write(f"\nCustom env from {self.env_yml_override}:\n")
                for k, v in custom_env.items():
                    f.write(f"  {k}={v}\n")
            # Write YAML config from file (works for both override and generated)
            with open(config_yaml_path, 'r') as yaml_f:
                yaml_content = yaml_f.read()
            source = f"override from {self.trtllm_yml_override}" if self.trtllm_yml_override else "generated"
            f.write(f"\nConfig YAML ({source}):\n{yaml_content}\n\n")
            f.flush()

            worker_proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT
            )

        logging.info(f"Disaggregated decode worker started, PID: {worker_proc.pid}, log: {worker_log}")

        if self.blocking:
            try:
                worker_proc.wait()
            except KeyboardInterrupt:
                logging.info("Stopping decode worker...")
                worker_proc.terminate()

        return {"worker_pid": worker_proc.pid}

    @classmethod
    def output_keys(cls):
        return ["worker_pid"]

    @classmethod
    def immediate_dependencies(cls):
        return None
