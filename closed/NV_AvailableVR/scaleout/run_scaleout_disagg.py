#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Topology-driven scaleout script for disaggregated serving.
# Takes a --server-topology <path> pointing to a trtllm-disagg topology JSON
# instead of individual --ctx-atomic-system / --gen-atomic-system / etc. flags.
#
# Launches CTX workers, GEN workers, master servers, and harness using
# trtllm-serve directly (no nv-mlpinf run_llm_server).
#
# Analogous to how run_scaleout.sh uses topology JSON for IFB serving,
# this script uses topology JSON for disaggregated serving.

import argparse
import datetime
import getpass
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Import topology parsing from sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_topology import DisaggTopology, load_topology, resolve_path, parse_env_yml


def run_srun(args, env=None, dry_run=False, background=False, log_file=None):
    cmd = ["srun"] + args
    sys.stderr.write("================================================\n")
    sys.stderr.write("Executing srun command:\n")
    sys.stderr.write(f"{' '.join(cmd)}\n")
    sys.stderr.write("================================================\n")

    if log_file:
        try:
            with open(log_file, "a") as f:
                f.write(' '.join(cmd) + '\n')
        except Exception as e:
            sys.stderr.write(f"WARNING: Failed to write to log file {log_file}: {e}\n")

    if not dry_run:
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        if background:
            return subprocess.Popen(cmd, env=run_env)
        else:
            subprocess.check_call(cmd, env=run_env)
            return None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Topology-driven scaleout for disaggregated serving")
    parser.add_argument("--stage", default="all",
                        help="Stage to run (server|harness|all)")
    parser.add_argument("--server-topology", required=True,
                        help="Path to server-topology.json (trtllm-disagg type)")
    parser.add_argument("--harness-system", required=True,
                        help="Harness system name")
    parser.add_argument("--harness-run-args", default="",
                        help="Arguments passed to nv-mlpinf harness command")
    parser.add_argument("--container-image",
                        help="Container image")
    parser.add_argument("--mlperf-scratch-path",
                        default="/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone",
                        help="Scratch path")
    parser.add_argument("--extra-srun-flags", default="",
                        help="Additional srun flags")
    parser.add_argument("--base-port", type=int, default=30000,
                        help="Base port for all servers (default: 30000)")
    parser.add_argument("--launch-master", default="true",
                        choices=["true", "false"],
                        help="Launch master servers in server stage (default: true)")
    parser.add_argument("--log-dir",
                        help="Log directory (default: auto-generated with timestamp)")
    parser.add_argument("--audit", action="store_true",
                        help="Run audit harness instead of regular harness")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print srun commands without executing them")
    parser.add_argument("--workspace",
                        help="Override workspace (host_vol) directory")
    parser.add_argument("--num-ctx-workers", type=int, default=None,
                        help="Override num_ctx_workers from server-topology.json")
    parser.add_argument("--num-gen-workers", type=int, default=None,
                        help="Override num_gen_workers from server-topology.json")
    parser.add_argument("--num-master-servers", type=int, default=None,
                        help="Override num_master_servers from server-topology.json")
    parser.add_argument("--nsys", action="store_true",
                        help="Wrap trtllm-serve workers with `nsys profile` for per-rank profile collection")
    parser.add_argument("--nsys-config", default=None,
                        help="Path to nsys YAML config (default: <workspace>/internal/nsys/sample_nsys_config.yml)")

    args = parser.parse_args()

    stage = args.stage
    dry_run = args.dry_run
    harness_target = "run_audit_harness" if args.audit else "run_harness"

    # ── Resolve workspace paths ───────────────────────────────────────────

    script_path = Path(__file__).resolve()
    if args.workspace:
        host_vol = str(Path(args.workspace).resolve())
    elif os.environ.get("SLURM_SUBMIT_DIR"):
        host_vol = str(Path(os.environ["SLURM_SUBMIT_DIR"]).resolve())
    else:
        host_vol = str(script_path.parent.parent)
    container_vol = "/work"

    # ── Load topology ─────────────────────────────────────────────────────

    topo_path = Path(args.server_topology)
    if not topo_path.is_file():
        sys.stderr.write(f"ERROR: server-topology.json not found: {topo_path}\n")
        sys.exit(1)

    topo = load_topology(topo_path)
    if not isinstance(topo, DisaggTopology):
        sys.stderr.write(
            f"ERROR: Expected server_type='trtllm-disagg', "
            f"got '{topo.server_type}' in {topo_path}\n")
        sys.exit(1)

    model_path = topo.model_path
    gpus_per_node = topo.gpus_per_node
    num_ctx_workers = topo.num_ctx_workers
    num_gen_workers = topo.num_gen_workers
    gpus_per_ctx = topo.gpus_per_ctx_worker
    gpus_per_gen = topo.gpus_per_gen_worker
    num_master_servers = topo.num_master_servers

    if args.num_ctx_workers is not None:
        print(f"Overriding num_ctx_workers: {num_ctx_workers} -> {args.num_ctx_workers}")
        num_ctx_workers = args.num_ctx_workers
    if args.num_gen_workers is not None:
        print(f"Overriding num_gen_workers: {num_gen_workers} -> {args.num_gen_workers}")
        num_gen_workers = args.num_gen_workers
    if args.num_master_servers is not None:
        print(f"Overriding num_master_servers: {num_master_servers} -> {args.num_master_servers}")
        num_master_servers = args.num_master_servers

    # Resolve trtllm yml and env exports for each worker type
    ctx_container_yml = resolve_path(topo.ctx_trtllm_yml, container_vol)
    gen_container_yml = resolve_path(topo.gen_trtllm_yml, container_vol)

    ctx_env_exports = ""
    if topo.ctx_env_yml:
        ctx_env_exports = parse_env_yml(resolve_path(topo.ctx_env_yml, host_vol))
    gen_env_exports = ""
    if topo.gen_env_yml:
        gen_env_exports = parse_env_yml(resolve_path(topo.gen_env_yml, host_vol))

    # ── Compute GPU totals and node requirements ──────────────────────────

    num_ctx_gpus = num_ctx_workers * gpus_per_ctx
    num_gen_gpus = num_gen_workers * gpus_per_gen
    total_gpus = num_ctx_gpus + num_gen_gpus
    total_nodes_required = math.ceil(total_gpus / gpus_per_node)

    # Deployment modes
    ctx_deployment = "cross_node" if gpus_per_ctx > gpus_per_node else "intra_node"
    gen_deployment = "cross_node" if gpus_per_gen > gpus_per_node else "intra_node"

    if ctx_deployment == "cross_node" and gpus_per_ctx % gpus_per_node != 0:
        sys.stderr.write(
            f"ERROR: CTX cross-node requires gpus_per_ctx_worker ({gpus_per_ctx}) "
            f"to be a multiple of gpus_per_node ({gpus_per_node})\n")
        sys.exit(1)
    if gen_deployment == "cross_node" and gpus_per_gen % gpus_per_node != 0:
        sys.stderr.write(
            f"ERROR: GEN cross-node requires gpus_per_gen_worker ({gpus_per_gen}) "
            f"to be a multiple of gpus_per_node ({gpus_per_node})\n")
        sys.exit(1)

    harness_system = args.harness_system

    # ── Log directory ─────────────────────────────────────────────────────

    slurm_jobid = os.environ.get("SLURM_JOBID", "unknown")

    if args.log_dir:
        host_log_dir = args.log_dir
        log_dir = args.log_dir.replace(host_vol, container_vol, 1)
    else:
        timestamp = datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        log_dir = f"/work/build/logs/scaleout_disagg_{harness_system}_slurm-{slurm_jobid}_{timestamp}"
        host_log_dir = log_dir.replace(container_vol, host_vol, 1)

    if not dry_run:
        os.makedirs(host_log_dir, exist_ok=True)
        # Copy topology and config files to log directory for debugging
        try:
            topo_dir = str(topo_path.parent)
            for f in os.listdir(topo_dir):
                shutil.copy2(os.path.join(topo_dir, f), host_log_dir)
        except Exception as e:
            sys.stderr.write(f"WARNING: Failed to copy config files to log dir: {e}\n")
    srun_log_file = os.path.join(host_log_dir, "srun_commands.log")

    # ── Container image ───────────────────────────────────────────────────

    container_image = args.container_image
    if not container_image:
        docker_tag = f"{getpass.getuser()}-aarch64"
        container_image = os.path.join(
            host_vol, "build", "sqsh_images",
            f"mlperf-inference-{docker_tag}-release.sqsh")

    # ── SLURM validation ──────────────────────────────────────────────────

    if not os.environ.get("SLURM_JOB_NODELIST") or not os.environ.get("SLURM_JOBID"):
        sys.stderr.write("ERROR: Not running in a SLURM allocation\n")
        sys.exit(1)

    try:
        slurm_output = subprocess.check_output(
            ["scontrol", "show", "hostname", os.environ["SLURM_JOB_NODELIST"]],
            text=True)
        node_array = slurm_output.strip().splitlines()
        allocated_node_count = len(node_array)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"ERROR: Failed to get node list: {e}\n")
        sys.exit(1)

    total_gpus_allocated = allocated_node_count * gpus_per_node
    if total_gpus_allocated < total_gpus:
        sys.stderr.write("ERROR: GPU count mismatch\n")
        sys.stderr.write(f"  SLURM allocated: {total_gpus_allocated} "
                         f"({allocated_node_count} nodes x {gpus_per_node} GPUs/node)\n")
        sys.stderr.write(f"  Required: {total_gpus} GPUs "
                         f"({num_ctx_workers} CTX x {gpus_per_ctx} + "
                         f"{num_gen_workers} GEN x {gpus_per_gen})\n")
        sys.exit(1)

    gpus_wasted = total_gpus_allocated - total_gpus

    # ── Summary ───────────────────────────────────────────────────────────

    print("============================================")
    print(f"MLPerf Disagg Scaleout (topology-driven) - {stage}")
    print("============================================")
    print(f"Topology:       {args.server_topology}")
    print(f"Harness system: {harness_system}")
    print(f"Model path:     {model_path}")
    print(f"GPUs per node:  {gpus_per_node}")
    print(f"Allocated nodes: {allocated_node_count}")
    print()
    print(f"GEN: {num_gen_workers} workers x {gpus_per_gen} GPUs = {num_gen_gpus} GPUs [{gen_deployment}]")
    print(f"     trtllm_yml: {topo.gen_trtllm_yml}")
    print(f"CTX: {num_ctx_workers} workers x {gpus_per_ctx} GPUs = {num_ctx_gpus} GPUs [{ctx_deployment}]")
    print(f"     trtllm_yml: {topo.ctx_trtllm_yml}")
    print()
    if args.launch_master == "true":
        print(f"Master servers: {num_master_servers}")
    print(f"Total: {total_nodes_required} nodes, {total_gpus} GPUs used"
          f"{f', {gpus_wasted} GPUs wasted' if gpus_wasted > 0 else ''}")
    print(f"Log directory:  {log_dir}")
    print(f"Harness target: {harness_target}")
    print(f"Base port:      {args.base_port}")
    print(f"Dry run:        {dry_run}")
    print("============================================")

    # ── Common srun arguments ─────────────────────────────────────────────

    srun_flags_list = args.extra_srun_flags.split() if args.extra_srun_flags else []

    base_srun_args = [
        "--overlap",
        "--mpi=pmix",
        f"--container-image={container_image}",
        f"--container-mounts={host_vol}:{container_vol},{args.mlperf_scratch_path}:/home/mlperf_inference_storage",
        f"--container-workdir={container_vol}",
        "--container-remap-root",
    ]

    base_export = "ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage"

    # ── nsys profiling ────────────────────────────────────────────────────
    nsys_enabled = args.nsys
    nsys_path = nsys_profile_name = nsys_extra_flags_str = None
    if nsys_enabled:
        nsys_config_path = args.nsys_config or os.path.join(
            host_vol, "internal", "nsys", "sample_nsys_config.yml")
        if not os.path.isfile(nsys_config_path):
            sys.stderr.write(f"ERROR: nsys config not found: {nsys_config_path}\n")
            sys.exit(1)
        with open(nsys_config_path) as f:
            _nsys_cfg = yaml.safe_load(f)
        for k in ("nsys_path", "profile_name", "extra_flags"):
            if k not in _nsys_cfg:
                sys.stderr.write(f"ERROR: nsys config missing required key: {k}\n")
                sys.exit(1)
        nsys_path = _nsys_cfg["nsys_path"]
        nsys_profile_name = _nsys_cfg["profile_name"]
        nsys_extra_flags_str = " ".join(shlex.quote(x) for x in _nsys_cfg["extra_flags"])
        print(f"nsys: enabled (config: {nsys_config_path})")

    def nsys_profile_prefix(tag):
        """Per-worker nsys prefix matching scaleout.sh behavior."""
        if not nsys_enabled:
            return ""
        return (f"{nsys_path} profile {nsys_extra_flags_str} "
                f"--output='{log_dir}/{nsys_profile_name}-dp{tag}-rank'${{SLURM_PROCID:-0}} ")

    # ── Worker launcher ───────────────────────────────────────────────────

    gen_worker_urls = []
    ctx_worker_urls = []
    master_urls = []

    def launch_worker(worker_type, worker_idx, gpus_per_worker, deployment,
                      container_yml, env_exports):
        """Launch a single CTX or GEN worker via trtllm-serve."""
        if worker_type == "GEN":
            start_gpu = worker_idx * gpus_per_gen
            port_offset = worker_idx
        else:  # CTX
            start_gpu = num_gen_gpus + worker_idx * gpus_per_ctx
            port_offset = num_gen_workers + worker_idx

        end_gpu = start_gpu + gpus_per_worker - 1
        start_node = start_gpu // gpus_per_node
        end_node = end_gpu // gpus_per_node

        server_nodes = node_array[start_node:end_node + 1]
        server_nodes_str = ",".join(server_nodes)

        if deployment == "intra_node":
            gpu_offset = start_gpu % gpus_per_node
            cuda_devices = [str(gpu_offset + g) for g in range(gpus_per_worker)]
        else:
            cuda_devices = [str(g) for g in range(gpus_per_node)]
        cuda_devices_str = ",".join(cuda_devices)

        first_node = server_nodes[0]
        port = args.base_port + port_offset
        worker_url = f"{first_node}:{port}"

        num_nodes = len(server_nodes)
        num_tasks_per_node = len(cuda_devices)
        ipc_port = 10012 + port_offset
        ipc_addr = f"tcp://127.0.0.1:{ipc_port}"

        # Per-worker log dir so trtllm-serve writes per-rank logs separately
        worker_log_dir = f"{log_dir}/{worker_type.lower()}/{worker_type.lower()}{worker_idx}"

        print(f"  {worker_type} worker {worker_idx} on {server_nodes_str} "
              f"(GPUs: {cuda_devices_str}, URL: {worker_url})")

        env_vars = {
            "LOG_DIR": worker_log_dir,
            "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR": ipc_addr,
            "NVIDIA_VISIBLE_DEVICES": cuda_devices_str,
        }

        export_str = base_export + ",NVIDIA_VISIBLE_DEVICES"
        if env_exports:
            export_str += f",{env_exports}"

        # PATCHED: CVD wrapper only when nsys enabled (nsys --gpu-metrics-devices=cuda-visible
        # needs single GPU). For PERF runs the wrapper hurts gpt-oss disagg perf — strip it.
        cvd_prefix = "export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0} && " if nsys_enabled else ""
        serve_str = (
            f"trtllm-llmapi-launch trtllm-serve {model_path} "
            f"--host 0.0.0.0 --port {port} --extra_llm_api_options {container_yml}"
        )
        nsys_prefix = nsys_profile_prefix(f"{worker_type.lower()}{worker_idx}")
        launch_cmd = ["bash", "-c", f"{cvd_prefix}{nsys_prefix}{serve_str}"]

        srun_cmd = base_srun_args + [
            f"--export={export_str}",
            f"--output={host_log_dir}/slurm_logs/{worker_type.lower()}{worker_idx}_rank%t.log",
            f"--nodelist={server_nodes_str}",
            f"--ntasks-per-node={num_tasks_per_node}",
            f"--nodes={num_nodes}",
        ] + srun_flags_list + launch_cmd

        run_srun(srun_cmd, env=env_vars, dry_run=dry_run, background=True, log_file=srun_log_file)
        return worker_url

    # ── Stage: server ─────────────────────────────────────────────────────

    if stage in ["server", "all"]:
        print("============================================")
        print("Stage 1: Launching Workers")
        print("============================================")

        print(f"Launching {num_gen_workers} GEN worker(s)...")
        for idx in range(num_gen_workers):
            url = launch_worker("GEN", idx, gpus_per_gen, gen_deployment,
                                gen_container_yml, gen_env_exports)
            gen_worker_urls.append(url)

        print(f"Launching {num_ctx_workers} CTX worker(s)...")
        for idx in range(num_ctx_workers):
            url = launch_worker("CTX", idx, gpus_per_ctx, ctx_deployment,
                                ctx_container_yml, ctx_env_exports)
            ctx_worker_urls.append(url)

        print("All workers launched.")

        # ── Master servers ────────────────────────────────────────────────

        if args.launch_master == "true":
            total_workers = num_gen_workers + num_ctx_workers
            print("============================================")
            print(f"Launching {num_master_servers} Master Server(s)")
            print("============================================")

            for master_idx in range(num_master_servers):
                master_node_idx = master_idx % len(node_array)
                master_node = node_array[master_node_idx]
                master_port = args.base_port + total_workers + master_idx
                master_url = f"{master_node}:{master_port}"
                master_urls.append(master_url)

                print(f"  Master {master_idx}: {master_url}")

                config = {
                    "hostname": master_node,
                    "port": master_port,
                    "backend": "pytorch",
                    "context_servers": {
                        "num_instances": num_ctx_workers,
                        "urls": ctx_worker_urls,
                    },
                    "generation_servers": {
                        "num_instances": num_gen_workers,
                        "urls": gen_worker_urls,
                    },
                }

                host_config_file = os.path.join(
                    host_log_dir, f"master_server_config_{master_idx}.yaml")
                container_config_file = f"{log_dir}/master_server_config_{master_idx}.yaml"

                if not dry_run:
                    with open(host_config_file, "w") as f:
                        yaml.dump(config, f, default_flow_style=False)
                    print(f"  Config: {host_config_file}")
                else:
                    print(f"  Dry run: would create {host_config_file}")
                    if master_idx == 0:
                        print(yaml.dump(config, default_flow_style=False))

                master_log = f"{log_dir}/master_server_{master_idx}.log"
                master_cmd = (
                    f"trtllm-serve disaggregated "
                    f"--config_file {container_config_file} "
                    f"--server_start_timeout 7200 "
                    f"--request_timeout 7200 "
                    f"> {master_log} 2>&1"
                )

                master_srun_cmd = base_srun_args + [
                    f"--export={base_export}",
                    f"--nodelist={master_node}",
                    "--ntasks=1",
                    "--nodes=1",
                ] + srun_flags_list + [
                    "bash", "-c", master_cmd,
                ]

                run_srun(master_srun_cmd, dry_run=dry_run, background=True,
                         log_file=srun_log_file)

            print(f"Master URLs: {','.join(master_urls)}")
            print("============================================")

    # ── Stage: harness ────────────────────────────────────────────────────

    if stage in ["harness", "all"]:
        if not master_urls:
            total_workers = num_gen_workers + num_ctx_workers
            for master_idx in range(num_master_servers):
                master_node_idx = master_idx % len(node_array)
                master_node = node_array[master_node_idx]
                master_port = args.base_port + total_workers + master_idx
                master_urls.append(f"{master_node}:{master_port}")

        master_urls_str = ",".join(master_urls)
        harness_run_args = args.harness_run_args

        if "--trtllm_server_urls=" not in harness_run_args:
            harness_run_args += f" --trtllm_server_urls={master_urls_str}"

        print("============================================")
        print("Launching MLPerf Harness")
        print(f"  Master URLs: {master_urls_str}")
        print(f"  Target: {harness_target}")
        print("============================================")

        # Ensure slurm_logs directory exists for harness output capture
        host_slurm_logs_dir = os.path.join(host_log_dir, "slurm_logs")
        if not dry_run:
            os.makedirs(host_slurm_logs_dir, exist_ok=True)

        harness_srun_cmd = base_srun_args + [
            f"--export={base_export},PYTHONPATH={container_vol}/src",
            "--nodes=1",
            f"--output={host_slurm_logs_dir}/run_harness.log",
            f"--error={host_slurm_logs_dir}/run_harness.log",
        ] + srun_flags_list + [
            "nv-mlpinf", harness_target,
            f"--system_name={harness_system}",
        ] + harness_run_args.split()

        run_srun(harness_srun_cmd,
                 env={"LOG_DIR": log_dir},
                 dry_run=dry_run, background=False, log_file=srun_log_file)


if __name__ == "__main__":
    main()
