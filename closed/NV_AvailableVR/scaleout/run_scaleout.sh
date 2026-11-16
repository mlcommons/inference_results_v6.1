#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

stage="server"
harness_system=""
harness_run_args=""
container_image=""
mlperf_scratch_path="/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone"
extra_srun_flags=""
base_port=30000
server_spawn_time=0
dry_run=false
nodelist=""
jobid=""
log_dir=""
gpu_offset=0
harness_target="run_harness"
workspace_override=""
server_topology=""
dp_multiplicity_override=""
nsys_enabled=false
nsys_config=""

# Helper function to format command with indentation
format_command() {
    local cmd="$1"
    shift
    echo "$cmd \\"
    local last_idx=$(($# - 1))
    local idx=0
    for arg in "$@"; do
        if [[ $idx -eq $last_idx ]]; then
            echo "    $arg"
        else
            echo "    $arg \\"
        fi
        idx=$((idx + 1))
    done
}

# Helper function to execute srun with command printing
run_srun() {
    {
        flock -x 9
        echo "================================================"
        echo "Executing srun command:"
        echo "================================================"
        format_command "srun" "$@"
        echo ""
        if [[ -n "${HOST_LOG_DIR:-}" ]]; then
            format_command "srun" "$@" >> "${HOST_LOG_DIR}/srun_commands.log"
            echo "" >> "${HOST_LOG_DIR}/srun_commands.log"
            echo "" >> "${HOST_LOG_DIR}/srun_commands.log"
        fi
    } 9>"/tmp/run_scaleout_srun_${SLURM_JOBID:-$$}.lock"
    if [[ "$dry_run" == "false" ]]; then
        srun "$@"
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage=*)
            stage="${1#*=}"
            shift
            ;;
        --stage)
            stage=$2
            shift 2
            ;;
        --harness-system=*)
            harness_system="${1#*=}"
            shift
            ;;
        --harness-system)
            harness_system=$2
            shift 2
            ;;
        --harness-run-args=*)
            harness_run_args="${1#*=}"
            shift
            ;;
        --harness-run-args)
            harness_run_args=$2
            shift 2
            ;;
        --container-image=*)
            container_image="${1#*=}"
            shift
            ;;
        --container-image)
            container_image=$2
            shift 2
            ;;
        --mlperf-scratch-path=*)
            mlperf_scratch_path="${1#*=}"
            shift
            ;;
        --mlperf-scratch-path)
            mlperf_scratch_path=$2
            shift 2
            ;;
        --extra-srun-flags=*)
            extra_srun_flags="${1#*=}"
            shift
            ;;
        --extra-srun-flags)
            extra_srun_flags=$2
            shift 2
            ;;
        --base-port=*)
            base_port="${1#*=}"
            shift
            ;;
        --base-port)
            base_port=$2
            shift 2
            ;;
        --server-spawn-time=*)
            server_spawn_time="${1#*=}"
            shift
            ;;
        --server-spawn-time)
            server_spawn_time=$2
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --jobid=*)
            jobid="${1#*=}"
            shift
            ;;
        --jobid)
            jobid=$2
            shift 2
            ;;
        --nodelist=*|-w=*)
            nodelist="${1#*=}"
            shift
            ;;
        --nodelist|-w)
            nodelist=$2
            shift 2
            ;;
        --log-dir=*)
            log_dir="${1#*=}"
            shift
            ;;
        --log-dir)
            log_dir=$2
            shift 2
            ;;
        --gpu-offset=*)
            gpu_offset="${1#*=}"
            shift
            ;;
        --gpu-offset)
            gpu_offset=$2
            shift 2
            ;;
        --audit)
            harness_target="run_audit_harness"
            shift
            ;;
        --workspace=*)
            workspace_override="${1#*=}"
            shift
            ;;
        --workspace)
            workspace_override=$2
            shift 2
            ;;
        --server-topology=*)
            server_topology="${1#*=}"
            shift
            ;;
        --server-topology)
            server_topology=$2
            shift 2
            ;;
        --dp-multiplicity=*)
            dp_multiplicity_override="${1#*=}"
            shift
            ;;
        --dp-multiplicity)
            dp_multiplicity_override=$2
            shift 2
            ;;
        --nsys)
            nsys_enabled=true
            shift
            ;;
        --nsys-config=*)
            nsys_config="${1#*=}"
            shift
            ;;
        --nsys-config)
            nsys_config=$2
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --stage <server|harness|all> --server-topology <path> [OPTIONS]"
            echo ""
            echo "Note: All options accept both --key=value and --key value formats"
            echo ""
            echo "Required:"
            echo "  --stage <server|harness|all> Stage: server, harness, or all (runs server then harness)"
            echo ""
            echo "Topology:"
            echo "  --server-topology <path>     Path to server-topology.json (provides model_path, dp_multiplicity, gpus_per_dp_rank, gpus_per_node, trtllm_yml, env_yml)"
            echo ""
            echo "Optional:"
            echo "  --harness-system <name>      Override harness system (default: calculated from atomic-system x dp-multiplicity)"
            echo "  --jobid <id>                 SLURM job ID to target (default: use \$SLURM_JOBID from environment)"
            echo "  --container-image <path>     Container image (default: build/sqsh_images/mlperf-inference-\$USER-aarch64-release.sqsh)"
            echo "  --mlperf-scratch-path <path> Scratch path (default: /lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone)"
            echo "  --extra-srun-flags <flags>   Additional srun flags"
            echo "  --base-port <N>              Base port (default: 30000)"
            echo "  --server-spawn-time <sec>    Sleep time after launching servers in seconds (default: 0)"
            echo "  --nodelist <nodes>           Comma-separated node list (default: use SLURM_JOB_NODELIST)"
            echo "  -w <nodes>                   Alias for --nodelist"
            echo "  --log-dir <path>             Log directory (default: auto-generated with timestamp)"
            echo "  --gpu-offset <N>             Starting GPU index for intra-node deployments (default: 0)"
            echo "  --harness-run-args <args>    Extra arguments passed to nv-mlpinf harness command"
            echo "  --dp-multiplicity <N>        Override dp_multiplicity from server-topology.json"
            echo "  --nsys                       Wrap trtllm-serve with 'nsys profile' for per-rank profile collection"
            echo "  --nsys-config <path>         Path to nsys YAML config (default: <workspace>/internal/nsys/sample_nsys_config.yml)"
            echo "  --audit                      Run audit harness (run_audit_harness) instead of regular harness"
            echo "  --dry-run                    Print srun commands without executing them"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Determine SLURM job ID: use --jobid if provided, otherwise fall back to $SLURM_JOBID
if [[ -n "$jobid" ]]; then
    # User provided explicit job ID
    SLURM_JOBID="$jobid"
    echo "Using explicit job ID: $SLURM_JOBID"
    # Add --jobid to srun flags
    extra_srun_flags="--jobid=$SLURM_JOBID $extra_srun_flags"
elif [[ -n "${SLURM_JOBID:-}" ]]; then
    # Already in SLURM allocation
    echo "Using SLURM_JOBID from environment: $SLURM_JOBID"
else
    echo "ERROR: No SLURM job ID found. Provide --jobid or run inside a SLURM allocation." >&2
    exit 1
fi

# Fetch SLURM_JOB_NODELIST from the job if --nodelist not provided
if [[ -z "$nodelist" ]]; then
    SLURM_JOB_NODELIST=$(scontrol show hostnames $(squeue -j "$SLURM_JOBID" -h -o "%N") | paste -sd "," - 2>/dev/null)
    if [[ -z "$SLURM_JOB_NODELIST" ]]; then
        echo "ERROR: Failed to fetch nodelist for job $SLURM_JOBID. Job may not exist or is not running." >&2
        exit 1
    fi
    echo "Fetched nodelist from job $SLURM_JOBID: $SLURM_JOB_NODELIST"
fi

# Determine workspace (host_vol) - can be overridden via --workspace
if [[ -n "$workspace_override" ]]; then
    # Use explicit workspace override
    host_vol="$(readlink -f "$workspace_override")"
    script_dir="$host_vol/scaleout"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    # sbatch: use the directory from which sbatch was invoked
    script_dir="$SLURM_SUBMIT_DIR/scaleout"
    host_vol="$(readlink -f "$SLURM_SUBMIT_DIR")"
else
    # script: use the directory containing the script
    script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    host_vol="$(readlink -f "$script_dir/..")"
fi
container_vol="/work"

# Read server topology JSON (required)
if [[ -z "$server_topology" ]]; then
    echo "ERROR: --server-topology is required" >&2
    exit 1
fi
[[ ! -f "$server_topology" ]] && { echo "ERROR: server-topology.json not found: $server_topology" >&2; exit 1; }
eval "$(python3 "$script_dir/parse_topology.py" --host-vol "$host_vol" --container-vol "$container_vol" "$server_topology")"

dp_multiplicity="$TOPO_DP_MULTIPLICITY"
gpus_per_node="$TOPO_GPUS_PER_NODE"
num_gpus_per_dp="$TOPO_GPUS_PER_DP_RANK"
topo_model_path="$TOPO_MODEL_PATH"
topo_trtllm_yml="$TOPO_TRTLLM_YML"
topo_env_exports="$TOPO_ENV_EXPORTS"

if [[ -n "$dp_multiplicity_override" ]]; then
    echo "Overriding dp_multiplicity: $dp_multiplicity -> $dp_multiplicity_override"
    dp_multiplicity="$dp_multiplicity_override"
fi

# Build the base srun --export string with env vars from topology
srun_base_exports="ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage"
if [[ -n "${topo_env_exports:-}" ]]; then
    srun_base_exports="${srun_base_exports},${topo_env_exports}"
fi

# ── nsys profiling ──────────────────────────────────────────────────────────
if [[ "$nsys_enabled" == "true" ]]; then
    if [[ -z "$nsys_config" ]]; then
        nsys_config="$host_vol/internal/nsys/sample_nsys_config.yml"
    fi
    if [[ ! -f "$nsys_config" ]]; then
        echo "ERROR: nsys config not found: $nsys_config" >&2
        exit 1
    fi
    eval "$(python3 - "$nsys_config" <<'PY'
import yaml, shlex, sys
c = yaml.safe_load(open(sys.argv[1]))
for k in ("nsys_path", "profile_name", "extra_flags"):
    if k not in c:
        sys.stderr.write(f"ERROR: nsys config missing required key: {k}\n")
        sys.exit(1)
flags = " ".join(shlex.quote(x) for x in c["extra_flags"])
print("nsys_path=" + shlex.quote(c["nsys_path"]))
print("nsys_profile_name=" + shlex.quote(c["profile_name"]))
print("nsys_extra_flags=" + shlex.quote(flags))
PY
)"
    echo "nsys: enabled (config: $nsys_config)"
fi

# Per-worker nsys prefix. $1 = worker tag used in output filename.
nsys_profile_prefix() {
    [[ "$nsys_enabled" != "true" ]] && { echo ""; return; }
    local tag="$1"
    echo "$nsys_path profile $nsys_extra_flags --output='${LOG_DIR}/${nsys_profile_name}-dp${tag}-rank'\${SLURM_PROCID:-0} "
}

if [[ -z "$harness_system" ]]; then
    echo "ERROR: --harness-system is required"
    exit 1
fi
echo "Using harness system: $harness_system"

# Determine log directory: use --log-dir if provided, otherwise auto-generate
if [[ -n "$log_dir" ]]; then
    # Use provided log directory (absolute path)
    HOST_LOG_DIR="$log_dir"
    LOG_DIR="${log_dir/#${host_vol}/${container_vol}}"
else
    # Auto-generate log directory with timestamp (using harness system)
    TIMESTAMP=$(date +'%Y.%m.%d-%H.%M.%S')
    LOG_DIR="/work/build/logs/scaleout_${harness_system}_slurm-${SLURM_JOBID}_${TIMESTAMP}"
    HOST_LOG_DIR="${LOG_DIR/#${container_vol}/${host_vol}}"
fi
export LOG_DIR
mkdir -p "$HOST_LOG_DIR"

# Copy topology and config files to log directory for debugging
if [[ -n "$server_topology" ]]; then
    topo_dir="$(dirname "$server_topology")"
    cp "$topo_dir"/* "$HOST_LOG_DIR/" 2>/dev/null || true
fi

if [[ -z "$container_image" ]]; then
    docker_tag=$(whoami)-aarch64
    container_image="$host_vol/build/sqsh_images/mlperf-inference-$docker_tag-release.sqsh"
fi

# Check for node list
[[ -z "${SLURM_JOB_NODELIST:-}" && -z "$nodelist" ]] && { echo "ERROR: No node list found. Provide --nodelist or run inside a SLURM allocation." >&2; exit 1; }

# Use provided nodelist or fall back to SLURM allocation
if [[ -n "$nodelist" ]]; then
    # User provided explicit nodelist
    job_nodelist="$nodelist"
    echo "Using explicit nodelist: $job_nodelist"
else
    # Use SLURM allocation
    job_nodelist="$SLURM_JOB_NODELIST"
    echo "Using SLURM allocation nodelist: $job_nodelist"
fi

allocated_node_count=$(scontrol show hostname "$job_nodelist" | wc -l)
allocated_nodes=$(scontrol show hostname "$job_nodelist")

total_gpus_allocated=$((allocated_node_count * gpus_per_node))
total_gpus_required=$((num_gpus_per_dp * dp_multiplicity))

# Validate that allocated resources match DP topology requirements
if [[ $total_gpus_allocated -lt $total_gpus_required ]]; then
    echo "ERROR: Insufficient GPU count" >&2
    echo "  Allocated: $total_gpus_allocated ($allocated_node_count nodes x $gpus_per_node GPUs/node)" >&2
    echo "  Required: $total_gpus_required ($dp_multiplicity DP ranks x $num_gpus_per_dp GPUs/rank)" >&2
    exit 1
fi

echo "============================================"
echo "MLPerf Scaleout - $stage"
echo "============================================"
[[ -n "$server_topology" ]] && echo "Server topology: $server_topology"
echo "Harness system: $harness_system"
echo "Model path: ${topo_model_path:-N/A}"
echo "GPUs per DP: $num_gpus_per_dp"
echo "GPUs per node: $gpus_per_node"
echo "DP multiplicity: $dp_multiplicity"
echo "Allocated nodes: $allocated_node_count"
echo "Total GPUs: $total_gpus_allocated"
echo "Log directory: $LOG_DIR"
echo "Harness target: $harness_target"
[[ -n "${topo_trtllm_yml:-}" ]] && echo "TRTLLM config: $topo_trtllm_yml"
[[ -n "${topo_env_exports:-}" ]] && echo "Env exports: $topo_env_exports"
echo "Server spawn time: ${server_spawn_time}s"
echo "Dry run: $dry_run"

# Generate server URLs for all DP ranks
declare -a server_url_array
if [[ $num_gpus_per_dp -ge $gpus_per_node ]]; then
    # Cross-node: each DP rank spans multiple nodes
    num_nodes_per_dp=$((num_gpus_per_dp / gpus_per_node))
    node_array=($allocated_nodes)
    for ((i=0; i<dp_multiplicity; i++)); do
        server_node_idx=$((i * num_nodes_per_dp))
        server_node=${node_array[$server_node_idx]}
        server_url_array[$i]="${server_node}:${base_port}"
    done
else
    # Intra-node: multiple DP ranks per node, each on different port
    dp_per_node=$((gpus_per_node / num_gpus_per_dp))
    rank_idx=0
    for node_name in $allocated_nodes; do
        for ((j=0; j<dp_per_node; j++)); do
            if [[ $rank_idx -ge $dp_multiplicity ]]; then
                break 2
            fi
            port=$((base_port + j))
            server_url_array[$rank_idx]="${node_name}:${port}"
            rank_idx=$((rank_idx + 1))
        done
    done
fi

if [[ "$stage" == "server" || "$stage" == "all" ]]; then
    if [[ $num_gpus_per_dp -ge $gpus_per_node ]]; then
        num_nodes_per_dp=$((num_gpus_per_dp / gpus_per_node))
        node_array=($allocated_nodes)

        echo "Deployment: Cross-node ($num_nodes_per_dp nodes per DP, $gpus_per_node tasks/node)"
        echo "============================================"

        for ((i=0; i<dp_multiplicity; i++)); do
            server_url="${server_url_array[$i]}"

            # Calculate node list for this DP rank
            start_node_idx=$((i * num_nodes_per_dp))
            end_node_idx=$((start_node_idx + num_nodes_per_dp - 1))
            node_list=""
            for ((n=start_node_idx; n<=end_node_idx; n++)); do
                if [[ -z "$node_list" ]]; then
                    node_list="${node_array[$n]}"
                else
                    node_list="${node_list},${node_array[$n]}"
                fi
            done

            echo "Launching DP rank $i (URL: $server_url, Nodes: $node_list)..."
            export DP_RANK=$i
            # CVD wrapper only when nsys is enabled (nsys --gpu-metrics-devices=cuda-visible
            # needs single-GPU per task). Without nsys it's harmful for MoE alltoall paths.
            cvd_prefix=""
            [[ "$nsys_enabled" == "true" ]] && cvd_prefix="export CUDA_VISIBLE_DEVICES=\${SLURM_LOCALID:-0} && "
            serve_cmd="${cvd_prefix}$(nsys_profile_prefix $i)trtllm-llmapi-launch trtllm-serve '${topo_model_path}' --host 0.0.0.0 --port ${server_url##*:} --extra_llm_api_options '${topo_trtllm_yml}'"
            run_srun \
                --overlap \
                --output="${HOST_LOG_DIR}/slurm_logs/run_llm_server_dp${i}_rank%t.log" \
                --export="${srun_base_exports}" \
                --container-image="$container_image" \
                --container-mounts="${host_vol}:${container_vol},${mlperf_scratch_path}:/home/mlperf_inference_storage" \
                --container-workdir="$container_vol" \
                --container-remap-root \
                --nodelist="$node_list" \
                --ntasks-per-node="$gpus_per_node" \
                --nodes="$num_nodes_per_dp" \
                --mpi=pmix \
                $extra_srun_flags \
                bash -c "$serve_cmd" &
        done
    else
        echo "Deployment: Intra-node ($dp_per_node DP/node, $num_gpus_per_dp tasks/DP)"
        echo "============================================"

        # Launch all DP ranks in parallel with unique IPC addresses per rank
        for ((j=0; j<dp_per_node; j++)); do
            node_idx=0
            # Unique IPC port for each DP rank on the same node
            ipc_port=$((10012 + j))
            ipc_addr="tcp://127.0.0.1:${ipc_port}"
            for node_name in $allocated_nodes; do
                rank_idx=$((node_idx * dp_per_node + j))
                if [[ $rank_idx -ge $dp_multiplicity ]]; then
                    node_idx=$((node_idx + 1))
                    continue
                fi
                start_gpu=$((gpu_offset + j * num_gpus_per_dp))
                gpu_list=$(seq -s, $start_gpu $((start_gpu + num_gpus_per_dp - 1)))
                server_url="${server_url_array[$rank_idx]}"

                echo "Launching DP rank $rank_idx on $node_name (GPUs: $gpu_list, URL: $server_url, IPC: $ipc_addr)..."
                # CVD wrapper only when nsys is enabled — see comment above.
                cvd_prefix=""
                [[ "$nsys_enabled" == "true" ]] && cvd_prefix="export CUDA_VISIBLE_DEVICES=\${SLURM_LOCALID:-0} && "
                serve_cmd="${cvd_prefix}$(nsys_profile_prefix $rank_idx)trtllm-llmapi-launch trtllm-serve '${topo_model_path}' --host 0.0.0.0 --port ${server_url##*:} --extra_llm_api_options '${topo_trtllm_yml}'"
                export NVIDIA_VISIBLE_DEVICES="${gpu_list}"
                run_srun --overlap \
                    --output="${HOST_LOG_DIR}/slurm_logs/run_llm_server_dp${rank_idx}_rank%t.log" \
                    --export="${srun_base_exports},TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR=${ipc_addr},NVIDIA_VISIBLE_DEVICES" \
                    --container-image="$container_image" \
                    --container-mounts="${host_vol}:${container_vol},${mlperf_scratch_path}:/home/mlperf_inference_storage" \
                    --container-workdir="$container_vol" \
                    --container-remap-root \
                    --nodes=1 \
                    --nodelist="$node_name" \
                    --ntasks-per-node="$num_gpus_per_dp" \
                    --mpi=pmix \
                    $extra_srun_flags \
                    bash -c "$serve_cmd" &
                unset NVIDIA_VISIBLE_DEVICES

                node_idx=$((node_idx + 1))
            done
            # Optional sleep to allow servers to fully initialize
            if [ $server_spawn_time -gt 0 ] && [ "$dry_run" == "false" ]; then
                echo "Sleeping for ${server_spawn_time} seconds to avoid IPC port collision race condition. Increase --server-spawn-time if you see ZMQ errors."
                sleep "$server_spawn_time"
            fi
        done
    fi
    echo "All DP ranks launched in background. Check logs for details."

    if [[ "$stage" == "server" ]]; then
        echo "Waiting for all background server processes to finish (Ctrl+C or scancel to stop)..."
        wait
        echo "All server processes have exited."
    fi
fi

if [[ "$stage" == "harness" || "$stage" == "all" ]]; then
    server_urls=$(IFS=,; echo "${server_url_array[*]}")

    [[ ! "$harness_run_args" =~ --trtllm_server_urls= ]] && harness_run_args="$harness_run_args --trtllm_server_urls=$server_urls"

    echo "Server URLs: $server_urls"
    echo "============================================"
    echo "Launching harness (target: $harness_target)..."

    run_srun \
        --export="ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage,PYTHONPATH=${container_vol}/src" \
        --container-image="$container_image" \
        --container-mounts="${host_vol}:${container_vol},${mlperf_scratch_path}:/home/mlperf_inference_storage" \
        --container-workdir="$container_vol" \
        --container-remap-root \
        --output="${HOST_LOG_DIR}/slurm_logs/run_harness.log" \
        --error="${HOST_LOG_DIR}/slurm_logs/run_harness.log" \
        --nodes=1 \
        --overlap \
        $extra_srun_flags \
        nv-mlpinf ${harness_target} --system_name=$harness_system $harness_run_args
fi
