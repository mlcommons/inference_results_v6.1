#!/bin/bash
#SBATCH --account=coreai_mlperf_inference
#SBATCH --partition=36x2-a01r
#SBATCH --time=01:00:00
#SBATCH --job-name=dlrmv3_perf_run
#SBATCH --output=dlrmv3_perf_run_%j.out

# ============================================================================
# MLPerf DLRMv3 - GB200 Cluster Script
# ============================================================================
# Usage (SCENARIO, MODE, OUTPUT_DIR are REQUIRED):
#   sbatch --nodes=N --segment=N GB200_run_performance_harness.sh SCENARIO=Server MODE=performance OUTPUT_DIR=results/Server [USER_CONF=path/to/conf]
#   sbatch --nodes=N --segment=N GB200_run_performance_harness.sh SCENARIO=Offline MODE=accuracy OUTPUT_DIR=results/Offline [USER_CONF=path/to/conf]
#
# Examples:
#   sbatch --nodes=4 --segment=4 GB200_run_performance_harness.sh SCENARIO=Server MODE=performance OUTPUT_DIR=results/Server
#   sbatch --nodes=18 --segment=18 GB200_run_performance_harness.sh SCENARIO=Offline MODE=accuracy OUTPUT_DIR=results/Offline
#   sbatch --nodes=4 --segment=4 GB200_run_performance_harness.sh SCENARIO=Server MODE=performance OUTPUT_DIR=results/Server USER_CONF=user_custom.conf
#
# NUMA Configuration:
#   - Server scenario:  NUMA binding ENABLED (optimal for latency)
#   - Offline scenario: NUMA binding DISABLED (not needed for throughput)
#
# GPU Batching & CUDA Streams:
#   - Server scenario:  --batching-on-gpu --use-cuda-streams (enabled)
#   - Offline scenario: disabled (not used)
# ============================================================================

export USER="${USER:-$(whoami)}"
export HOME="${HOME:-$HOME}"

# ============================================================================
# COMMAND LINE ARGUMENT PARSING (REQUIRED)
# ============================================================================

SCENARIO=""
MODE=""
OUTPUT_DIR=""
USER_CONF=""

for arg in "$@"; do
    case $arg in
        SCENARIO=*)
            SCENARIO="${arg#*=}"
            ;;
        MODE=*)
            MODE="${arg#*=}"
            ;;
        OUTPUT_DIR=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        USER_CONF=*)
            USER_CONF="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: sbatch --nodes=N --segment=N $0 SCENARIO=Server|Offline MODE=performance|accuracy OUTPUT_DIR=path/to/output [USER_CONF=path/to/conf]"
            exit 1
            ;;
    esac
done

# Validate required arguments
MISSING_ARGS=""
if [ -z "$SCENARIO" ]; then
    MISSING_ARGS="$MISSING_ARGS SCENARIO"
fi
if [ -z "$MODE" ]; then
    MISSING_ARGS="$MISSING_ARGS MODE"
fi
if [ -z "$OUTPUT_DIR" ]; then
    MISSING_ARGS="$MISSING_ARGS OUTPUT_DIR"
fi

if [ -n "$MISSING_ARGS" ]; then
    echo "ERROR: Missing required arguments:$MISSING_ARGS"
    echo "Usage: sbatch --nodes=N --segment=N $0 SCENARIO=Server|Offline MODE=performance|accuracy OUTPUT_DIR=path/to/output [USER_CONF=path/to/conf]"
    echo ""
    echo "Example:"
    echo "  sbatch --nodes=4 --segment=4 $0 SCENARIO=Offline MODE=performance OUTPUT_DIR=results/Offline"
    exit 1
fi


# =============================================================================
# USER CONFIGURATION - EDIT THESE VARIABLES
# =============================================================================

# Container configuration
CONTAINER_IMAGE="/lustre/fsw/coreai_mlperf_inference/zihaok/mlpinf+mlperf-inference+dlrm_images+dlrmv3-release-aarch64-Grace.sqsh"
CONTAINER_MOUNTS="/lustre/fsw/coreai_mlperf_inference/zihaok/mlperf-inference/closed/NVIDIA/src/nv_mlpinf/benchmarks/dlrm_v3:/work,/lustre/fsw/coreai_mlperf_inference/zihaok:/lustre/fsw/coreai_mlperf_inference/zihaok,/lustre/fsw/coreai_mlperf_inference/mlperf_inference_storage/:/lustre/fsw/coreai_mlperf_inference/mlperf_inference_storage/"

# Dataset and checkpoint paths
DATASET_PATH="/lustre/fsw/coreai_mlperf_inference/mlperf_inference_storage/preprocessed_data/dlrmv3/preprocess_final_1"
CHECKPOINT_PATH="/lustre/fsw/coreai_mlperf_inference/mlperf_inference_storage/models/dlrmv3/trained_checkpoint"

# Tunables
BATCH_SIZE=64
WARMUP_STEPS=1000
DATASET_PERCENTAGE=1
GPUS_PER_NODE=4

# Work Directories
SCRIPT_DIR=/work/benchmarks

# Set default user config if not provided
if [ -z "$USER_CONF" ]; then
    USER_CONF="$SCRIPT_DIR/user_gb200.conf"
fi

# Set Python prefix
PYTHON_PREFIX="python"

# Optional NSYS profiling (set NSYS=1 when launching)
NSYS="${NSYS:-0}"
NSYS_WRAPPER="/work/benchmarks/cluster_commands/nsys_wrapper.sh"
RUNNER_PREFIX=""
if [ "$NSYS" = "1" ]; then
    if [[ "$OUTPUT_DIR" = /* ]]; then
        NSYS_OUTPUT_DIR_DEFAULT="${OUTPUT_DIR}/nsys_profiles"
    else
        NSYS_OUTPUT_DIR_DEFAULT="/work/${OUTPUT_DIR}/nsys_profiles"
    fi
    export NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-$NSYS_OUTPUT_DIR_DEFAULT}"
    RUNNER_PREFIX="$NSYS_WRAPPER"
fi

# ============================================================================
# NODE & MPI SETUP
# ============================================================================

# Get the list of allocated nodes
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODES_ARRAY=($NODELIST)
NUM_SHARDS=${#NODES_ARRAY[@]}

# LoadGen runs on the last node
LAST_NODE_IDX=$((NUM_SHARDS - 1))
LOADGEN_NODE=${NODES_ARRAY[$LAST_NODE_IDX]}
LOADGEN_HOSTNAME="${LOADGEN_NODE}.ptyche.clusters.nvidia.com"

# ============================================================================
# SCENARIO-SPECIFIC CONFIGURATION
# ============================================================================

NUMA_MODE="on"
ADDITIONAL_ARGS="--batching-on-gpu --use-cuda-streams"

echo "============================================"
echo "MLPerf DLRMv3 - GB200 Cluster Run"
echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Nodes:       $NUM_SHARDS (${SLURM_JOB_NODELIST})"
echo "First Node:  ${NODES_ARRAY[0]}"
echo "Last Node:   ${LOADGEN_NODE} (LoadGen)"
echo "Scenario:    $SCENARIO"
echo "Mode:        $MODE"
echo "Output Dir:  $OUTPUT_DIR"
echo "User Config: $USER_CONF"
echo "Batch Size:  $BATCH_SIZE"
echo "NUMA Mode:   $NUMA_MODE"
echo "Additional Args: $ADDITIONAL_ARGS"
if [ "$NSYS" = "1" ]; then
    echo "NSYS Profiling: ENABLED"
else
    echo "NSYS Profiling: DISABLED"
fi
echo "============================================"
echo ""

# Build Python command
PYTHON_CMD="$RUNNER_PREFIX $PYTHON_PREFIX $SCRIPT_DIR/run_benchmark.py \
    --dataset-path $DATASET_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    --scenario $SCENARIO \
    --mode $MODE \
    --use-mpi-lookup \
    --batch-size $BATCH_SIZE \
    --warmup-steps $WARMUP_STEPS \
    --dataset-percentage $DATASET_PERCENTAGE \
    --user-conf $USER_CONF \
    --output-dir $OUTPUT_DIR \
    --communicator-type zmq \
    --loadgen-hostname $LOADGEN_HOSTNAME \
    --num-shards $NUM_SHARDS \
    --gpus-per-node $GPUS_PER_NODE \
    $ADDITIONAL_ARGS"

# ============================================================================
# NUMA BINDING CONFIGURATION
# ============================================================================
# Grace Blackwell topology:
#   NUMA 0: CPUs 0-71,  GPU 0,1
#   NUMA 1: CPUs 72-143, GPU 2,3
#
# For 4 worker ranks per node:
#   Task 0 -> GPU0 -> should run on NUMA0
#   Task 1 -> GPU1 -> should run on NUMA0  
#   Task 2 -> GPU2 -> should run on NUMA1
#   Task 3 -> GPU3 -> should run on NUMA1
# ============================================================================

if [ "$NUMA_MODE" = "on" ]; then
    echo "NUMA binding ENABLED (Server scenario - optimal for latency):"
    echo "  --gpu-bind=map_gpu:0,1,2,3"
    echo "  --cpu-bind=map_ldom:0,0,1,1  (Task 0,1->NUMA0, Task 2,3->NUMA1)"
    echo "  --mem-bind=local (except Node0 Rank0 uses both NUMA nodes)"
    echo ""
    
    # For node 0 (4 tasks) - rank 0 needs both NUMA nodes for weight loading
    # mask_mem: 0x1=NUMA0, 0x2=NUMA1, 0x3=both
    NUMA_OPTS_NODE0="--gpu-bind=map_gpu:0,1,2,3 --cpu-bind=map_ldom:0,0,1,1 --mem-bind=mask_mem:0x3,0x1,0x2,0x2"
    # For regular nodes (4 tasks) - all ranks use local mem binding
    NUMA_OPTS_4="--gpu-bind=map_gpu:0,1,2,3 --cpu-bind=map_ldom:0,0,1,1 --mem-bind=local"
    # For last node (5 tasks: 4 workers + 1 loadgen)
    # Loadgen (task 4) runs on NUMA1 since it's CPU-only
    NUMA_OPTS_5="--gpu-bind=map_gpu:0,1,2,3,3 --cpu-bind=map_ldom:0,0,1,1,1 --mem-bind=local"
else
    echo "NUMA binding DISABLED (Offline scenario - not needed for throughput):"
    echo "  --gpu-bind=map_gpu:0,1,2,3 (GPU binding only)"
    echo ""
    
    NUMA_OPTS_NODE0="--gpu-bind=map_gpu:0,1,2,3"
    NUMA_OPTS_4="--gpu-bind=map_gpu:0,1,2,3"
    NUMA_OPTS_5="--gpu-bind=map_gpu:0,1,2,3,3"
fi

echo "============================================"
echo "Starting Run..."
echo "============================================"
echo ""

# ============================================================================
# DYNAMIC MPMD LAUNCH
# ============================================================================

# Build srun command dynamically using array indexing
SRUN_CMD="srun --mpi=pmix \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts=$CONTAINER_MOUNTS \
    --container-mount-home \
    --container-workdir=/work \
    --no-container-remap-root"

# Add node 0 (with special NUMA binding for rank 0)
SRUN_CMD="$SRUN_CMD --nodelist=${NODES_ARRAY[0]} --ntasks=4 $NUMA_OPTS_NODE0 $PYTHON_CMD"

# Add middle nodes (1 to NUM_SHARDS-2)
for i in $(seq 1 $((NUM_SHARDS - 2))); do
    SRUN_CMD="$SRUN_CMD : --nodelist=${NODES_ARRAY[$i]} --ntasks=4 $NUMA_OPTS_4 $PYTHON_CMD"
done

# Add last node (with LoadGen)
SRUN_CMD="$SRUN_CMD : --nodelist=${NODES_ARRAY[$LAST_NODE_IDX]} --ntasks=5 $NUMA_OPTS_5 $PYTHON_CMD"

# Execute the command
eval $SRUN_CMD

EXIT_CODE=$?

# ============================================================================
# POST-RUN PROCESSING
# ============================================================================

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Run Finished Successfully!"
    echo "============================================"
    echo "Scenario:   $SCENARIO"
    echo "Mode:       $MODE"
    echo "NUMA Mode:  $NUMA_MODE"
    if [ "$NUMA_MODE" = "on" ]; then
        echo "  (NUMA binding enabled - optimal performance)"
    else
        echo "  (NUMA binding disabled)"
    fi
    echo "Results saved to: $OUTPUT_DIR"
    echo "============================================"
    
    # Run accuracy check if in accuracy mode
    if [ "$MODE" = "accuracy" ]; then
        echo ""
        echo "============================================"
        echo "Running Accuracy Check"
        echo "============================================"
        ACCURACY_LOG_PATH="${OUTPUT_DIR}/mlperf_log_accuracy.json"
        
        # Run accuracy checker on 1 node, 1 rank
        srun --mpi=pmix \
            --container-image=$CONTAINER_IMAGE \
            --container-mounts=$CONTAINER_MOUNTS \
            --container-mount-home \
            --container-workdir=/work \
            --no-container-remap-root \
            --nodes=1 \
            --ntasks=1 \
            python /work/benchmarks/accuracy.py --path $ACCURACY_LOG_PATH &> $OUTPUT_DIR/accuracy.txt
        
        echo ""
        echo "============================================"
        echo "Accuracy Check Complete!"
        echo "============================================"
    fi
else
    echo ""
    echo "============================================"
    echo "Run Failed with exit code $EXIT_CODE"
    echo "============================================"
    exit $EXIT_CODE
fi

exit $EXIT_CODE
