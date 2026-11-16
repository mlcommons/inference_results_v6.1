#!/bin/bash
# Local MPI run script for MLPerf LoadGen with DLRMv3 for B200x8 single node run
# Uses ZMQ communicator similar to cluster setup
# Edit the variables below to match your setup
#
# =============================================================================
# NUMA Configuration
# =============================================================================
# Default: NUMA binding enabled for optimal performance
# Usage (all three arguments are REQUIRED):
#   ./B200_run_performance_harness.sh SCENARIO=Server MODE=performance OUTPUT_DIR=dlrmv3/Server/performance
#   ./B200_run_performance_harness.sh SCENARIO=Offline MODE=accuracy OUTPUT_DIR=dlrmv3/Offline/accuracy
#   NUMA_MODE=off ./B200_run_performance_harness.sh SCENARIO=Server MODE=performance OUTPUT_DIR=output  # Without NUMA
#
# NUMA Modes:
#   on  - Ranks 0-3 on NUMA0 (GPU0-3), Ranks 4-7 on NUMA1 (GPU4-7),
#                  Rank 8 (LoadGen) on NUMA1
#   off - No CPU/memory binding
#
# NUMA binding ensures that each worker process runs on CPUs that are on the
# same NUMA node as its GPU, minimizing memory access latency and maximizing
# performance, this is crucial in Server scenario to make latency abide the constraint.
# =============================================================================


# =============================================================================
# COMMAND LINE ARGUMENT PARSING (REQUIRED)
# =============================================================================
# Usage: bash B200_run_performance_harness.sh SCENARIO=Offline MODE=performance OUTPUT_DIR=dlrmv3/Offline/performance
# Required arguments: SCENARIO, MODE, OUTPUT_DIR

SCENARIO=""
MODE=""
OUTPUT_DIR=""

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
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 SCENARIO=Server|Offline MODE=performance|accuracy OUTPUT_DIR=path/to/output"
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
    echo "Usage: $0 SCENARIO=Server|Offline MODE=performance|accuracy OUTPUT_DIR=path/to/output"
    echo ""
    echo "Example:"
    echo "  $0 SCENARIO=Offline MODE=performance OUTPUT_DIR=dlrmv3/Offline/performance"
    exit 1
fi


# =============================================================================
# USER CONFIGURATION - EDIT THESE VARIABLES
# =============================================================================

# Path to your preprocessed dataset (REQUIRED - change this!)
DATASET_PATH="/home/mlperf_inference_storage_01/preprocessed_data/dlrmv3/preprocess_final_1/"
CHECKPOINT_PATH="/raid/data/zihaok/trained_checkpoint/"

# Batch size (only applies to Server scenario)
BATCH_SIZE=64
WARMUP_STEPS=1000
# Maximum samples to load into memory
DATASET_PERCENTAGE=1


# =============================================================================
# MPI / CLUSTER-STYLE CONFIGURATION
# =============================================================================

# Number of MPI processes (workers + 1 loadgen process)
# Adjust based on available GPUs. Default: 8 workers + 1 loadgen = 9 total
NUM_WORKERS=8
NUM_TOTAL_PROCESSES=$((NUM_WORKERS + 1))
NUM_SHARDS=1
GPUS_PER_NODE=8
LOADGEN_HOSTNAME="localhost"
COMMUNICATOR_TYPE="zmq"

# =============================================================================
# NSYS PROFILING CONFIGURATION
# =============================================================================

# Nsys profiling settings (set NSYS=1 to enable)
# Example: NSYS=1 ./benchmarks/run_loadgen_example_loadgen_separate_performance.sh
NSYS_OUTPUT_NAME="server_bs64_b200"
NSYS_GPU_DEVICES="0"  # or specify like "0,1,2,3"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check if dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset path does not exist: $DATASET_PATH"
    echo "Please edit this script and set DATASET_PATH to your dataset location."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "============================================"
echo "MLPerf LoadGen for DLRMv3 - Local MPI Run"
echo "(Adapted from cluster script)"
echo "============================================"
echo "User Configuration:"
echo "  Dataset Path:    $DATASET_PATH"
echo "  Checkpoint Path: $CHECKPOINT_PATH"
echo "  Scenario:        $SCENARIO"
echo "  Mode:            $MODE"
echo "  Batch Size:      $BATCH_SIZE"
echo "  Warmup Steps:    $WARMUP_STEPS"
echo "  Dataset %:       $DATASET_PERCENTAGE"
echo "  Output Dir:      $OUTPUT_DIR"
echo "============================================"
echo "Communicator Configuration:"
echo "  Num Workers:     $NUM_WORKERS"
echo "  Total Processes: $NUM_TOTAL_PROCESSES"
echo "  Num Shards:      $NUM_SHARDS"
echo "  GPUs Per Node:   $GPUS_PER_NODE"
echo "  LoadGen Host:    $LOADGEN_HOSTNAME"
echo "  Communicator:    $COMMUNICATOR_TYPE"
echo "============================================"
echo ""

# Build base python arguments (using cluster-style script)
PYTHON_ARGS="$SCRIPT_DIR/run_benchmark.py \
    --dataset-path $DATASET_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    --scenario $SCENARIO \
    --mode $MODE \
    --use-mpi-lookup \
    --batch-size $BATCH_SIZE \
    --warmup-steps $WARMUP_STEPS \
    --dataset-percentage $DATASET_PERCENTAGE \
    --user-conf $SCRIPT_DIR/user_b200.conf \
    --output-dir $OUTPUT_DIR \
    --communicator-type $COMMUNICATOR_TYPE \
    --loadgen-hostname $LOADGEN_HOSTNAME \
    --num-shards $NUM_SHARDS \
    --gpus-per-node $GPUS_PER_NODE \
    --batching-on-gpu \
    --use-cuda-streams" 

# =============================================================================
# NUMA Binding Configuration (Server mode only)
# =============================================================================
echo "NUMA binding ENABLED (Server scenario - OPTIMAL for latency):"
echo "  Ranks 0-3 (GPU 0-3) -> NUMA node 0"
echo "  Ranks 4-7 (GPU 4-7) -> NUMA node 1"
echo "  Rank 8 (LoadGen)    -> NUMA node 1"
echo ""
echo "Using numactl for CPU and memory binding"

# Use repo wrapper script for NUMA binding
NUMA_WRAPPER="$SCRIPT_DIR/numa_wrapper.sh"
chmod +x "$NUMA_WRAPPER"

# Build mpirun command with NUMA wrapper
export PYTHON_ARGS
PYTHON_CMD="mpirun -n $NUM_TOTAL_PROCESSES --bind-to none $NUMA_WRAPPER"

echo "============================================"
echo ""

# Run with or without nsys profiling
if [ "${NSYS:-0}" -eq 1 ]; then
    echo "Running with NVIDIA Nsight Systems profiling..."
    echo ""

    nsys profile \
        --trace=cuda,nvtx,osrt,python-gil,mpi \
        -o "$NSYS_OUTPUT_NAME" \
        --force-overwrite true \
        --delay=1080 \
        --gpu-metrics-devices="$NSYS_GPU_DEVICES" \
        --duration=120 \
        $PYTHON_CMD
    EXIT_CODE=$?
else
    echo "Starting MPI processes..."
    echo ""
    $PYTHON_CMD
    EXIT_CODE=$?
fi

# Handle exit status
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Test completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    if [ "$NUMA_MODE" == "on" ]; then
        echo "NUMA binding: ENABLED (optimal performance)"
    else
        echo "NUMA binding: DISABLED"
    fi
    echo "============================================"
    
    # Run accuracy check if in accuracy mode
    if [ "$MODE" = "accuracy" ]; then
        echo "Running accuracy check..."
        python $SCRIPT_DIR/accuracy.py --path $OUTPUT_DIR/mlperf_log_accuracy.json &> $OUTPUT_DIR/accuracy.txt
        exit $?
    fi
else
    echo ""
    echo "============================================"
    echo "Test failed with exit code: $EXIT_CODE"
    echo "============================================"
    exit $EXIT_CODE
fi

