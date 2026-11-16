#!/bin/bash
# nsys wrapper script for multi-rank profiling
# Only collects system-wide metrics (nic, gpu) on local rank 0 per node

NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-/work/nsys_profiles/}"


if [ "$SLURM_LOCALID" -eq 0 ]; then
    mkdir -p "$NSYS_OUTPUT_DIR"
    # Only profile local rank 0 per node
    echo "[Rank $SLURM_PROCID / LocalRank $SLURM_LOCALID] Running with nsys profiling"
    nsys profile -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 \
        --trace=cuda,nvtx,osrt,python-gil,mpi \
        -o "${NSYS_OUTPUT_DIR}/profile_node%q{SLURM_NODEID}_rank%q{SLURM_PROCID}_offline_bs64_stream_on" \
        --force-overwrite true \
        --gpu-metrics-devices=all \
        --delay=800 \
        "$@"
else
    # Other ranks: no profiling, just run the command
    echo "[Rank $SLURM_PROCID / LocalRank $SLURM_LOCALID] Running WITHOUT nsys profiling"
    "$@"
fi

