#!/bin/bash
# NUMA binding wrapper for MPI ranks (used by B200_run_performance_server.sh)

RANK=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}

if [ "$RANK" -ge 0 ] && [ "$RANK" -le 3 ]; then
    # Ranks 0-3: NUMA node 0 (GPUs 0-3)
    exec numactl --cpunodebind=0 --membind=0 python -u $PYTHON_ARGS
elif [ "$RANK" -ge 4 ] && [ "$RANK" -le 8 ]; then
    # Ranks 4-8: NUMA node 1 (GPUs 4-7 + LoadGen)
    exec numactl --cpunodebind=1 --membind=1 python -u $PYTHON_ARGS
else
    echo "ERROR: Unsupported MPI rank: $RANK (expected 0-8)" >&2
    exit 1
fi
