#!/bin/bash
#SBATCH --account=coreai_mlperf_inference
#SBATCH --partition=36x2-a01r
#SBATCH --job-name=coreai_mlperf_inference-debug.pmix-tensorrtllm
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:10:00
#SBATCH --output=pmix_tensorrtllm_%j.out
#SBATCH --error=pmix_tensorrtllm_%j.err

set -euo pipefail

MPI_MODE="${MPI_MODE:-pmix}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker://nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.1}"

echo "===== job configuration ====="
date -Is
echo "submit_host=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_job_nodelist=${SLURM_JOB_NODELIST}"
echo "slurm_partition=${SLURM_JOB_PARTITION}"
echo "slurm_account=${SLURM_JOB_ACCOUNT}"
echo "mpi_mode=${MPI_MODE}"
echo "container_image=${CONTAINER_IMAGE}"

echo
echo "===== Slurm advertised MPI plugins ====="
srun --version
srun --mpi=list

echo
echo "===== Host-side PMIx environment test ====="
srun \
  --mpi="${MPI_MODE}" \
  --nodes=1 \
  --ntasks=2 \
  --ntasks-per-node=2 \
  --label \
  bash -lc '
    set -euo pipefail
    echo "host=$(hostname) arch=$(uname -m) proc=${SLURM_PROCID:-unset} localid=${SLURM_LOCALID:-unset}"
    echo "SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-unset}"
    echo "PMIX_RANK=${PMIX_RANK:-unset} PMIX_NAMESPACE=${PMIX_NAMESPACE:-unset} PMIX_VERSION=${PMIX_VERSION:-unset}"
    echo "PMI_RANK=${PMI_RANK:-unset} PMI_SIZE=${PMI_SIZE:-unset}"
    test -n "${PMIX_RANK:-${PMI_RANK:-}}"
  '

echo
echo "===== Container-side Open MPI / PMIx test ====="
srun \
  --mpi="${MPI_MODE}" \
  --nodes=1 \
  --ntasks=2 \
  --ntasks-per-node=2 \
  --label \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mount-home \
  --container-remap-root \
  bash -lc '
    set -euo pipefail
    export OMPI_MCA_orte_base_help_aggregate=0
    export OMPI_MCA_prte_base_help_aggregate=0

    echo "container_host=$(hostname) arch=$(uname -m) proc=${SLURM_PROCID:-unset} localid=${SLURM_LOCALID:-unset}"
    echo "SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-unset}"
    echo "PMIX_RANK=${PMIX_RANK:-unset} PMIX_NAMESPACE=${PMIX_NAMESPACE:-unset} PMIX_VERSION=${PMIX_VERSION:-unset}"
    echo "PMI_RANK=${PMI_RANK:-unset} PMI_SIZE=${PMI_SIZE:-unset}"

    echo "--- MPI tools visible in the container ---"
    command -v mpirun || true
    command -v mpiexec || true
    command -v ompi_info || true
    command -v trtllm-llmapi-launch || true
    command -v mpirun >/dev/null && mpirun --version | head -30 || true

    echo "--- mpi4py COMM_WORLD smoke test ---"
    python3 - <<'"'"'PY'"'"'
import os
import sys

print("python", sys.version.replace("\n", " "))
print("PMIX_RANK", os.environ.get("PMIX_RANK"), "PMIX_NAMESPACE", os.environ.get("PMIX_NAMESPACE"), "PMIX_VERSION", os.environ.get("PMIX_VERSION"))

from mpi4py import MPI

comm = MPI.COMM_WORLD
print(f"mpi4py rank={comm.Get_rank()} size={comm.Get_size()}")
PY
  '

echo
echo "===== result ====="
echo "PASS: Slurm PMIx env was visible and mpi4py COMM_WORLD initialized inside the container."
