#!/bin/bash
#SBATCH --account=coreai_mlperf_inference
#SBATCH --partition=36x2-a01r
#SBATCH --job-name=coreai_mlperf_inference-debug.cpus-default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:05:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "=== Purpose ==="
echo "This job intentionally omits #SBATCH --cpus-per-task."
echo "It asks for one node and four tasks per node, then prints Slurm's effective job and step state."
echo

echo "=== Job environment ==="
echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-UNSET}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-UNSET}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-UNSET}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-UNSET}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-UNSET}"
echo

echo "=== scontrol show job -dd selected fields ==="
scontrol show job -dd "${SLURM_JOB_ID}" | tr ' ' '\n' | grep -E '^(JobId|JobName|Partition|NodeList|NumNodes|NumCPUs|NumTasks|CPUs/Task|ReqTRES|AllocTRES|TRES)=' || true
echo

echo "=== Full job record ==="
scontrol show job -dd "${SLURM_JOB_ID}"
echo

echo "=== Nested srun without --cpus-per-task ==="
srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --label bash -lc '
  echo "rank=${SLURM_PROCID} localid=${SLURM_LOCALID} host=$(hostname) SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-UNSET}"
'
