# Q3VL Benchmark — Run Server

Before running this, make sure the setup in `closed/Nebius/src/nv_mlpinf/benchmarks/q3vl/vllm/README.md` is done.

```bash
ACCT=root
PARTITION=main
SYSTEM=B300-SXM-270GBx8
NODES=1
TIME=04:00:00

SROOT=configs/qwen3_vl_235b_a22b
SFLOW_DUMP_DIR=build/sbatch_scripts_sflow

HF_CACHE_HOST_DIR=~/hf_cache
HF_TOKEN=hf_xxxxxxx

mkdir -p "$SFLOW_DUMP_DIR"

CONTAINER_IMAGE=~/v6.1-jul20-q3vl-amd64__latest.sqsh \
ENDPOINT_CONTAINER_IMAGE=~/mlcommons_endpoints__cc75b38e.sqsh
BACKEND_CONTAINER_NAME=q3vl-amd64-latest-$(date +%s)

RUN=q3vl_${SYSTEM}_server_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/${SYSTEM}/VLLM/Server/qwen3vl_config.yaml \
  --set CONTAINER_IMAGE=$CONTAINER_IMAGE \
  --set ENDPOINT_CONTAINER_IMAGE=$ENDPOINT_CONTAINER_IMAGE \
  --set BACKEND_CONTAINER_NAME=$BACKEND_CONTAINER_NAME \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HF_CACHE_HOST_DIR \
  --set HF_TOKEN=$HF_TOKEN \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --set SLURM_TIME=$TIME \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  --time=$TIME \
  -e "--exclusive" \
  -e "--nodelist=worker-1" \
  -e "--gpus-per-node=8" \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
```