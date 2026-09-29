# Qwen3-VL — GB300-NVL72 Offline (perf)

18-node × 4 GPUs (72 total) offline-scenario perf run. sflow port of the
multi-node `mlperf-inf-mm-q3vl benchmark nv mpi-dynamo-vllm-serve` flow.

## Launch

From `closed/NVIDIA/`:

```bash
SROOT=configs/qwen3_vl_235b_a22b
RUN=q3vl_gb300x72_offline
OUT=sflow_output/$RUN
ACCT=<your-slurm-account>
mkdir -p $OUT

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/GB300-NVL72_GB300-288GB_aarch64x72/VLLM/Offline/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HOME/.cache/huggingface \
  --set SLURM_ACCOUNT=$ACCT \
  --partition gb300 --account $ACCT --time 04:00:00 --nodes 18 \
  --output-dir $OUT --sbatch-path $OUT/sbatch.sh --submit
```

The shared workflow runs a head-node `prefetch_model` task before
`vllm_worker` starts. If `HF_CACHE_HOST_DIR` is set, it fills that mounted
cache; otherwise it falls back to `/work/.cache/huggingface` under `WORK_DIR`,
which must be shared across nodes. Set `HF_TOKEN=...` via `--set` if the model
repo is gated.

## Results

- Per-task logs: `$OUT/<jobid>-<workflow>-<ts>/{sflow.log,<task>/...}`
- Final QPS / TPS: `grep "QPS:" <jobid-dir>/sflow.log`
- Loadgen percentiles:
  `results/qwen3_vl_235b_a22b_shopify_benchmark_offline/report.txt`

## Per-recipe overrides

Adjust knobs at the top of `qwen3vl_config.yaml`; warmup count in
`endpoint.yaml` defaults to `TOTAL_GPUS × 400 = 28800` to match the
binary's `--dynamo.num_warmup_requests_per_vllm_instance=400`. Pass
`--set KEY=VALUE` to override at submit time without editing files.
