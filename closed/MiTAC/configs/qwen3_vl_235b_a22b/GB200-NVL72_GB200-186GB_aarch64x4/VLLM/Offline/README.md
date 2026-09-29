# Qwen3-VL — GB200-NVL4 Offline (perf)

Single-node × 4 GPUs offline-scenario perf run.

## Launch

From `closed/NVIDIA/`:

```bash
SROOT=configs/qwen3_vl_235b_a22b
RUN=q3vl_nvl4_offline_$(date +%Y%m%d-%H%M%S)
OUT=scaleout/sflow_output/$RUN
ACCT=<your-slurm-account>

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/GB200-NVL72_GB200-186GB_aarch64x4/VLLM/Offline/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HOME/.cache/huggingface \
  --set SLURM_ACCOUNT=$ACCT \
  --partition gb200 --account $ACCT --time 04:00:00 --nodes 1 \
  --output-dir $OUT --sbatch-path $OUT/sbatch.sh --submit
```

The shared workflow runs a head-node `prefetch_model` task before
`vllm_worker` starts. If `HF_CACHE_HOST_DIR` is set, it fills that mounted
cache; otherwise it falls back to `/work/.cache/huggingface` under `WORK_DIR`,
which all tasks share through the `/work` mount. Set `HF_TOKEN=...` via `--set`
if the model repo is gated.

## Results

- Per-task logs: `$OUT/<jobid>-<workflow>-<ts>/{sflow.log,<task>/...}`
- Final QPS / TPS: `grep "QPS:" <jobid-dir>/sflow.log`
- Loadgen percentiles (TTFT, TPOT, end-to-end latency):
  `results/qwen3_vl_235b_a22b_shopify_benchmark_offline_gb200x4/report.txt`

## Per-recipe overrides

Adjust knobs at the top of `qwen3vl_config.yaml` in this folder. Common ones:
`MAX_NUM_BATCHED_TOKENS`, `COMPILATION_CONFIG_JSON`, and the warmup count in
`endpoint.yaml`. Pass `--set KEY=VALUE` to override at submit time without
editing files.
