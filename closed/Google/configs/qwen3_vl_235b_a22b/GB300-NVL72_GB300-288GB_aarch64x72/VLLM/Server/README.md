# Qwen3-VL — GB300-NVL72 Server (perf)

18-node × 4 GPUs (72 total) server-scenario perf run (Poisson load at the
conservative `target_qps=1205`). sflow port of the multi-node server flow.

## Launch

From `closed/NVIDIA/`:

```bash
SROOT=configs/qwen3_vl_235b_a22b
RUN=q3vl_x72_server_$(date +%Y%m%d-%H%M%S)
OUT=scaleout/sflow_output/$RUN
ACCT=<your-slurm-account>

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/GB300-NVL72_GB300-288GB_aarch64x72/VLLM/Server/qwen3vl_config.yaml \
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

The full server-scenario run includes the 600-second performance phase plus
accuracy, warmup, and model spin-up; retain the full 4-hour walltime.

## Results

- Per-task logs: `$OUT/<jobid>-<workflow>-<ts>/{sflow.log,<task>/...}`
- Final QPS / TPS: `grep "QPS:" <jobid-dir>/sflow.log`
- Loadgen percentiles (TTFT, TPOT, end-to-end latency — what server scoring
  cares about): `results/qwen3_vl_235b_a22b_shopify_benchmark_server_gb300x72/report.txt`

## Per-recipe overrides

Adjust knobs at the top of `qwen3vl_config.yaml`; tune `target_qps`,
`warmup.n_requests` (defaults to `TOTAL_GPUS × 100 = 7200`), or
`runtime.min_duration_ms` in `endpoint.yaml`. Pass `--set KEY=VALUE` to
override at submit time without editing files.
