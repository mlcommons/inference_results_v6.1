# Qwen3-VL — GB200-NVL4 Interactive (perf)

Single-node × 4 GPUs interactive-scenario perf run. Each `dynamo.vllm` worker
spans all 4 GPUs (**TP=4**, so 1 worker on the node) and the client drives
the smaller **8k** shopify dataset (`shopify_product_catalogue_8k::q3vl`).
Poisson load at `target_qps=4`.

## Launch

From `closed/NVIDIA/`:

```bash
SROOT=configs/qwen3_vl_235b_a22b
RUN=q3vl_nvl4_interactive_$(date +%Y%m%d-%H%M%S)
OUT=scaleout/sflow_output/$RUN
ACCT=<your-slurm-account>

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/GB200-NVL72_GB200-186GB_aarch64x4/VLLM/Interactive/qwen3vl_config.yaml \
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
- Loadgen percentiles (TTFT, TPOT, end-to-end latency — what interactive
  scoring cares about):
  `results/qwen3_vl_235b_a22b_shopify_8k_benchmark_interactive_gb200x4/report.txt`

## Per-recipe overrides

Adjust knobs at the top of `qwen3vl_config.yaml`; tune `target_qps`,
`warmup.n_requests`, or `runtime.min_duration_ms` in `endpoint.yaml`. Pass
`--set KEY=VALUE` to override at submit time without editing files.
