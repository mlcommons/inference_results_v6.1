# Qwen3-VL — GB300-NVL72 Interactive (perf)

18-node × 4 GPUs (72 total) interactive-scenario perf run. Same recipe as
Server, but each `dynamo.vllm` worker spans all 4 GPUs of its node (**TP=4**,
so 18 workers across 18 nodes) and the client drives the smaller **8k** shopify
dataset (`shopify_product_catalogue_8k::q3vl`). Poisson load at `target_qps=132.5`.

## Launch

From `closed/NVIDIA/`:

```bash
SROOT=configs/qwen3_vl_235b_a22b
RUN=q3vl_x72_interactive_$(date +%Y%m%d-%H%M%S)
OUT=scaleout/sflow_output/$RUN
ACCT=<your-slurm-account>

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/GB300-NVL72_GB300-288GB_aarch64x72/VLLM/Interactive/qwen3vl_config.yaml \
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

## Launch inside an existing allocation (`sflow run`)

If you already hold the 18 nodes interactively (e.g. `salloc -N18 --partition
gb300 --account $ACCT --gpus-per-node 4 --time 04:00:00`), use `sflow run`
instead. It runs in the foreground and **reuses the current allocation** — its
`srun` steps inherit `$SLURM_JOB_ID` — rather than queuing a new job:

```bash
SROOT=configs/qwen3_vl_235b_a22b
sflow run \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/GB300-NVL72_GB300-288GB_aarch64x72/VLLM/Interactive/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HOME/.cache/huggingface \
  --set SLURM_ACCOUNT=$ACCT
```

No `--partition/--account/--nodes/--time` (those come from the allocation
you're in). Add `--tui` for the live DAG, or `--dry-run` to validate the
config composition without running. Output defaults to `sflow_output/`
(override with `--output-dir`).

## Results

- Per-task logs: `$OUT/<jobid>-<workflow>-<ts>/{sflow.log,<task>/...}`
- Final QPS / TPS: `grep "QPS:" <jobid-dir>/sflow.log`
- Loadgen percentiles (TTFT, TPOT, end-to-end latency — what interactive
  scoring cares about):
  `results/qwen3_vl_235b_a22b_shopify_8k_benchmark_interactive/report.txt`

## Per-recipe overrides

Adjust knobs at the top of `qwen3vl_config.yaml`; tune `target_qps`,
`warmup.n_requests` (defaults to `TOTAL_GPUS × 100 = 7200`), or
`runtime.min_duration_ms` in `endpoint.yaml`. Pass `--set KEY=VALUE` to
override at submit time without editing files.
