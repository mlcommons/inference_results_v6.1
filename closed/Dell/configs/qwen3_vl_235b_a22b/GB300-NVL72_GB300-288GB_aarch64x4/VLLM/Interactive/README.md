# Qwen3-VL — GB300-NVL4 Interactive (perf)

Single-node × 4 GPUs interactive-scenario perf run. Same recipe as Server, but
each `dynamo.vllm` worker spans all 4 GPUs (**TP=4**, so 1 worker on the node)
and the client drives the smaller **8k** shopify dataset
(`shopify_product_catalogue_8k::q3vl`). Poisson load at `target_qps=5`.

## Zero-copy shared-memory tensor arena

This TP=4 recipe automatically uses vLLM's zero-copy shared-memory tensor
arena for large multimodal tensors. Avoiding repeated engine-to-worker tensor
serialization significantly reduces p99 latency when operating close to the
Interactive latency constraint. The feature is currently enabled by default;
see the [general q3vl documentation](../../../../../src/nv_mlpinf/benchmarks/q3vl/vllm/README.md#zero-copy-shared-memory-tensor-arena)
for tuning parameters, memory-sizing guidance, safe fallback behavior, and the
manual disable switch.

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
  -f $SROOT/GB300-NVL72_GB300-288GB_aarch64x4/VLLM/Interactive/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HOME/.cache/huggingface \
  --set SLURM_ACCOUNT=$ACCT \
  --partition gb300 --account $ACCT --time 04:00:00 --nodes 1 \
  --output-dir $OUT --sbatch-path $OUT/sbatch.sh --submit
```

The shared workflow runs a head-node `prefetch_model` task before
`vllm_worker` starts. If `HF_CACHE_HOST_DIR` is set, it fills that mounted
cache; otherwise it falls back to `/work/.cache/huggingface` under `WORK_DIR`,
which all tasks share through the `/work` mount. Set `HF_TOKEN=...` via `--set`
if the model repo is gated.

## Launch inside an existing allocation (`sflow run`)

If you already hold the node interactively (e.g. `salloc -N1 --partition gb300
--account $ACCT --gpus-per-node 4 --time 04:00:00`), use `sflow run` instead.
It runs in the foreground and **reuses the current allocation** — its `srun`
steps inherit `$SLURM_JOB_ID` — rather than queuing a new job:

```bash
SROOT=configs/qwen3_vl_235b_a22b
sflow run \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/GB300-NVL72_GB300-288GB_aarch64x4/VLLM/Interactive/qwen3vl_config.yaml \
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
  `results/qwen3_vl_235b_a22b_shopify_8k_benchmark_interactive_gb300x4/report.txt`

## Per-recipe overrides

Adjust knobs at the top of `qwen3vl_config.yaml`; tune `target_qps`,
`warmup.n_requests`, or `runtime.min_duration_ms` in `endpoint.yaml`. Pass
`--set KEY=VALUE` to override at submit time without editing files. With TP=4
there is a single worker on the node. The configured `target_qps=5` is the
confirmed maximum below the end-to-end p99 latency gate; `5.5` is the first
validated latency failure.
