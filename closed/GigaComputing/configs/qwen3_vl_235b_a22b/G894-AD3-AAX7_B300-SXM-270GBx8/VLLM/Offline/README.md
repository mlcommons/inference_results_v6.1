# Qwen3-VL — B300x8 Offline (perf)

Single-node × 8 GPUs (8× B300 SXM6, x86/amd64) Offline-scenario perf run, via
the sflow + dynamo + inference-endpoint flow. Same flow as the GB300-NVL4
recipe, ported to B300x8 with `TOTAL_GPUS=8`, `GPUS_PER_NODE=8`.


## Launch

From `closed/NVIDIA/`:

```bash
SROOT=configs/qwen3_vl_235b_a22b
RUN=q3vl_b300x8_offline_$(date +%Y%m%d-%H%M%S)
OUT=scaleout/sflow_output/$RUN
ACCT=root

.venv/bin/sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/G894-AD3-AAX7_B300-SXM-270GBx8/VLLM/Offline/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$PWD/.cache/huggingface \
  --set SLURM_ACCOUNT=$ACCT \
  --partition b300 --account $ACCT --time 04:00:00 --nodes 1 \
  --output-dir $OUT --sbatch-path $OUT/sbatch.sh --submit
```

## Results

- Per-task logs: `$OUT/<jobid>-<workflow>-<ts>/{sflow.log,<task>/...}`
- Final QPS: `grep "QPS:" <jobid-dir>/sflow.log`
- Loadgen percentiles: `results/qwen3_vl_235b_a22b_shopify_benchmark_offline_b300x8/report.txt`

## Per-recipe overrides

Adjust knobs at the top of `qwen3vl_config.yaml` in this folder. The warmup
count is `warmup.n_requests = 3200` (8 GPUs × 400/instance; GB300-NVL4 uses
1600 = 4×400). Offline uses `load_pattern.type=max_throughput` (no target_qps)
— run it first to get the throughput ceiling, then set the Server `target_qps`
from it. Pass `--set KEY=VALUE` to override at submit time without editing files.
