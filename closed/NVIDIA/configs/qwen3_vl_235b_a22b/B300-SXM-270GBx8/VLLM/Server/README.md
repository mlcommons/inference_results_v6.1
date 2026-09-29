# Qwen3-VL — B300x8 Server (perf)

Single-node × 8 GPUs (8× B300 SXM6, x86/amd64) Server-scenario perf run, via the
sflow + dynamo + inference-endpoint flow. Same flow as the GB300-NVL4 Server
recipe, ported to B300x8 with `TOTAL_GPUS=8`, `GPUS_PER_NODE=8`.

See the Offline README in the sibling folder for image / HF-cache specifics —
identical here.

## target_qps

`endpoint.yaml: load_pattern.target_qps` is set from the measured Offline
throughput ceiling, then stepped while watching e2e p99 against the MLPerf q3vl
Server latency bound. Run Offline first; adjust this knob per cluster.

## Launch

From `closed/NVIDIA/`:

```bash
SROOT=configs/qwen3_vl_235b_a22b
RUN=q3vl_b300x8_server
OUT=sflow_output/$RUN
ACCT=<your-slurm-account>
mkdir -p $OUT

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/B300-SXM-270GBx8/VLLM/Server/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HOME/.cache/huggingface \
  --set SLURM_ACCOUNT=$ACCT \
  --partition <partition> --account $ACCT --time 04:00:00 --nodes 1 \
  --output-dir $OUT --sbatch-path $OUT/sbatch.sh --submit
```

## Results

- Final QPS + percentiles: `results/qwen3_vl_235b_a22b_shopify_benchmark_online_b300x8/report.txt`
  (TTFT / TPOT / e2e avg/p50/p99/p99.9).
- Warmup count is `warmup.n_requests = 800` (8 GPUs × 100/instance).
