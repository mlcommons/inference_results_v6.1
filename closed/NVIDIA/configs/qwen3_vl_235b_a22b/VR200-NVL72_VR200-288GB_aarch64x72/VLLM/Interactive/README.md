# Qwen3-VL VR200x72 P/D Disagg Recipe

- 48 prefill workers, TP1
- 24 decode workers, TP1
- 18 VR200 nodes, 72 GPUs total
- V6.1 FP8-KV checkpoint
- Dynamo frontend in KV router mode with load-aware routing
- 7200 warmup requests
- Default QPS 1310 from the endpoint config

## Launch

From `closed/NVIDIA/`:

```bash
source /path/to/sflow/venv/.venv/bin/activate

SROOT=configs/qwen3_vl_235b_a22b
RUN=q3vl_pd_disagg_p48tp1_d24tp1_$(date +%Y%m%d_%H%M%S)
OUT=sflow_output/$RUN
ACCT=coreai_mlperf_inference
PARTITION=<your-slurm-partition>
mkdir -p "$OUT"

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_pd_disagg_serve_endpoints.yaml \
  -f $SROOT/VR200-NVL72_VR200-288GB_aarch64x72/VLLM/Interactive/qwen3vl_pd_disagg_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=<HF_CACHE_HOST_DIR> \
  --set SLURM_ACCOUNT=$ACCT \
  --partition $PARTITION --account $ACCT --time 04:00:00 --nodes 18 \
  --output-dir "$OUT" --sbatch-path "$OUT/sbatch.sh" --submit
```


The default endpoint config targets QPS 1310 and writes reports to:

```text
results/qwen3_vl_235b_a22b_shopify_8k_benchmark_interactive_disagg_vr200x72
```

## Important Settings

Note that the `max-num-batched-tokens` and `max_cudagraph_capture_size` are different
between the prefill workers and the decode workers.

```text
--max-num-batched-tokens 17920
max_cudagraph_capture_size=17920
```

Decode overrides both values to lower value due to decode batch size should be only around `--max-num-seqs`:

```text
--max-num-batched-tokens 2048
max_cudagraph_capture_size=2048
```

Keep decode `--max-num-batched-tokens` and `max_cudagraph_capture_size`
matched.

```text
--enable-mm-embeds 
```
This is the dynamo-vllm setting that allows decoding worker to read embeddings from prefill worker, which does not avoid any computation since it
is a decoding worker only config.

The template computes node usage from worker counts and TP:

```text
PREFILL_NUM_NODES = ceil(NUM_PREFILL_WORKERS / (4 / PREFILL_TENSOR_PARALLEL_SIZE))
DECODE_NUM_NODES = ceil(NUM_DECODE_WORKERS / (4 / DECODE_TENSOR_PARALLEL_SIZE))
NUM_NODES = PREFILL_NUM_NODES + DECODE_NUM_NODES
```

For the default `P48xTP1 + D24xTP1` topology this is `12 + 6 = 18` nodes.

## Files

- `qwen3vl_pd_disagg_config.yaml`: topology, model path, and vLLM flags.
- `worker_env.yaml`: worker environment shared by prefill/decode roles.
- `endpoint.yaml`: inference-endpoint benchmark config.
- `_shared/templates/vllm_dynamo_pd_disagg_serve_endpoints.yaml`: Dynamo
  frontend, workers, endpoint benchmark, and runtime Qwen placeholder patch.
