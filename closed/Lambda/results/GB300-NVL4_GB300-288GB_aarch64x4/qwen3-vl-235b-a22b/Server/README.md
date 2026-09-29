# Qwen3-VL-235B-A22B Server (Lambda 4xGB300)

Served with vLLM+NVIDIA Dynamo (4 data-parallel replicas, TP=1, one GB300 per replica),
benchmarked via the NVIDIA/MLCommons inference-endpoints client.
Endpoint result format. Seeds: dataloader=2747215439041700203, scheduler=16159082839903944936. min_duration=600s.

Result: 64.62 QPS (2773 TPS) at target_qps=65 (Poisson arrivals) over 749.83s,
48289/48289 samples completed, 0 failed.
Latency: end-to-end p99 = 11.86s, TTFT p99 = 2.15s, TPOT p99 = 225ms.
Accuracy: shopify_category_f1 = 0.7871 (threshold 0.782397 = 99% of 0.7903). PASS.

## Artifacts

- `config.yaml` - resolved endpoints-client configuration for this run (also copied under
  `accuracy/` and `performance/run_1/`).
- `performance/result_summary.json` - performance metrics for the run.
- `accuracy/accuracy_results.json` - accuracy metrics for the run.
- `accuracy/events.jsonl` - per-sample event log including every model response
  (`sample.complete` / `TextModelOutput`). Covers both the performance and the accuracy pass of
  this single "both"-mode run; `accuracy/sample_idx_map.json` maps each `sample_uuid` to its
  dataset index per pass.
- `performance/run_1/metrics_final_snapshot.json` - final client-side counter snapshot.
- `report.txt` - human-readable run report emitted by the endpoints client.
- `as_run/` - the exact launch configuration consumed by this job (see below).

## As-run configuration

The run was orchestrated with nv-sflow (pinned to `NVIDIA/nv-sflow@6efbf59`) on a single-node
Slurm 23.11.4 cluster using Pyxis 0.24.0 / Enroot 3.5.0. `as_run/sbatch.sh` is the generated
batch script (the `HF_TOKEN` value is redacted), and `as_run/{slurm_env,vllm_dynamo_serve_endpoints,qwen3vl_config,worker_env,endpoint}.yaml`
are the files it consumed.

These differ from `configs/qwen3_vl_235b_a22b/GB300-NVL72_GB300-288GB_aarch64x4/VLLM/Server/`
in this package, which is an earlier drop of the NVIDIA partner code. The functional deltas,
all required to run on this single self-hosted GB300 NVL4 node, are:

1. `worker_env.yaml`: `HOME=/tmp/home` plus `XDG_CACHE_HOME`/`TRITON_CACHE_DIR`/
   `TORCHINDUCTOR_CACHE_DIR`/`VLLM_CACHE_ROOT` redirected under `/tmp/home/.cache`, because the
   container rootfs is read-only and flashinfer/triton/torch derive their cache paths from `HOME`.
2. `vllm_dynamo_serve_endpoints.yaml`: each vLLM worker pins itself with
   `CUDA_VISIBLE_DEVICES=$(( SFLOW_REPLICA_INDEX % GPUS_PER_NODE ))`. All four workers attach to a
   single shared per-node Pyxis container that exposes all 4 GPUs, so per-step GRES isolation does
   not separate them and without the pin every replica lands on GPU 0.
3. `slurm_env.yaml`: `container_writable: true`, and container images overridden to locally
   imported Enroot squashfs images (`CONTAINER_IMAGE`, `ENDPOINT_CONTAINER_IMAGE`) since the
   internal `gitlab-master.nvidia.com` registry is not reachable from this node.
4. `sbatch.sh`: `#SBATCH --gres=gpu:4` added, because this version of `sflow batch` does not emit
   a `--gres` directive from `--gpus-per-node`.
5. `qwen3vl_config.yaml`/`worker_env.yaml` otherwise match the NVIDIA GB300 4-GPU recipe
   (NVFP4 weights, FP8 KV cache, Poisson load pattern at target_qps=65).
