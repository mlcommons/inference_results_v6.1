# Qwen3-VL — VR-NVL72 Interactive (18 nodes × 4 GPUs)

Dynamo prefill/decode (P/D) disaggregation on 18 nodes × 4 GPUs (72 total): 48 TP=1 prefill
workers on 12 nodes and 24 TP=1 decode workers on 6 nodes, with the KV cache moved over NIXL,
behind one `dynamo.frontend` (`--router-mode=kv --load-aware`) on node 0. `endpoint.yaml` sends
Poisson load at `target_qps=1760` on `shopify_product_catalogue_8k::q3vl` for at least 600 s
after a 7200-request warmup, then runs the accuracy phase (`shopify_category_f1`) on the same
dataset. One Slurm job runs both phases.

## Configuration

| | |
|---|---|
| Image | `mlperf-inf-mm-q3vl-nv:arm64_cuda134_q3vlpublic_202610272018`, `sha256:768f0b37f404e14c177cf386b824ab5ce864dccca2a283bc17ef7e175172cfb6` (`CONTAINER_IMAGE`, pinned by digest). Built from public sources; to build it, see [the Rubin image README](../../../../../src/nv_mlpinf/benchmarks/q3vl/rubin/README.md). It includes NIXL 1.5.0 with its bundled UCX for the KV transfers, and the tuned CuTe-DSL FMHA exports, including the page-64 paged-GQA exports that `--block-size 64` prefill uses |
| Model | `nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.1-FP8-KV` |
| Topology (`qwen3vl_pd_disagg_config.yaml`) | `NUM_PREFILL_WORKERS` 48 and `NUM_DECODE_WORKERS` 24, both TP=1, `GPUS_PER_NODE` 4 (12 + 6 nodes); `KV_LEASE_DURATION` 180; `ENABLE_MEMCLK_PIN` "1"; `WORKFLOW_TIMEOUT` 230m |
| All workers | `--async-scheduling`, `--max-model-len 32768`, `--block-size 64`, `max_new_tokens` 150, `--no-enable-prefix-caching`, FP8 ViT attention (`--mm-encoder-attn-backend FLASHINFER`, `--mm-encoder-attn-dtype=fp8`, `_shared/vit_fp8_scales_all1.json`), `--mm-processor-cache-gb 0`, `--scheduling-policy priority`, `--enable-flashinfer-autotune` |
| Prefill workers | `--max-num-batched-tokens 17920`, `--max-num-seqs 1024`, `--moe-backend flashinfer_cutedsl`; `--compilation-config` with the compiled ViT encoder, encoder CUDA graphs and `max_cudagraph_capture_size` 17920 |
| Decode workers | `--max-num-batched-tokens 2048`, `--max-num-seqs 256`, `--enable-mm-embeds`, `--skip-mm-profiling`; `max_cudagraph_capture_size` 2048. No `--moe-backend`, so vLLM selects `FLASHINFER_TRTLLM` |
| Env (`worker_env.yaml`) | `VLLM_RUBIN_CUTEDSL_{VIT,VIT_PERSISTENT,PREFILL}=1` and `VLLM_RUBIN_CUTEDSL_PREFILL_POLICY=tuned` (Rubin CuTe-DSL ViT FMHA and paged GQA prefill); `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=""` (autotune every FlashInfer op); `DYN_LOG: info`; TCP request plane; `UCX_TLS=rc,cuda_copy,cuda_ipc`. No `FLASHINFER_DSL_FMHA_LOCAL_DIR` and no cache directory: the tuned FMHA exports come from the image, and all caches are container-local. The image's vLLM does not read the four `VLLM_SHM_TENSOR_ARENA*` keys, so each worker logs "Unknown vLLM environment variable" for them |
| `endpoint.yaml` | online, Poisson `target_qps` 1760, streaming; performance and accuracy datasets `shopify_product_catalogue_8k::q3vl` (`shopify_category_f1`); warmup 7200 requests; `min_duration_ms` 600000; seeds 42; `report_dir` `results/qwen3_vl_235b_a22b_shopify_8k_benchmark_interactive_disagg_vr_nvl72x72` |
| Template | `VR-NVL72_VR-288GB_aarch64x72/_shared/templates/vllm_dynamo_pd_disagg_serve_endpoints.yaml` (a VR-only copy of the shared template, composed with `VR-NVL72_VR-288GB_aarch64x72/_shared/slurm_env.yaml`): etcd, NATS and the frontend on node 0, the prefill and decode workers (NixlConnector with `kv_lease_duration`), `memclk_pin` / `memclk_reset`, and the `endpoint_benchmark` client |

## Launch

From `closed/NVIDIA/` (replace the `<...>` placeholders first). Install the `sflow` CLI and activate its venv first (see [scaleout/sflow/README.md](../../../../../scaleout/sflow/README.md), Section 1).

```bash
source /path/to/sflow/venv/.venv/bin/activate

SROOT=configs/qwen3_vl_235b_a22b
RUN=q3vl_pd_disagg_p48tp1_d24tp1_$(date +%Y%m%d_%H%M%S)
OUT=sflow_output/$RUN
ACCT=<your-slurm-account>
PARTITION=<your-slurm-partition>
SFLOW_RT=/path/on/a/shared/filesystem/sflow_1b9e0cdb
mkdir -p "$OUT"

sflow batch \
  -f $SROOT/VR-NVL72_VR-288GB_aarch64x72/_shared/slurm_env.yaml \
  -f $SROOT/VR-NVL72_VR-288GB_aarch64x72/_shared/templates/vllm_dynamo_pd_disagg_serve_endpoints.yaml \
  -f $SROOT/VR-NVL72_VR-288GB_aarch64x72/VLLM/Interactive/qwen3vl_pd_disagg_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HOME/.cache/huggingface \
  --set SLURM_ACCOUNT=$ACCT --set SLURM_PARTITION=$PARTITION \
  --partition $PARTITION --account $ACCT --time 04:00:00 --nodes 18 \
  -e '--segment=1' \
  --sflow-venv-path "$SFLOW_RT" --sflow-version 1b9e0cdb1389ae60c612f9cf5219648b1b9759b9 \
  --output-dir "$OUT" --sbatch-path "$OUT/sbatch.sh" --submit
```

- **sflow runtime.** The generated `sbatch.sh` installs the runtime sflow given by
  `--sflow-version` into a venv under `--sflow-venv-path` (a directory every node can read) at
  job start. Keep this commit, which writes each task's output straight to its log file: the
  workers log every request at INFO (`DYN_LOG: info`), and the expected result below was measured
  with this runtime.
- **Placement.** 18 nodes, `-e '--segment=1'`, no `--constraint`. On a Slurm cluster with
  `topology/block` and one 18-node block per NVL72 rack, `--segment=1` lets Slurm take the 18
  nodes from any blocks.
- **Image.** If pyxis cannot pull from the registry, import the image by digest with enroot
  once and pass `--set CONTAINER_IMAGE=<abs>.sqsh`:
  `enroot import -o <abs>.sqsh 'docker://gitlab-master.nvidia.com#mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv@sha256:768f0b37f404e14c177cf386b824ab5ce864dccca2a283bc17ef7e175172cfb6'`.
- **Model and HF token.** `HF_CACHE_HOST_DIR` is bind-mounted at `/root/.cache/huggingface` in
  every container; use a directory that all 18 nodes share. If the model repo is gated,
  `export HF_TOKEN=<token>` in the shell that runs `sflow batch`: sbatch passes that environment
  to the job (its default `--export=ALL`) and every task inherits it, so the token is not written
  into `sbatch.sh`, `sbatch.yaml` or the sflow logs. The VR templates have no `HF_TOKEN`
  variable, so `--set HF_TOKEN=...` is rejected.
- **Caches.** The FlashInfer autotune, FlashInfer JIT and torch.compile caches live inside the
  backend container. To start with cold caches, as the run in
  [Expected result](#expected-result) did, pass a new `--set BACKEND_CONTAINER_NAME=<name>`
  for each job.
- **Memory clock.** With `ENABLE_MEMCLK_PIN=1`, the template runs `sudo -n nvidia-smi -lmc` at
  4752 MHz on all 18 nodes before the workers start, waits until every GPU reads 4752 MHz, and
  runs `nvidia-smi -rmc` only after a successful benchmark, so the cluster must reset clocks
  after a failed job (for example in a Slurm epilog). Check for 72 `[memclk pinned] … 4752`
  lines from 18 hosts. They show that each GPU reads 4752 MHz, not a verified lock, because
  the step discards the `-lmc` output.
- **Time.** `--time 04:00:00`, with `WORKFLOW_TIMEOUT` 230m inside it.

## Expected result

One run of this folder's settings on the pinned image (n = 1, 2026-10-27): 18 nodes (12
prefill + 6 decode), `-e '--segment=1'`, `ENABLE_MEMCLK_PIN=1`, runtime sflow `1b9e0cdb`, cold
caches, performance and accuracy in one job.

| | Measured | MLPerf v6.1 gate | Verdict |
|---|---|---|---|
| Achieved QPS (target 1760) | 1755.84 | | |
| Samples issued / completed / failed | 1168000 / 1168000 / 0 | every issued sample completed, 0 failed | PASS |
| Performance phase | 665.35 s | ≥ 600 s | PASS |
| ES-p99 end-to-end latency | 1.404 s (1404241103 ns, 93.6 % of the limit) | < 1.5 s (1,500,000,000 ns) | **PASS** |
| TTFT / TPOT ES-p99 | 805.5 / 14.71 ms | no limit (information only) | |
| `shopify_category_f1` | 0.787092, 8000/8000 scored | ≥ 0.779922 (0.7878 × 0.99) | **PASS** |

The latency limit is `latency-constraint["qwen3-vl-235b-a22b"]["Interactive"]` in
`mlcommons/inference` `tools/submission/submission_checker/constants.py`; equality counts as
FAIL. The run also had 48 + 24 workers registered, MoE backend `FLASHINFER_CUTEDSL` on 48/48
prefill and `FLASHINFER_TRTLLM` on 24/24 decode workers, and 0 failed or expired KV
transfers. Engine init took 963-997 s (prefill) and 680-698 s (decode); the job took 32 min.

## Results

- Reports: `report_dir` under `WORK_DIR`, i.e.
  `results/qwen3_vl_235b_a22b_shopify_8k_benchmark_interactive_disagg_vr_nvl72x72/`, with
  `report.txt`, `performance/result_summary.json`, `accuracy/accuracy_results.json` and
  `events.jsonl`.
- `performance/result_summary.json` (latencies in ns): `.qps`, `.n_samples_issued`,
  `.n_samples_completed`, `.n_samples_failed`, `.complete`, `.duration_ns` (performance
  phase), `.latency.early_stopping_percentiles["99.0"]` (the latency gate),
  `.ttft.early_stopping_percentiles["99.0"]` and `.tpot.early_stopping_percentiles["99.0"]`.
- `accuracy/accuracy_results.json`: `.accuracy_scores[0].score` (`shopify_category_f1`) and
  `.accuracy_scores[0].response_counts.scored`.
- Per-task logs: `$OUT/<jobid>-<workflow>-<ts>/{sflow.log,<task>/...}`
- Final QPS / TPS: `grep -E "INFO - (QPS|TPS): " <jobid-dir>/endpoint_benchmark/endpoint_benchmark.log`
