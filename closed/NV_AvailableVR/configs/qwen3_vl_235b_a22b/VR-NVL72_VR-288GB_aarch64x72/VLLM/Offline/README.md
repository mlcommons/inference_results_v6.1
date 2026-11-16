# Qwen3-VL — VR-NVL72 Offline (18 nodes × 4 GPUs)

72 aggregated TP=1 `dynamo.vllm` workers, one per GPU on 18 nodes × 4 GPUs, behind one
`dynamo.frontend` (`--router-mode=kv --load-aware`) on node 0. `endpoint.yaml` issues
1738404 samples (36 × 48289) of `shopify_product_catalogue::q3vl` at `max_throughput` after a
28800-request warmup, then runs the 48289-sample accuracy phase on the same dataset. One run
does the warmup, the performance phase and the accuracy phase.

## Configuration

| | |
|---|---|
| Image | `mlperf-inf-mm-q3vl-nv:arm64_cuda134_q3vlpublic_202610272018`, pinned by digest `sha256:768f0b37f404e14c177cf386b824ab5ce864dccca2a283bc17ef7e175172cfb6` (`CONTAINER_IMAGE` in `qwen3vl_config.yaml`). Built from public sources; to build it, see [the Rubin image README](../../../../../src/nv_mlpinf/benchmarks/q3vl/rubin/README.md) |
| Model | `nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.1-FP8-KV` |
| Topology | `TOTAL_GPUS` 72, `GPUS_PER_NODE` 4, `GPUS_PER_WORKER` 1 (TP = PP = DP = 1), `WORKFLOW_TIMEOUT` 230m. `ENABLE_MEMCLK_PIN: "1"`: before the benchmark the workflow tries to lock each GPU's memory clock at 4752 MHz (`sudo -n nvidia-smi -lmc`) and aborts if the clocks do not read 4752 MHz; the `[memclk pinned]` log line shows the reading, not a verified lock |
| vLLM flags | `--max-num-seqs 512`, `--max-num-batched-tokens 26624`, `--max-model-len 32768`, `--async-scheduling`, `--moe-backend flashinfer_cutedsl`, `--scheduling-policy priority`, `--enable-flashinfer-autotune`, `--mm-encoder-attn-backend FLASHINFER` with `--mm-encoder-attn-dtype=fp8`, `--no-enable-prefix-caching` |
| `--compilation-config` | compiled and CUDA-graphed ViT encoder (token budgets 1024-16384), norm/act/attn quant fusion, inductor graph partition with `max_autotune`. `cudagraph_capture_sizes` has 399 sizes up to `max_cudagraph_capture_size` 26624: 1, 2, 4, every 8 to 256, every 16 to 512, every 64 to 21504, every 256 to 26624. |
| `worker_env.yaml` | `VLLM_RUBIN_CUTEDSL_{VIT,VIT_PERSISTENT,PREFILL}=1` and `VLLM_RUBIN_CUTEDSL_PREFILL_POLICY=tuned` (Rubin CuTe-DSL ViT FMHA with the persistent kernel, paged GQA prefill with the tuned policy; the tuned exports are baked in the image, so do not set `FLASHINFER_DSL_FMHA_LOCAL_DIR`), `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=""` (autotune every FlashInfer op), `VLLM_USE_FLASHINFER_SAMPLER=1`, `DYN_LOG: error`. The persistent ViT exports are built for 212 SMs per GPU |
| `endpoint.yaml` | `max_throughput`, `n_samples_to_issue` 1738404, warmup 28800 (`TOTAL_GPUS` × 400, drained), `max_new_tokens` 150, performance + accuracy datasets (`shopify_category_f1` against `ground_truth_category`), `submission_ref.ruleset: mlperf-inference-v6.1`, `report_dir: results/qwen3_vl_235b_a22b_shopify_benchmark_offline_vr_nvl72x72/` |

## Launch

From `closed/NVIDIA/`. Install the `sflow` CLI and activate its venv first (see [scaleout/sflow/README.md](../../../../../scaleout/sflow/README.md), Section 1).

```bash
SROOT=configs/qwen3_vl_235b_a22b
RUN=q3vl_x72_offline_$(date +%Y%m%d-%H%M%S)
OUT=scaleout/sflow_output/$RUN
ACCT=<your-slurm-account>
PARTITION=<your-slurm-partition>
SFLOW_RT=/path/on/a/shared/filesystem/sflow_6efbf59c
mkdir -p "$OUT"

sflow batch \
  -f $SROOT/VR-NVL72_VR-288GB_aarch64x72/_shared/slurm_env.yaml \
  -f $SROOT/VR-NVL72_VR-288GB_aarch64x72/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/VR-NVL72_VR-288GB_aarch64x72/VLLM/Offline/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HOME/.cache/huggingface \
  --set SLURM_ACCOUNT=$ACCT --set SLURM_PARTITION=$PARTITION \
  --partition $PARTITION --account $ACCT --time 04:00:00 --nodes 18 \
  -e '--segment=1' \
  --sflow-venv-path "$SFLOW_RT" --sflow-version 6efbf59c4a835473b08362eb623358e3ace31d6e \
  --output-dir $OUT --sbatch-path $OUT/sbatch.sh --submit
```

- Submit with `-e '--segment=1'` and no `--constraint`; the expected result below was measured
  with this placement.
- If pyxis cannot pull from the registry, import the image by digest with enroot once and pass
  `--set CONTAINER_IMAGE=<abs>.sqsh`:
  `enroot import -o <abs>.sqsh 'docker://gitlab-master.nvidia.com#mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv@sha256:768f0b37f404e14c177cf386b824ab5ce864dccca2a283bc17ef7e175172cfb6'`.
- The generated `sbatch.sh` installs the runtime sflow given by `--sflow-version` into a venv
  under `--sflow-venv-path` (a directory every node can read) at job start; the expected result
  below was measured with this commit.
- The workflow runs a head-node `prefetch_model` task before `vllm_worker` starts. It
  fills `HF_CACHE_HOST_DIR` if set, otherwise `/work/.cache/huggingface` under `WORK_DIR`; either
  must be on storage that all 18 nodes share. If the model repo is gated, `export HF_TOKEN=<token>`
  in the shell that runs `sflow batch`: sbatch passes that environment to the job (its default
  `--export=ALL`) and every task inherits it, so the token is not written into `sbatch.sh`,
  `sbatch.yaml` or the sflow logs. The VR templates have no `HF_TOKEN` variable, so
  `--set HF_TOKEN=...` is rejected.
- The FlashInfer autotune, FlashInfer JIT and torch.compile caches live inside the backend
  container. To start cold, as the expected result did, pass a new
  `--set BACKEND_CONTAINER_NAME=<name>` for each job.
- Keep the 4-hour walltime: engine start-up took 916-957 s per worker, and the whole measured
  run 32:01.

## Expected result

One run of this folder's settings on the pinned image (2026-10-27, 18 nodes, `--segment=1`, no
node constraint, runtime sflow `6efbf59c`, cold caches; n = 1) passes both MLPerf v6.1 gates:

| | Result | Gate |
|---|---|---|
| Samples issued / completed / failed | 1738404 / 1738404 / 0 | |
| Performance phase | 673.65 s | ≥ 600 s: PASS |
| **QPS** | **2580.58** | |
| TPS | 110807.05 | |
| **`shopify_category_f1`** | **0.786985** | ≥ 0.7903 × 0.99 = 0.782397: PASS |
| Accuracy samples scored | 48289 / 48289 | all 48289: PASS |

The gates come from `mlcommons/inference` `tools/submission/submission_checker/constants.py`
(master `3fbc3299`); Offline has no latency limit. The endpoints client has no Qwen3-VL
ruleset, so it reports the scores without a verdict: compare them with the gates yourself.
All 72 workers registered with MoE backend `FLASHINFER_CUTEDSL`.

## Results

- Per-task logs: `$OUT/<jobid>-<workflow>-<ts>/{sflow.log,<task>/...}`
- Final QPS / TPS: `grep -E "INFO - (QPS|TPS): " <jobid-dir>/endpoint_benchmark/endpoint_benchmark.log`
- `report_dir` is relative to `WORK_DIR`, so the outputs land in
  `closed/NVIDIA/results/qwen3_vl_235b_a22b_shopify_benchmark_offline_vr_nvl72x72/`:
  - `report.txt`: the client's summary;
  - `performance/result_summary.json`: `.qps`, `.tps`, `.n_samples_issued`,
    `.n_samples_completed`, `.n_samples_failed`, `.duration_ns`;
  - `accuracy/accuracy_results.json`: `.accuracy_scores[0].score` (`shopify_category_f1`) and
    `.accuracy_scores[0].response_counts.scored` (must be 48289).
