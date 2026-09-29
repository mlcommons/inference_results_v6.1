# MLPerf Inference — gpt-oss-120b Offline (8×MI300X, fp8)

**Submitter:** MangoBoost · **System:** 8xMI300X_2xEPYC_9534 (gfx942 / CDNA3) · **Division:** closed

Full step-by-step reproduction: **`~/docs/REPRO_GPTOSS_OFFLINE_MI300X_FP8.md`**
(fp8 quantization recipe: `~/docs/RECIPE_GPTOSS_120B_FP8_QUANTIZATION.md`).
Building & validating this v6.1 submission (seeds → clean loadgen → truncate →
checker, passes with NO `--submission-exceptions`):
**`~/docs/REPRO_GPTOSS_V61_SUBMISSION_MI300X_FP8.md`**.

> **Not comparable to the reference fp4 submission.** This uses a **self-quantized
> w-fp8-a-fp8** checkpoint (not the official MXFP4 model) on **stock vLLM 0.24.0**,
> because the MXFP4 MoE kernels are CDNA4-only and do not run on gfx942 (MI300X).
> Each result dir therefore carries a model-MD5 `INVALID` marker (fp8 hash ≠ fp4
> reference hash) — expected/non-fatal; the verdict is in `mlperf_log_summary.txt`.

> **⚠️ Note on config names.** The Hydra configs are named `offline_mi355x_fp8` /
> `server_mi355x_fp8`. The **"mi355x" is a historical artifact** — these are the
> **fp8** configs (derived from AMD's mi355x config, adapted for gfx942 + stock
> vLLM). They are what we run **on MI300X**; the name does not imply MI355X
> hardware. The config filename is unchanged so the commands below reproduce
> byte-for-byte against `src/gpt-oss-120b/offline_mi355x_fp8.yaml`.

## 1. Model & datasets

- **Model:** `openai/gpt-oss-120b` self-quantized to **fp8** with AMD Quark 0.12
  (MoE experts fp8_e4m3 per-tensor static; attention/router/lm_head bf16).
  Path: `/models/models/gpt-oss-120b-w-fp8-a-fp8` (24 shards).
- **Datasets:** `acc_eval_ref.parquet` (accuracy, 4395), `perf_eval_ref.parquet`
  (perf, 6396), `acc_eval_compliance_gpqa.parquet` (TEST07, 990).

## 2. Container — stock vLLM 0.24.0 (no custom image build)

```bash
docker run -d --name gptoss-mlperf-fp8 --ipc=host --network=host --privileged \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /models/models/gpt-oss-120b-w-fp8-a-fp8:/model/gpt-oss-120b/fp4_quantized:ro \
  -v <DATA>/perf_eval_ref.parquet:/data/gpt-oss-120b/perf_eval_ref.parquet:ro \
  -v <DATA>/acc_eval_ref.parquet:/data/gpt-oss-120b/acc_eval_ref.parquet:ro \
  -v <DATA>/acc_eval_compliance_gpqa.parquet:/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet:ro \
  -v <REPO>/src:/lab-mlperf-inference/code \
  -v <HARNESS_OVERLAY>/harness_llm:/lab-mlperf-inference/code/harness_llm \
  -v <RESULTS>:/lab-mlperf-inference/results \
  --entrypoint sleep vllm/vllm-openai-rocm:v0.24.0 infinity
```
Then install harness deps once (apt numactl/…, pip absl-py/datasets/…, build MLCommons
loadgen rev `52235a45` + rocm_bandwidth_test rev `6bd5c1a2`) — see the REPRO doc Part C.
The harness overlay applies the upstream-vLLM compat patch (TokenInputs→TokensPrompt,
ray-optional, AsyncEngineArgs swap_space filter).

## 3. Runtime tunables (once per boot)

```bash
bash setup/runtime_tunables.sh    # governor=performance, THP on, numa_balancing off
# on this node cpupower is partial (kernel 5.15) → disable CPU C-state 2 manually
```

## 4. Run (inside the container; `PYTHONPATH=/lab-mlperf-inference/code`)

`user.conf` used: `gpt-oss-120b.Offline.target_qps = 70`,
`gpt-oss-120b.Offline.min_duration = 2400000`. Permissible tune applied:
`gpu_memory_utilization=0.95` (MLPerf allows batch/KV/memory tuning).

```bash
# Performance
python3 /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/gpt-oss-120b/ --config-name offline_mi355x_fp8 \
   test_mode=performance harness_config.device_count=8 \
   harness_config.user_conf_path=<user.conf> \
   harness_config.output_log_dir=.../Offline/performance/run_1 \
   vllm_engine_config.gpu_memory_utilization=0.95

# Accuracy (then score)
python3 /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/gpt-oss-120b/ --config-name offline_mi355x_fp8 \
   test_mode=accuracy harness_config.device_count=8 \
   harness_config.user_conf_path=<user.conf> \
   harness_config.output_log_dir=.../Offline/accuracy \
   vllm_engine_config.gpu_memory_utilization=0.95 vllm_sampling_config.max_tokens=32768
ARCH=gfx942 bash /lab-mlperf-inference/code/scripts/check_gptoss_accuracy_scores.sh \
   .../Offline/accuracy/mlperf_log_accuracy.json
```

**Compliance** TEST07 (GPQA accuracy audit, 990) and TEST09 (output-length audit,
6396): copy the test's `compliance/TEST0{7,9}/gpt-oss-120b/audit.config` into the run
CWD, run the same config in `test_mode=performance`, then verify with the MLCommons
`compliance/TEST0{7,9}/run_verification.py`. Full commands in the REPRO doc.

## 5. Result (this submission)

Offline performance (min_duration 2400 s, util 0.95): **VALID, 30,739 tok/s**.
Accuracy: **84.02%** (aime25 83.75 / gpqa_diamond 75.66 / livecodebench_v6 85.59).
TEST07 **PASS**, TEST09 **PASS**. See `results/.../Offline/{performance,accuracy,TEST07,TEST09}`.
