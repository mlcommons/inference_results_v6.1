# MLPerf Inference — gpt-oss-120b Server (8×MI300X, fp8)

**Submitter:** MangoBoost · **System:** 8xMI300X_2xEPYC_9534 (gfx942 / CDNA3) · **Division:** closed

Full step-by-step reproduction: **`~/docs/REPRO_GPTOSS_SERVER_COMPLIANCE_MI300X_FP8.md`**
(shared setup: `~/docs/REPRO_GPTOSS_OFFLINE_MI300X_FP8.md`).
Building & validating this v6.1 submission (seeds → clean loadgen → truncate →
checker, passes with NO `--submission-exceptions`):
**`~/docs/REPRO_GPTOSS_V61_SUBMISSION_MI300X_FP8.md`**.

> **Not comparable to the reference fp4 submission.** Self-quantized **w-fp8-a-fp8**
> checkpoint on **stock vLLM 0.24.0** (MXFP4 MoE kernels are CDNA4-only, so the fp4
> model cannot run on gfx942). Result dirs carry the model-MD5 `INVALID` marker
> (fp8 ≠ fp4 reference hash) — expected/non-fatal; verdict is in `mlperf_log_summary.txt`.

> **⚠️ Note on config names.** `server_mi355x_fp8` — the **"mi355x" is historical**
> (derived from AMD's mi355x config); this is the **fp8** Server config run **on
> MI300X**, not MI355X hardware. The filename is unchanged so the commands reproduce
> against `src/gpt-oss-120b/server_mi355x_fp8.yaml`.

## 1–3. Model, datasets, container, tunables

Identical to the Offline README (self-quantized fp8 model at
`/models/models/gpt-oss-120b-w-fp8-a-fp8`, stock `vllm/vllm-openai-rocm:v0.24.0`
container + harness deps, `bash setup/runtime_tunables.sh`). Container additionally
mounts `acc_eval_compliance_gpqa.parquet` for TEST07.

## 4. Run (inside the container; `PYTHONPATH=/lab-mlperf-inference/code`)

Config `server_mi355x_fp8` = async engine + **cudagraphs on** (`enforce_eager: False`),
quark, AITER off, `gpu_memory_utilization=0.90`. Server latency bounds: **p99 TTFT ≤
3000 ms, p99 TPOT ≤ 80 ms.** `user.conf`: `Server.target_qps = 20`,
`Server.min_duration = 1800000`.

> **Why 1800 s (not the 600 s floor):** Server validity also requires enough queries
> for the p99 **early-stopping** confidence test. At 20 QPS a 600 s run is a few
> queries short (INVALID despite TPOT < 80); the full 1800 s run satisfies it.

```bash
# Performance @ 20 QPS
python3 /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/gpt-oss-120b/ --config-name server_mi355x_fp8 \
   test_mode=performance harness_config.device_count=8 \
   harness_config.user_conf_path=<user.conf> \
   harness_config.output_log_dir=.../Server/performance/run_1

# Accuracy (then score)
python3 /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/gpt-oss-120b/ --config-name server_mi355x_fp8 \
   test_mode=accuracy harness_config.device_count=8 \
   harness_config.user_conf_path=<user.conf> \
   harness_config.output_log_dir=.../Server/accuracy vllm_sampling_config.max_tokens=32768
ARCH=gfx942 bash /lab-mlperf-inference/code/scripts/check_gptoss_accuracy_scores.sh \
   .../Server/accuracy/mlperf_log_accuracy.json
```

**Compliance** TEST07/TEST09 as in the Offline README (audit.config in CWD + MLCommons
`run_verification.py`), run with `server_mi355x_fp8`.

## 5. Result (this submission)

Server @20 QPS (min_duration 1800 s): **VALID, 25,826 tok/s** — p99 TTFT 690 ms,
p99 TPOT 79.65 ms, early-stopping satisfied. Accuracy score matches Offline (**84.02%**,
scenario-independent). TEST07/TEST09 in `results/.../Server/{TEST07,TEST09}`.
