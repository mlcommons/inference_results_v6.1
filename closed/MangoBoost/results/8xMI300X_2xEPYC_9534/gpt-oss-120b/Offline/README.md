# Reproducing MangoBoost's gpt-oss-120b (fp8) MLPerf v6.1 Results on a Fresh 8×MI300X

Audience: an external engineer/evaluator with a **fresh 8×MI300X machine that has
nothing set up**. Every step is copy-paste. Follow top to bottom.

> **What is being reproduced.** A **closed-division datacenter** submission of
> `gpt-oss-120b`, run as a **self-quantized w-fp8-a-fp8** checkpoint on **stock
> vLLM 0.24.0** (`vllm/vllm-openai-rocm:v0.24.0`). The official reference model is
> **MXFP4**, whose MoE kernels are **CDNA4-only** and do **not** run on MI300X
> (gfx942 / CDNA3) — hence fp8. Results are therefore **not comparable** to the
> reference fp4 submission, and every result dir carries a model-MD5 `INVALID`
> marker (fp8 hash ≠ fp4 reference) that is **expected and non-fatal** — the
> verdict is in `mlperf_log_summary.txt`, not that marker.

### Target results (what a correct run should produce)

| Scenario | Metric | Target | Constraints |
|---|---|---|---|
| Offline | throughput | **~30,198 tok/s**, `Result is : VALID` | min_duration 600 s |
| Server  | throughput | **~24,863 tok/s** @ 19.5 QPS, `VALID` | p99 TTFT ≤ 3000 ms, p99 TPOT ≤ 80 ms, early-stopping |
| Offline | accuracy (exact_match) | **~83.7 %** | ≥ **82.30 %** (= 83.13 × 0.99) |
| Server  | accuracy (exact_match) | **~82.8 %** | ≥ 82.30 % |
| Both | Compliance TEST07 (GPQA) | PASS | ≥ 60.698 % |
| Both | Compliance TEST09 (output len) | PASS | 1278.20 tok ±10 % |

Accuracy is stochastic (`temperature=1`), so expect ±0.5 % run-to-run. Throughput
varies a few % with node/thermal state.

---

## 0. Hardware / OS baseline

Reference system for these numbers (yours can differ; MI300X 192 GB is the key part):

- 8 × **AMD Instinct MI300X 192 GB HBM3** (gfx942 / CDNA3), all-to-all Infinity Fabric
- 2 × AMD EPYC 9534 (64c each), **2.2 TB** DDR5-4800, 7 TB NVMe
- **Ubuntu 22.04.5**, **ROCm 7.2.1**, Docker with the ROCm runtime

Requirements: **≥ ~250 GB free disk** (model + datasets + image + logs), the 8
GPUs **idle** (no other tenant), and network egress to HuggingFace + MLCommons.

```bash
# sanity: GPUs visible & idle
rocm-smi --showproductname
rocm-smi --showuse | grep "GPU use" | head    # all should read 0% before you start
docker --version
```

---

## 1. Get the three inputs

Set paths once (edit the left-hand side to real locations on your box):

```bash
export BASE=/data/models/gpt-oss-120b-base               # original model download (1a)
export MODEL_FP8=/data/models/gpt-oss-120b-w-fp8-a-fp8   # fp8 checkpoint you produce (1a)
export QWORK=/data/quant-work                             # quantization scripts + logs (1a)
export DATA=/data/gpt-oss-120b                            # holds the 3 parquets (1b)
export SRC=/data/submission/closed/MangoBoost/src         # harness from the submission (1c)
export RESULTS=/data/gptoss-repro                          # your output dir
export SYSTEM=8xMI300X_2xEPYC_9534                         # submission's system descriptor; use the one you evaluate, or name it after your host
mkdir -p "$RESULTS" "$QWORK"
```
> `$MODEL_FP8`'s basename (`gpt-oss-120b-w-fp8-a-fp8`) is reused literally in 1a —
> keep them matching if you rename.

### 1a. Model — download `openai/gpt-oss-120b` and self-quantize to fp8

FP4 model from AMD ships as **MXFP4**, whose MoE kernels are CDNA4-only and won't
run on MI300X (gfx942). We self-quantize the **MoE experts → fp8** (attention,
router, `lm_head` stay bf16). Needs **~180 GB free disk** and the **8 GPUs idle**
(`rocm-smi --showuse` ~0% — a busy node can GPU-hang mid-calibration). ~30 min.

**Step 1 — download the base model.** Only the root safetensors are needed; skip
the 122 GB `metal/` + `original/` dirs.

```bash
pip install -U "huggingface_hub[cli]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
hf auth login          # gpt-oss is gated: accept the license on the HF model page first
hf download openai/gpt-oss-120b --revision b5c939d --local-dir "$BASE" \
  --exclude "metal/*" --exclude "original/*"     # repeat --exclude PER pattern (it is not a list)
ls "$BASE"/*.safetensors | wc -l                 # expect 15 ;  du -sh "$BASE" ≈ 61 GB
```
The base `config.json` lists `modules_to_not_convert` = `self_attn`, `mlp.router`,
`embed_tokens`, `lm_head` — exactly what we keep in bf16 below (don't invent your own).

**Step 2 — quantization container.** Any ROCm + PyTorch (torch ≥ 2.x) image works
as the base; the one below is verified. Quark is pip-installed into it.

```bash
docker run -d --name gptoss-quant \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=32g --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  -v "$BASE":/model_src:ro -v "$(dirname "$MODEL_FP8")":/out -v "$QWORK":/work \
  --entrypoint sleep rocm/mlperf-inference:gptoss_vllm_v0.14.0_amd_dev infinity
```

**Step 3 — install AMD Quark 0.12** (the exact pins matter):

```bash
docker exec gptoss-quant bash -lc '
set -e
pip install --no-cache-dir --no-deps amd-quark==0.12.post1   # 0.12 (0.10 breaks on torch>=2.9)
pip install --no-cache-dir scipy onnx onnxslim zstandard     # PTQ deps; zstandard reads .zst calib
git clone --depth 1 --branch release/0.12 https://github.com/amd/Quark.git /work/Quark_012'
```

**Step 4 — run the quantization** (detached, ~30 min: ~10 load+calibrate, ~19 export):

```bash
docker exec gptoss-quant bash -lc 'cat > /work/run_quant.sh <<"EOF"
#!/bin/bash
cd /work/Quark_012/examples/torch/language_modeling/llm_ptq/
python3 quantize_quark.py \
  --model_dir /model_src \
  --quant_scheme fp8 \
  --exclude_layers "*lm_head*" "*self_attn*" "*router*" \
  --num_calib_data 128 --dataset pileval --seq_len 512 \
  --multi_gpu --skip_evaluation \
  --model_export hf_format \
  --output_dir /out/gpt-oss-120b-w-fp8-a-fp8
EOF
rm -f /work/quant.log; nohup bash /work/run_quant.sh > /work/quant.log 2>&1 & echo launched'
```
Flags: `fp8` = per-tensor **static** weight-fp8 + activation-fp8 (the 0.12 name;
0.10's `w_fp8_a_fp8` is rejected). `--exclude_layers` keeps attention/router/lm_head
in bf16 (quantize **MoE experts only**). 128 pileval samples @ `seq_len 512` are
enough for static scales. `--multi_gpu` shards the bf16 model across the 8 GPUs (it
won't fit on one). `hf_format` writes vLLM-readable safetensors + `quantization_config`.

**Step 5 — wait and verify** (expect **24 shards, ~111 GB, experts NOT excluded**):

```bash
docker exec gptoss-quant bash -lc '
for i in $(seq 1 220); do
  [ -f /out/gpt-oss-120b-w-fp8-a-fp8/model.safetensors.index.json ] && ! pgrep -f quantize_quark.py >/dev/null && { echo DONE; break; }
  grep -qaiE "Traceback|GPU Hang|core dumped|error:" /work/quant.log && { echo FAILED; tail -5 /work/quant.log; break; }
  sleep 10
done
python3 - <<PY
import json, glob
d = "/out/gpt-oss-120b-w-fp8-a-fp8"
qc = json.load(open(f"{d}/config.json"))["quantization_config"]; g = qc.get("global_quant_config", {})
print("shards :", len(glob.glob(f"{d}/*.safetensors")), "(expect 24)")
print("weight :", g.get("weight",{}).get("dtype"), "| act:", g.get("input_tensors",{}).get("dtype"),
      "| static:", not g.get("input_tensors",{}).get("is_dynamic"), "(expect fp8_e4m3 / fp8_e4m3 / True)")
print("experts excluded (must be False):", any("experts" in e for e in qc.get("exclude", [])))
PY'
docker rm -f gptoss-quant
```
> `config.json` has `torch_dtype: null` — every consumer must pass **`dtype=bfloat16`**
> explicitly (the harness config already does; left unset, some loaders pick fp32 and OOM).
> The quantization is deterministic (fixed 128-sample calibration), so it reproduces an
> equivalent checkpoint that meets the accuracy target; the ±0.5 % run-to-run accuracy
> you may see comes from `temperature=1` **sampling at inference**, not the weights.

> **Sanity smoke (optional, recommended):** load `$MODEL_FP8` in
> `vllm/vllm-openai-rocm:v0.24.0` with `tensor_parallel_size=2, dtype=bfloat16,
> enforce_eager=True` and generate a couple of prompts. Coherent text with **no
> `PassManager::run failed`** confirms the fp8 conversion cleared the gfx942 MoE blocker.

### 1b. Datasets → MLCommons (Parquet)

Three files, into `$DATA/`: `perf_eval_ref.parquet` (6396, perf),
`acc_eval_ref.parquet` (4395, accuracy), `acc_eval_compliance_gpqa.parquet`
(990, TEST07).

```bash
pip install mlc-scripts
mlcr get-dataset-mlperf-inference-gpt-oss,_mlc,_r2-downloader --outdirname="$DATA" -j
# or browse/download directly: https://inference.mlcommons-storage.org/index.html
ls "$DATA"/*.parquet    # expect the 3 files above
```

### 1c. Harness source → from this submission

The full harness (Hydra configs, the MangoBoost `harness_llm` overlay, `main.py`,
`scripts/`) is bundled in the submission at `closed/MangoBoost/src/`. Point `$SRC`
at it (step 1). No separate patch is needed — the overlay is already inside.

---

## 2. Launch the container (stock vLLM 0.24.0)

```bash
docker run -d --name gptoss-mlperf-fp8 --ipc=host --network=host --privileged \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$MODEL_FP8":/model/gpt-oss-120b/fp4_quantized:ro \
  -v "$DATA/perf_eval_ref.parquet":/data/gpt-oss-120b/perf_eval_ref.parquet:ro \
  -v "$DATA/acc_eval_ref.parquet":/data/gpt-oss-120b/acc_eval_ref.parquet:ro \
  -v "$DATA/acc_eval_compliance_gpqa.parquet":/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet:ro \
  -v "$SRC":/lab-mlperf-inference/code \
  -v "$RESULTS":/lab-mlperf-inference/results \
  --entrypoint sleep vllm/vllm-openai-rocm:v0.24.0 infinity
```
> The mount target `/model/gpt-oss-120b/fp4_quantized` is a **historical name** —
> it holds the **fp8** model; the Hydra configs point `model:` there. Config files
> are named `*_mi355x_fp8` (derived from AMD's mi355x config) but run on **MI300X**.

---

## 3. Install deps + build loadgen (once per container, ~5 min)

```bash
docker exec gptoss-mlperf-fp8 bash -lc '
set -e
apt update && apt install -y numactl libnuma-dev sqlite3 libfmt-dev cmake build-essential
pip install --no-cache-dir absl-py==2.1.0 datasets==2.20.0 nltk==3.8.1 py-libnuma==1.2 \
    rouge_score==0.1.2 omegaconf==2.3.0 evaluate==0.4.3 matplotlib==3.10.3 optuna==4.1.0 hydra-core==1.3.2
# LoadGen from mlcommons/inference master HEAD: it already ships the v6.1 RNG seeds
# (qsl 2085463073848966840) in a COMMITTED tree — so no seed patch, no "uncommitted
# changes" flag, and it passes the submission_checker clean.
git clone https://github.com/mlcommons/inference.git /tmp/mlperf_inference \
 && cd /tmp/mlperf_inference/loadgen && git submodule update --init --recursive \
 && CFLAGS="-std=c++14 -O3" python -m pip install .
mkdir -p /lab-mlperf-inference/mlperf_inference && cp -a /tmp/mlperf_inference/. /lab-mlperf-inference/mlperf_inference/
# rocm_bandwidth_test (used by the NUMA helper)
git clone https://github.com/ROCm/rocm_bandwidth_test /tmp/rbt && cd /tmp/rbt \
 && git checkout 6bd5c1a208069be5fbe9e2da088bca4be7f9de22 && mkdir -p build && cd build \
 && cmake -DCMAKE_MODULE_PATH=/tmp/rbt/cmake_modules -DCMAKE_PREFIX_PATH=/opt/rocm/ .. \
 && make -j$(nproc) && make install'
```

Write the two `user.conf` files (these are the exact submitted settings):
```bash
docker exec gptoss-mlperf-fp8 bash -lc '
printf "gpt-oss-120b.Offline.target_qps = 35\ngpt-oss-120b.Offline.min_duration = 600000\n" \
  > /lab-mlperf-inference/results/user_offline.conf
printf "gpt-oss-120b.Server.target_qps = 19.5\ngpt-oss-120b.Server.min_duration = 900000\n" \
  > /lab-mlperf-inference/results/user_server.conf'
```

---

## 4. Runtime tunables (once per boot)

Run these **on the host** (they are host-level kernel/CPU settings). They are the
MLPerf-permissible host tunables; each is best-effort — skip any your kernel
rejects (e.g. partial `cpupower`).

```bash
echo 3    | sudo tee /proc/sys/vm/drop_caches                          > /dev/null   # drop caches
sudo cpupower frequency-set -g performance                                            # perf governor
sudo cpupower idle-set -d 2                             2>/dev/null || true            # disable deep C-state
echo 0    | sudo tee /proc/sys/kernel/nmi_watchdog                     > /dev/null   # NMI watchdog off
echo 0    | sudo tee /proc/sys/kernel/numa_balancing                   > /dev/null   # NUMA balancing off
echo 0    | sudo tee /proc/sys/kernel/randomize_va_space              > /dev/null   # ASLR off
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled     > /dev/null   # THP on
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag      > /dev/null   # THP defrag
```

---

## 5. Sanity check

```bash
docker exec gptoss-mlperf-fp8 bash -lc '
python3 -c "import mlperf_loadgen; print(\"loadgen OK\")"
# confirm the CLEAN v6.1 loadgen: these guarantee the checker passes with NO exceptions
grep -iE "^\*\.\*\.qsl_rng_seed" /tmp/mlperf_inference/loadgen/mlperf.conf   # = 2085463073848966840
cd /tmp/mlperf_inference/loadgen && git status -s -uno | head; echo "(empty = clean tree)"'
```

---

## 6. Run the benchmarks

All runs use `PYTHONPATH=/lab-mlperf-inference/code` (the harness sets it) and
`device_count=8`. Each run does a full model load (~7–8 min) first.

Helper (paths inside the container):
```bash
CODE=/lab-mlperf-inference/code ; RES=/lab-mlperf-inference/results
GO=$RES/gpt-oss-120b            # output root for the tree you will assemble
```

### 6.1 Offline — performance (~10 min run + load)

```bash
docker exec gptoss-mlperf-fp8 bash -lc '
export PYTHONPATH=/lab-mlperf-inference/code
mkdir -p /lab-mlperf-inference/results/gpt-oss-120b/Offline/performance/run_1
cd /tmp && python3 /lab-mlperf-inference/code/main.py \
  --config-path /lab-mlperf-inference/code/gpt-oss-120b/ --config-name offline_mi355x_fp8 \
  test_mode=performance harness_config.device_count=8 \
  harness_config.user_conf_path=/lab-mlperf-inference/results/user_offline.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Offline/performance/run_1 \
  vllm_engine_config.gpu_memory_utilization=0.95
grep -E "Tokens per second|Result is" \
  /lab-mlperf-inference/results/gpt-oss-120b/Offline/performance/run_1/mlperf_log_summary.txt'
```
> `gpu_memory_utilization=0.95` is the one **permissible** tune applied (MLPerf
> allows batch/KV/memory tuning); the config default is 0.90. Expect **VALID,
> ~30,198 tok/s**.

### 6.2 Offline — accuracy (~40 min)

Same command with `test_mode=accuracy`, `vllm_sampling_config.max_tokens=32768`,
output dir `…/Offline/accuracy`. (Scoring is step 7.)

### 6.3 Server — performance (~15 min run + load)

```bash
docker exec gptoss-mlperf-fp8 bash -lc '
export PYTHONPATH=/lab-mlperf-inference/code
mkdir -p /lab-mlperf-inference/results/gpt-oss-120b/Server/performance/run_1
cd /tmp && python3 /lab-mlperf-inference/code/main.py \
  --config-path /lab-mlperf-inference/code/gpt-oss-120b/ --config-name server_mi355x_fp8 \
  test_mode=performance harness_config.device_count=8 \
  harness_config.user_conf_path=/lab-mlperf-inference/results/user_server.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Server/performance/run_1
grep -E "Completed tokens per second|Result is|Performance constraints|Early stopping|percentile" \
  /lab-mlperf-inference/results/gpt-oss-120b/Server/performance/run_1/mlperf_log_summary.txt'
```
> Server config keeps cudagraphs on (`enforce_eager: False`), util 0.90. Expect
> **VALID, ~24,863 tok/s** @ 19.5 QPS, p99 TTFT ~685 ms, p99 TPOT ~78.5 ms. On a
> slower node, drop `target_qps` in `user_server.conf` (19.5 → 19.3 → 19.0) until
> `Result is : VALID` — 20 QPS is borderline (TPOT ~81 ms) on some MI300X nodes.

### 6.4 Server — accuracy (~40 min)

`server_mi355x_fp8`, `test_mode=accuracy`, `max_tokens=32768`, out `…/Server/accuracy`.

### 6.5 Compliance TEST07 + TEST09 (per scenario)

Both copy the test's `audit.config` into the run CWD, run the same config in
`test_mode=performance` (the audit.config forces accuracy logging), then verify.
Example — **Offline TEST09**:

```bash
docker exec gptoss-mlperf-fp8 bash -lc '
export PYTHONPATH=/lab-mlperf-inference/code
MLP=/lab-mlperf-inference/mlperf_inference; GO=/lab-mlperf-inference/results/gpt-oss-120b
OUT=$GO/Offline/compliance_run_TEST09; WD=/tmp/t09; rm -rf $WD $OUT; mkdir -p $WD $OUT; cd $WD
cp -f $MLP/compliance/TEST09/gpt-oss-120b/audit.config ./audit.config
python3 /lab-mlperf-inference/code/main.py \
  --config-path /lab-mlperf-inference/code/gpt-oss-120b/ --config-name offline_mi355x_fp8 \
  test_mode=performance harness_config.device_count=8 \
  harness_config.user_conf_path=$GO/../user_offline.conf \
  harness_config.output_log_dir=$OUT vllm_engine_config.gpu_memory_utilization=0.95
python3 $MLP/compliance/TEST09/run_verification.py -c $OUT -o $GO/Offline \
  --audit-config $MLP/compliance/TEST09/gpt-oss-120b/audit.config
tail -5 $GO/Offline/TEST09/verify_output_len.txt'
```

**TEST07** is the same shape but adds the GPQA dataset and 990 samples, and passes
an accuracy script to the verifier:
```
  harness_config.dataset_path=/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet \
  harness_config.total_sample_count=990 …
python3 $MLP/compliance/TEST07/run_verification.py -c $OUT -o $GO/<Scenario> \
  --accuracy-script "python3 $MLP/language/gpt-oss-120b/eval_mlperf_accuracy.py \
     --mlperf-log {accuracy_log} --reference-data /data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet \
     --tokenizer /model/gpt-oss-120b/fp4_quantized" \
  --audit-config $MLP/compliance/TEST07/gpt-oss-120b/audit.config
```
Run TEST07 + TEST09 for **both** Offline and Server (swap config name, user.conf,
and output scenario dir). Each `verify_*.txt` must say **PASS**.

> Run compliance in a **fresh container** and **never overlap** it with CPU-heavy
> accuracy scoring — the scorer's worker pool can starve the engine's host threads
> and kill the run (RCCL/collective timeout).

---

## 7. Score the accuracy runs

Writes `accuracy.txt` (`'exact_match': …`) next to each accuracy log. LiveCodeBench
code-exec is CPU-bound (~10–18 min, default 64 workers).

```bash
docker exec gptoss-mlperf-fp8 bash -lc '
ARCH=gfx942 bash /lab-mlperf-inference/code/scripts/check_gptoss_accuracy_scores.sh \
  /lab-mlperf-inference/results/gpt-oss-120b/Offline/accuracy/mlperf_log_accuracy.json'
# repeat for Server. Expect exact_match Offline ~83.7 %, Server ~82.8 % (≥ 82.30 %).
```
To score both at once safely, pin to disjoint cores so the 80 s per-item timeouts
stay honest and nothing oversubscribes:
```bash
taskset -c 0-63   bash -lc "ARCH=gfx942 bash …check_gptoss_accuracy_scores.sh <Offline>/…/mlperf_log_accuracy.json" &
taskset -c 64-127 bash -lc "ARCH=gfx942 bash …check_gptoss_accuracy_scores.sh <Server>/…/mlperf_log_accuracy.json" &
wait
```

Verify each accuracy run is complete before packaging:
```bash
grep -o qsl_idx <Scenario>/accuracy/mlperf_log_accuracy.json | wc -l   # want 4395
```

---

## 8. Assemble, truncate, and validate the submission

Lay the runs out under the MLPerf tree (see the submission you're reproducing for
the exact file set):
```
closed/MangoBoost/
  systems/$SYSTEM.json
  src/                                     # the harness ($SRC)
  measurements/$SYSTEM/gpt-oss-120b/<Scenario>/{measurements.json,user.conf,mlperf.conf,README.md}
  results/$SYSTEM/gpt-oss-120b/<Scenario>/{performance/run_1,accuracy,TEST07,TEST09}
```

**Truncate** the (multi-hundred-MB) accuracy logs — mandatory; appends a SHA to
`accuracy.txt`:
```bash
docker exec gptoss-mlperf-fp8 bash -lc '
cd /lab-mlperf-inference/mlperf_inference/tools/submission
python3 truncate_accuracy_log.py --input <submission_root> --output <submission_root_trunc> --submitter MangoBoost'
```

**Run the checker** — must pass with **NO** `--submission-exceptions`:
```bash
docker exec gptoss-mlperf-fp8 bash -lc '
cd /lab-mlperf-inference/mlperf_inference/tools/submission
python3 -m submission_checker.main --input <submission_root_trunc> --version v6.1 --submitter MangoBoost'
```
Expected: `All accuracy checks passed` (both scenarios), `TEST07/TEST09 passed`,
`Results=2, Closed Systems=1`, `SUMMARY: submission looks OK`.

> If the checker rejects `--version v6.1` ("invalid choice"), your loadgen clone is
> a checker copy that predates v6.1 — use a fresh `git clone` of
> `mlcommons/inference` master (as in step 3), which is v6.1-aware.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `qsl_rng_seed is wrong, expected=2085463073848966840` | loadgen built with old seeds — rebuild from **master HEAD** (step 3), which ships v6.1 seeds committed. |
| `… has loadgen errors … Loadgen built with uncommitted changes!` | you edited `mlperf.conf` without committing. Master HEAD needs no edit; if you must edit, `git commit` before `pip install .`. |
| Server `Result is : INVALID`, TPOT ~81 ms | node-limited at 20 QPS — lower `Server.target_qps` (19.5/19.3/19.0) until VALID. |
| `EngineDeadError` at the very end of a run | benign teardown race **iff** the run completed: check `grep -c qsl_idx …/mlperf_log_accuracy.json` = 4395 and the summary says "No errors encountered". |
| Per-dir `INVALID` file present | expected — fp8 model-MD5 ≠ fp4 reference. Not the run verdict. |
| Compliance run crashes / hangs | recreate a **fresh** container; don't run CPU scoring at the same time. |
| Accuracy a bit below target | `temperature=1` sampling stochasticity — re-run. The quantization (§1a) is deterministic, so the weights are not the variable. |

---

## Appendix — what "permissible tuning" means here

Per the MLPerf inference rules, only these were tuned: `gpu_memory_utilization`
(0.90→0.95, Offline), `max_num_seqs`/batch, cudagraph-vs-eager, kernel backend
(AITER off on stock), and host OS tunables (step 4). **Not** tuned (fixed by the
benchmark): output length, sampling `temperature`/`top_p`/`top_k`, LoadGen seeds,
`min_duration` (≥ 600 s), and query generation.
