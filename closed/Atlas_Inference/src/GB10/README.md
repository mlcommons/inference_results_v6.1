# DGX Spark GB10 Qwen3.6-27B-NVFP4 — source, build, serve, benchmark

Everything needed to rebuild and re-run the GB10 SingleStream result in
[`../../results/DGX_Spark_GB10/qwen3.6-27b/SingleStream/`](../../results/DGX_Spark_GB10/qwen3.6-27b/SingleStream/) is in this directory.

| What | Where | Pinned at |
|---|---|---|
| Atlas inference engine (`spark`) | [`atlas/`](atlas/) | `767cad97` — branch `perf/gb10-golden-conglomerate-2026-07-24` (PR #369) of [Avarok-Cybersecurity/atlas](https://github.com/Avarok-Cybersecurity/atlas) |
| Benchmark harness (`inference-endpoint`) | [`endpoints/`](endpoints/) | `bf9d12b` of [mlcommons/endpoints](https://github.com/mlcommons/endpoints) |
| Build config | `atlas/Cargo.toml`, `atlas/Cargo.lock`, `atlas/rust-toolchain.toml` | — |
| Kernel/model config | `atlas/kernels/gb10/qwen3.6-27b/nvfp4/MODEL.toml` | — |
| Harness run config | [`../../results/DGX_Spark_GB10/qwen3.6-27b/SingleStream/config.yaml`](../../results/DGX_Spark_GB10/qwen3.6-27b/SingleStream/config.yaml) | the exact config used; run it unmodified |

`atlas/` is the upstream tree at `767cad97` with `assets/`, `site/` and `book/` removed — a demo GIF and the docs website, none of which are build inputs. `endpoints/` is the unmodified upstream tree at `bf9d12b`. Nothing else is added, removed or edited; both are verifiable against the public repos with `git diff --no-index`.

**Result** (that folder): wall **3834.4 s**, 1007/1007, BFCL **87.24 / 89.07 norm**, IoU **0.6231**; TTFT med 1153.7 ms, TPOT med 32.90 ms, TPS 20.08. Run on the **NVIDIA DGX Spark** (GB10 Grace Blackwell, sm_121a, 128 GB unified LPDDR5X, CUDA 13.0).
**Model**: `centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf` snapshot `43fff389a96d8132cdc0b532fcb6c4aeacf9d848` (dense, all-NVFP4 including GDN).

The GDN register-resident warm-replay kernel used here is **merged upstream and public**: folded default-on in **`063d87ee`** on `main` — "perf(mlperf-edge): fold the GDN register-resident prefill lever and make the GB10 K=4 submission replicable (#369)". On current `main` the kernel is the default and no flag is needed; the opt-out is `ATLAS_NO_GDN_REGRESIDENT=1`. The tree vendored here is the pinned commit, where the kernel is still behind `ATLAS_GDN_REGRESIDENT=1`. Decode work merged as PR #366.

## Replicate

### 0. Get the code
Use the tree in this directory:
```bash
cd atlas
```
Or clone it fresh — the vendored tree is identical to the pinned commit:
```bash
git clone https://github.com/Avarok-Cybersecurity/atlas && cd atlas
git checkout 767cad97      # or: git checkout main   (kernel is default-on there)
```

Get the weights — `$SNAP` in step 2 is the local snapshot directory this prints:
```bash
SNAP=$(huggingface-cli download centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
         --revision 43fff389a96d8132cdc0b532fcb6c4aeacf9d848)
```

### 1. Build  (`ATLAS_TARGET_MODEL` is load-bearing — omitting it builds a different kernel set)
```bash
PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-27b \
  cargo build --release -p spark-server --bin spark --features cuda
```
The build log must report `compiled 158 kernels for target 0 (gb10, qwen3.6-27b, nvfp4)`.

### 2. Serve  (the ATLAS_* flags are load-bearing; keep --speculative)
```bash
export ATLAS_NO_FFN_NVFP4_MMQ=1 ATLAS_SSM_TAIL_MIDCHUNK=0 ATLAS_MTP_CATCHUP=0 \
       ATLAS_MTP_DRAFT_CONF=0.0 ATLAS_MTP_GATE_FORCE=1 ATLAS_SSM_TAIL_PROTECT=1 \
       ATLAS_SSM_TAIL_LEASE_TTL=128 ATLAS_BF16_TC_PREFILL=1 ATLAS_GDN_REGRESIDENT=1
spark serve $SNAP --model-name centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf --host 0.0.0.0 --port 8888 \
  --max-seq-len 32768 --gpu-memory-utilization 0.70 --kv-cache-dtype bf16 --max-batch-size 1 \
  --speculative --num-drafts 3 --mtp-quantization bf16 --disable-tool-grammar true \
  --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
  --tool-call-parser qwen3_xml --disable-thinking
```
`--num-drafts 3` is verify width **K=4** (`--num-drafts N` gives K=N+1). MTP drafter context (prefill + cross-turn carry) is ON by default — the older `ATLAS_MTP_DRAFTER_PREFILL` / `ATLAS_MTP_CARRY_DRAFTER` names are obsolete and ignored. Keep `--gpu-memory-utilization 0.70`: GB10 memory is unified and the box OOM-freezes above it. `ATLAS_*` presence-flags treat `=0` as ENABLED — to disable one, **omit** it rather than setting it to 0. On `main`, drop `ATLAS_GDN_REGRESIDENT=1` (it is the default).

### 3. Benchmark + compliance  (harness: `endpoints/`, mlcommons/endpoints @ `bf9d12b`)
```bash
cd ../endpoints && pip install -e .
inference-endpoint benchmark from-config \
  -c ../../results/DGX_Spark_GB10/qwen3.6-27b/SingleStream/config.yaml --mode both -v
PYTHONPATH=src python scripts/check_compliance.py results/<run> --ruleset mlperf-edge-current --model qwen3.6-27b
```
That `config.yaml` is the exact config used; run it unmodified. Seeds (mandated `mlperf.conf`, recorded in `config.yaml`): model **42**, scheduler **16159082839903944936**, dataloader **2747215439041700203**; `min_duration_ms` **600000**, `max_duration_ms` **14400000**.

Compliance: all substantive checks PASS (accuracy 87.24 ≥ 83.64, norm 89.07 ≥ 85.32, 995 samples, no dropped turns, all turns observed 1006/1006, temp 0, single-stream). The one `seed==42` line is a stale check — the ruleset itself mandates the large `mlperf.conf` seeds used here.
