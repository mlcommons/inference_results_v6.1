# Strix Halo Qwen3.6-27B-NVFP4 — source, build, serve, benchmark

Everything needed to rebuild and re-run the Strix Halo SingleStream result in
[`../../results/Strix_Halo/qwen3.6-27b/SingleStream/`](../../results/Strix_Halo/qwen3.6-27b/SingleStream/) is in this directory.

| What | Where | Pinned at |
|---|---|---|
| Atlas inference engine (`spark`) | [`atlas/`](atlas/) | `eabfa8f` — branch `strix/consolidate-tmp` (PR #353), tag `mlperf-edge-strix-k3-20260724`, of [Avarok-Cybersecurity/atlas](https://github.com/Avarok-Cybersecurity/atlas) |
| Benchmark harness (`inference-endpoint`) | [`endpoints/`](endpoints/) | `edc7ea0` of [Palanivelg/endpoints](https://github.com/Palanivelg/endpoints) |
| Build config | `atlas/Cargo.toml`, `atlas/Cargo.lock`, `atlas/rust-toolchain.toml` | — |
| Kernel/model config | `atlas/kernels/strix-hip/qwen3.6-27b/nvfp4/MODEL.toml` | — |
| CUDA→HIP shim patches | `atlas/3rdparty_patches/`, `atlas/crates/atlas-kernels/hip/compat/` | — |
| Harness run config | [`../../results/Strix_Halo/qwen3.6-27b/SingleStream/config.yaml`](../../results/Strix_Halo/qwen3.6-27b/SingleStream/config.yaml) | the exact config used; `tokenizer_name` is a local path — see step 3 |

`atlas/` is the upstream tree at `eabfa8f` with `assets/`, `site/` and `book/` removed — a demo GIF and the docs website, none of which are build inputs. `endpoints/` is the unmodified upstream tree at `edc7ea0`. Nothing else is added, removed or edited; both are verifiable against the public repos with `git diff --no-index`.

**Result** (that folder): wall **7108.6 s**, 1007/1007, BFCL **86.23 / 87.76 norm**, IoU **0.6272**; TTFT med 2713 ms, TPOT med 47.3 ms. Run on the **Strix Halo desktop** (AMD gfx1151 / RDNA3.5, ROCm core-7.13).
**Model**: `nvidia/Qwen3.6-27B-NVFP4` snapshot `0893e1606ff3d5f97a441f405d5fc541a6bdf404`.

## Replicate

### 0. Get the code
Use the tree in this directory:
```bash
cd atlas
```
Or clone it fresh — the vendored tree is identical to the pinned commit:
```bash
git clone https://github.com/Avarok-Cybersecurity/atlas && cd atlas
git checkout mlperf-edge-strix-k3-20260724     # tag == eabfa8f
```

Get the weights — `$SNAP` in step 2 is the local snapshot directory this prints:
```bash
SNAP=$(huggingface-cli download nvidia/Qwen3.6-27B-NVFP4 \
         --revision 0893e1606ff3d5f97a441f405d5fc541a6bdf404)
```

### 1. Build  (needs `libibverbs-dev`; hip-port nvcc/libcuda shim, PR #326)
```bash
export ATLAS_TARGET_HW=strix-hip ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_TARGET_QUANT=nvfp4 \
       ATLAS_HIP_COMPAT_INCLUDE=$PWD/crates/atlas-kernels/hip/compat ATLAS_HIPCC=/opt/rocm/bin/hipcc \
       CUDARC_CUDA_VERSION=12080 PATH=$HOME/hip-port/fakebin:/opt/rocm/bin:$PATH \
       RUSTFLAGS="-L native=$HOME/hip-port/link" LIBRARY_PATH="$HOME/hip-port/link:/opt/rocm/core-7.13/lib"
cargo build --release -p spark-server --no-default-features --features cuda
```

### 2. Serve  (the ATLAS_* flags are load-bearing; keep --speculative)
```bash
export ATLAS_W4A16_DP4A=1 ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1 ATLAS_KV_EXTERNAL_RESERVE_GB=6 \
       ATLAS_SSM_TAIL_MIDCHUNK=1 ATLAS_SSM_TAIL_PROTECT=1 ATLAS_MTP_GATE_REPROBE=64 \
       ATLAS_MTP_DRAFTER_PREFILL=1 ATLAS_MTP_CARRY_DRAFTER=1
spark serve $SNAP --model-name nvidia/Qwen3.6-27B-NVFP4 --host 0.0.0.0 --port 8081 \
  --max-seq-len 65536 --gpu-memory-utilization 0.40 --kv-cache-dtype bf16 --max-batch-size 1 \
  --speculative --num-drafts 2 --mtp-quantization bf16 --mtp-vocab 100000 --disable-tool-grammar true \
  --enable-prefix-caching --ssm-cache-slots 64 --ssm-checkpoint-interval 16 --disable-thinking
```
`--num-drafts 2` is verify width **K=3**.

### 3. Benchmark + compliance  (harness: `endpoints/`, Palanivelg/endpoints @ `edc7ea0`)
```bash
cd ../endpoints && pip install -e .
inference-endpoint benchmark from-config \
  --config ../../results/Strix_Halo/qwen3.6-27b/SingleStream/config.yaml --mode both
PYTHONPATH=src python scripts/check_compliance.py results/<run> --ruleset mlperf-edge-current --model qwen3.6-27b
```
That `config.yaml` is the exact config used, left as-run. One line in it is machine-specific: `tokenizer_name` is the absolute path to the local snapshot of `nvidia/Qwen3.6-27B-NVFP4` @ `0893e1606ff3d5f97a441f405d5fc541a6bdf404` on the submission box. Point it at your own snapshot dir (`$SNAP` from step 0) or just at the repo id `nvidia/Qwen3.6-27B-NVFP4` — same tokenizer either way. Everything else runs unmodified. Seeds (mandated `mlperf.conf`, recorded in `config.yaml`): model **42**, scheduler **16159082839903944936**, dataloader **2747215439041700203**.

Compliance: all substantive checks PASS (accuracy 86.23 ≥ 83.64, norm 87.76 ≥ 85.32, 995 samples, no dropped turns, temp 0, single-stream). The one `seed==42` line is a stale check — the ruleset itself mandates the large `mlperf.conf` seeds used here.
