# Strix Halo Qwen3.6-27B-NVFP4 — MLPerf-edge agentic submission

**Result** (this folder): wall **7108.6 s**, 1007/1007, BFCL **86.23 / 87.76 norm**, IoU **0.6272**; TTFT med 2713 ms, TPOT med 47.3 ms. Run on the **Strix Halo desktop** (AMD gfx1151 / RDNA3.5, ROCm core-7.13).
**Model**: `nvidia/Qwen3.6-27B-NVFP4` snapshot `0893e1606ff3d5f97a441f405d5fc541a6bdf404`.
**Code**: atlas `strix/consolidate-tmp` (PR #353) @ **`eabfa8f`**, tag **`mlperf-edge-strix-k3-20260724`**.

## Reproduce

Source code and the full build / serve / benchmark instructions are in **[`../../../../src/Halo/`](../../../../src/Halo/)** — the engine at `eabfa8f` in [`src/Halo/atlas/`](../../../../src/Halo/atlas/), the harness at `edc7ea0` in [`src/Halo/endpoints/`](../../../../src/Halo/endpoints/), and the commands in [`src/Halo/README.md`](../../../../src/Halo/README.md).

`config.yaml` in this folder is the exact harness config used, left as-run. Its `tokenizer_name` is the absolute path to the local snapshot of `nvidia/Qwen3.6-27B-NVFP4` @ `0893e1606ff3d5f97a441f405d5fc541a6bdf404` on the submission box — point it at your own snapshot dir or the repo id `nvidia/Qwen3.6-27B-NVFP4`; everything else runs unmodified. Seeds (mandated `mlperf.conf`, recorded in `config.yaml`): model **42**, scheduler **16159082839903944936**, dataloader **2747215439041700203**.

Compliance: all substantive checks PASS (accuracy 86.23 ≥ 83.64, norm 87.76 ≥ 85.32, 995 samples, no dropped turns, temp 0, single-stream). The one `seed==42` line is a stale check — the ruleset itself mandates the large `mlperf.conf` seeds used here.
