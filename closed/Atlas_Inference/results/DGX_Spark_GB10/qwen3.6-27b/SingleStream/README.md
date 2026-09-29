# DGX Spark GB10 Qwen3.6-27B-NVFP4 — MLPerf-edge agentic submission

**Result** (this folder): wall **3834.4 s**, 1007/1007, BFCL **87.24 / 89.07 norm**, IoU **0.6231**; TTFT med 1153.7 ms, TPOT med 32.90 ms, TPS 20.08. Run on the **NVIDIA DGX Spark** (GB10 Grace Blackwell, sm_121a, 128 GB unified LPDDR5X, CUDA 13.0).
**Model**: `centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf` snapshot `43fff389a96d8132cdc0b532fcb6c4aeacf9d848` (dense, all-NVFP4 including GDN).
**Code**: atlas `perf/gb10-golden-conglomerate-2026-07-24` (PR #369) @ **`767cad97`**, served with `ATLAS_GDN_REGRESIDENT=1`. Decode work merged as PR #366.

## Reproduce

Source code and the full build / serve / benchmark instructions are in **[`../../../../src/GB10/`](../../../../src/GB10/)** — the engine at `767cad97` in [`src/GB10/atlas/`](../../../../src/GB10/atlas/), the harness at `bf9d12b` in [`src/GB10/endpoints/`](../../../../src/GB10/endpoints/), and the commands in [`src/GB10/README.md`](../../../../src/GB10/README.md).

`config.yaml` in this folder is the exact harness config used; run it unmodified. Seeds (mandated `mlperf.conf`, recorded in `config.yaml`): model **42**, scheduler **16159082839903944936**, dataloader **2747215439041700203**; `min_duration_ms` **600000**, `max_duration_ms` **14400000**.

Compliance: all substantive checks PASS (accuracy 87.24 ≥ 83.64, norm 89.07 ≥ 85.32, 995 samples, no dropped turns, all turns observed 1006/1006, temp 0, single-stream). The one `seed==42` line is a stale check — the ruleset itself mandates the large `mlperf.conf` seeds used here.
