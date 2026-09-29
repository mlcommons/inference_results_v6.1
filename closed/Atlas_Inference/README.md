# Atlas_Inference — MLPerf Inference v6.1 submission

Atlas_Inference submits two Edge / closed datapoints, both **Edge Agentic LLM (Qwen3.6-27B-NVFP4), SingleStream**, both served by the Atlas inference engine (`spark`) on a single unified-memory desktop:

1. **NVIDIA DGX Spark (GB10 Grace Blackwell)** — CUDA/NVFP4 build, W4A4 weights including GDN, chain-widened MTP verify K=4 with drafter context prefill/carry, and a GDN register-resident warm-replay prefill kernel. Wall **3834.4 s**, BFCL **87.24 / 89.07 norm**.
2. **AMD Strix Halo desktop (Ryzen AI Max+ 395 / Radeon 8060S, gfx1151)** — native-HIP NVFP4 build via a CUDA→HIP shim, MTP K=3 with drafter context-prefill. Wall **7108.6 s**, BFCL **86.23 / 87.76 norm**.

Both clear the Edge Agentic accuracy floor (83.64 raw / 85.32 normalized) at 995 samples, temperature 0, single-stream, with no dropped turns.

## Layout

| Directory | What |
|---|---|
| [`src/`](src/) | Engine and harness source for both systems, at the exact commits the results were produced with, plus full build / serve / benchmark instructions — [`src/GB10/`](src/GB10/) and [`src/Halo/`](src/Halo/) |
| [`systems/`](systems/) | System descriptions — `DGX_Spark_GB10.json`, `Strix_Halo.json` |
| [`results/`](results/) | Per-system performance and accuracy results, `measurements.json`, and the exact harness `config.yaml` used for each run |

To reproduce either result, start at [`src/GB10/README.md`](src/GB10/README.md) or [`src/Halo/README.md`](src/Halo/README.md).

No compliance tests are defined for Edge Agentic LLM in v6.1, so no `compliance/` directory is present.
