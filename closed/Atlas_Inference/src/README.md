# Code

Source for both Atlas_Inference submissions, one directory per system. Each holds the Atlas inference engine (`spark`) and the benchmark harness (`inference-endpoint`) at the exact commit the result was produced with, plus the full build / serve / benchmark instructions.

| System | Directory | Engine | Harness |
|---|---|---|---|
| NVIDIA DGX Spark (GB10) | [`GB10/`](GB10/) | [`GB10/atlas/`](GB10/atlas/) @ `767cad97` | [`GB10/endpoints/`](GB10/endpoints/) @ `bf9d12b` |
| AMD Strix Halo desktop | [`Halo/`](Halo/) | [`Halo/atlas/`](Halo/atlas/) @ `eabfa8f` | [`Halo/endpoints/`](Halo/endpoints/) @ `edc7ea0` |

Start at [`GB10/README.md`](GB10/README.md) or [`Halo/README.md`](Halo/README.md) — each one lists where every config file lives and gives the build, serve and benchmark commands end to end.

The two engine trees are different commits of the same repo (a CUDA/NVFP4 build for GB10, a native-HIP build for gfx1151), so they are vendored separately rather than shared.

## Provenance

Upstream, all public:

| Vendored path | Upstream | Commit |
|---|---|---|
| `GB10/atlas/`, `Halo/atlas/` | [Avarok-Cybersecurity/atlas](https://github.com/Avarok-Cybersecurity/atlas) (AGPL-3.0) | `767cad97` (branch `perf/gb10-golden-conglomerate-2026-07-24`, PR #369) / `eabfa8f` (branch `strix/consolidate-tmp`, PR #353, tag `mlperf-edge-strix-k3-20260724`) |
| `GB10/endpoints/` | [mlcommons/endpoints](https://github.com/mlcommons/endpoints) | `bf9d12b` |
| `Halo/endpoints/` | [Palanivelg/endpoints](https://github.com/Palanivelg/endpoints) | `edc7ea0` |

The `atlas/` trees are the upstream trees at those commits with `assets/`, `site/` and `book/` removed — a demo GIF, the logo, and the docs website, none of which are build inputs. The `endpoints/` trees are unmodified. Nothing else is added, removed or edited, so each tree reproduces exactly:

```bash
git clone https://github.com/Avarok-Cybersecurity/atlas /tmp/atlas && git -C /tmp/atlas checkout 767cad97
rm -rf /tmp/atlas/{.git,assets,site,book}
diff -r --no-dereference /tmp/atlas GB10/atlas     # no output
```

The GDN register-resident prefill kernel used for the GB10 result has since been merged to `main` and made default-on as `063d87ee`; the pinned tree here keeps it behind `ATLAS_GDN_REGRESIDENT=1`, which is how the submitted run was served.
