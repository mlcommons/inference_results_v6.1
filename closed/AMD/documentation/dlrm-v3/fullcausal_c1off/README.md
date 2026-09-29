# Full-Causal (C1-off) Server Headroom Characterization — MI355X b24/b64

Captured 2026-06-05 on node `chi2835`. Question: with the 1024 sliding window **off**
(`DLRM_HSTU_MAX_ATTN_LEN=0`, i.e. full causal attention — the trained default and the
config NVIDIA/B200 run), what VALID Server throughput can MI355X sustain under the
80 ms p99 bound?

Stack held fixed at the b64 figure-of-record cert except the attention window:
A-FUSE fp8 (GEMM+attn), D2 epilogue fusion **on** (`FUSE_EPILOGUE=1`), 8 workers,
`sort_by_length=True` (default), last-layer target-only lever **off**. Only `BATCH`
and `target_qps` were varied. PROF90s probes (90 s `min_duration`, `target_latency=80`,
`percentile=99`).

> **Update (2026-06-06):** the b24/lever-off ~6,650 below was the *starting* point. After the
> last-layer target-only lever (D2 off) + a batch retune to **b40**, full causal now **certifies
> at 7,400 q/s** (p99 59.9 ms, 600 s PROD), +11.3%. See `TUNING_PLAN.md` for the full sweep,
> certs, and decomposition.

> **SUPERSEDED (2026-06-08):** the "b64 is INFEASIBLE" verdict below was true *for the slower fp8
> attention kernel*. With the **Win-B kernel** (`FASTMASK`+`FULLGRID`+`OCCTUNE`) and `INFLIGHT=128`,
> b64 full-causal now **certifies at 9,595 q/s VALID (p99 78.0 ms, 600 s PROD)** — the new C1-off
> figure of record. See the Win-B section in `../../HANDOFF_full_causal_optimization.md`. The b24/b40
> characterization below is kept as the kernel-limited baseline.

## Headline

- **b64 full causal is INFEASIBLE under the 80 ms bound.** p99 sits at ~90 ms at *every*
  offered load and gets *worse* at lower QPS — it is **compute/tail-bound, not load-bound**.
  A single b64 full-causal batch has a ~62 ms latency floor; the long-history tail drags
  p99 to ~90 ms regardless of queueing. There is no VALID knee at b64.
- **b24 is the feasible regime** (matches the prior `Plan22_AFUSE_ceiling_qps6000` anchor).
  **Max reliably-VALID ≈ 6,650 q/s** (p99 62.7 ms). The ceiling at ~6.7k is sharp and
  stochastic (6700 collapsed harder than 6750), so 6,650 is the honest VALID point.

## Sweep

### b64 (D2 on, lever off, C1 off) — infeasible
| offered | completed | p50 | p90 | p99 | result |
|---:|---:|---:|---:|---:|:---|
| 5000 | 4993 | 79.4 | 87.8 | 94.3 | INVALID |
| 6500 | 6485 | 78.5 | 85.4 | 91.1 | INVALID |
| 7000 | 6986 | 77.6 | 84.5 | 90.7 | INVALID |
| 8000 | 7720 | 380.7 | 1619.9 | 2816.2 | INVALID (saturated) |

p99 is flat ~90 ms across 5000–7000 and rises at lower QPS — the signature of a
compute/tail-bound (not queue-bound) regime. Min latency ~62 ms = one b64 full-causal batch.

### b24 (D2 on, lever off, C1 off) — feasible, knee at ~6.7k
| offered | completed | p50 | p90 | p99 | result |
|---:|---:|---:|---:|---:|:---|
| 6000 | 5994 | 32.2 | 35.7 | 40.4 | VALID |
| 6500 | 6485 | 32.2 | 36.9 | 46.4 | VALID |
| **6650** | **6642** | **32.5** | **38.4** | **62.7** | **VALID (max reliable)** |
| 6700 | 6692 | 32.7 | 41.7 | 512.1 | INVALID (collapse) |
| 6750 | 6741 | 32.6 | 39.0 | 84.5 | INVALID |
| 7000 | 6989 | 34.3 | 195.6 | 667.7 | INVALID (collapse) |

Min latency ~25 ms = one b24 full-causal batch (vs 62 ms at b64) — the reason b24 fits and
b64 cannot. The knee is a throughput ceiling: below it p99 is healthy (40–63 ms); at/above
it the queue blows up stochastically.

## What C1 (the 1024 window) is worth, system-level

| config | best VALID Server | batch |
|---|---:|---:|
| **C1-off (full causal)** | **~6,650 q/s** | 24 |
| C1-on (windowed) — prior cert | 10,595 q/s | 48 |
| C1-on (windowed) — figure of record | 11,994 q/s | 64 |

The window is worth **~+60% to +80%** attainable VALID throughput (~1.8×). It is a *double*
win: it caps per-query attention work to 1024 (less L² compute) **and** by lowering per-batch
latency it lets the server batch at b64 instead of being forced down to b24 (more amortization).
Turning C1 off forfeits both.

## Caveats / open items

- PROF90s probes are noisy near a sharp knee (6700 collapsed worse than 6750). The 6,650
  point should be confirmed with a 600 s PROD run before it is quoted as certified.
- ~~b24 is the reference batch, not proven optimal.~~ **Resolved:** batch sweep done — b40 is
  optimal (b24 6,800 -> b32 7,250 -> b40 7,400; b48 marginal/riskier). See `TUNING_PLAN.md`.
- ~~The lever was measured at b64 (infeasible) and must be re-measured at b24.~~ **Resolved:**
  re-measured at b24 (+2.3% VALID, big tail relief) and certified; net of batch tuning the lever
  config certifies at 7,400 q/s.

Files: each `*_summary.txt` is the run's `mlperf_log_summary.txt`. Full LoadGen MLLOG
`*_detail.txt` kept for the max-VALID b24 point and a b64-infeasible example; the rest remain
in `../../artifacts/Plan24_fullcausal_*`.
