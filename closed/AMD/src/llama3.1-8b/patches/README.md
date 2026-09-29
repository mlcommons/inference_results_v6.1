# In-container vLLM patches

vLLM is installed into the docker image's `site-packages`, so edits made inside a
running container are **lost when a new container is launched**. The repo's
`code/` directory is bind-mounted into every container, so patches are kept here
and re-applied at run time by `apply_vllm_patches.sh`.

The applier is **idempotent** (a no-op once already applied) and is invoked
automatically from `code/run.sh` and `code/run_harness.sh`.

## 0001 — ASM paged-attention: `high_precision=0`

`rocm_aiter_ops.paged_attention_common` (in `vllm/_aiter_ops.py`) did not pass
`high_precision`, so it fell back to aiter's default of `1`, selecting the
high-precision ASM kernel. We pin it to `0` to select the faster non-hp kernel.

| `high_precision` | aiter `.co` kernel (gfx950)                 |
|------------------|----------------------------------------------|
| 0 (this patch)   | `pa_bf16_pertokenFp8_gqa8_2tg_4w.co`         |
| 1 (vLLM default) | `pa_bf16_pertokenFp8_gqa8_2tg_4w_hp.co`      |
| 2                | `pa_bf16_pertokenFp8_gqa8_2tg_4w_uhp.co`     |

Validated on Llama3.1-8B Offline (MI355X, 8x):

- **Perf:** 1061.19 vs 1058.25 samples/s (hp=1 baseline) — fastest of the three.
- **Accuracy:** rougeL 24.4541 / rouge1 38.5278 / rouge2 16.0577 — passes the
  MLPerf 99% acceptance floor.

### Manual apply / revert

```bash
# apply (idempotent)
bash code/patches/apply_vllm_patches.sh

# revert: restore the backup the applier created next to the file
VLLM_DIR=$(python3 -c 'import os,vllm;print(os.path.dirname(vllm.__file__))')
cp "$VLLM_DIR/_aiter_ops.py.bak_hp" "$VLLM_DIR/_aiter_ops.py"
```

`0001-vllm-aiter-pa-high-precision0.patch` is the canonical unified diff of the
change for reference / code review.

## 0002 — aiter a4w4 (MXFP4) GEMM tuned config

`a4w4_blockscale_tuned_gemm.csv` replaces aiter's shipped
`aiter/configs/a4w4_blockscale_tuned_gemm.csv`. The shipped table only tuned
power-of-2 M values, so the actual decode batch sizes (the cudagraph capture
sizes, e.g. 2240, 5120, 6144) padded up to a worse kernel. We re-ran aiter's
`gemm_a4w4_blockscale_tune.py` over the 4 Llama3.1-8B linear shapes
(`4096×4096`, `6144×4096`, `28672×4096`, `4096×14336`) at every decode capture
size and added the winners (224 shapes improved, **+13–37%** on mid-range M).

Selection rules guarantee no regression: a row was only updated when the new
kernel was ≥2% faster, and every tuned kernel passed the exactness check
(errRatio 0.0), so it is strictly ≥ the shipped config per shape.

Measured impact:
- **Offline:** unchanged (1061.24 vs 1061.19 samples/s) — Offline runs at very
  large batch where the shipped CSV was already optimal, and is bandwidth-bound.
- **Server / Interactive:** expected to benefit (they operate in the mid/low-M
  range where the wins are); validate per scenario.

### Manual revert

```bash
AITER_DIR=$(python3 -c 'import os,aiter;print(os.path.dirname(aiter.__file__))')
cp "$AITER_DIR/configs/a4w4_blockscale_tuned_gemm.csv.bak_gemmtune" \
   "$AITER_DIR/configs/a4w4_blockscale_tuned_gemm.csv"
rm -rf /tmp/aiter_configs
```

To re-tune (e.g. new shapes/arch), add shapes to
`aiter/configs/a4w4_blockscale_untuned_gemm.csv` and run
`aiter_meta/csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py
-i <untuned> -o <tuned> --compare --update_improved`.

## 0003 — Prefill-admission scheduler control (OPT-IN, not auto-applied)

`apply_scheduler_patch.py` injects the MLPerf prefill-admission cadence into the
stock vLLM v1 scheduler and adds an adaptive bypass. **Important finding:** in the
`v0.22.0` image the build-time cadence patch is absent, so
`VLLM_SUBSEQUENT_DECODE_STEPS` / `VLLM_MIN_REQUEST_DECODE_STEP` set in the YAMLs
are **silently ignored** (verified: no such symbol in site-packages). This script
makes them live and adds:

- `VLLM_SUBSEQUENT_DECODE_STEPS` — pure-decode steps between prefill-admission steps.
- `VLLM_MIN_REQUEST_DECODE_STEP` — decode-batch floor below which prefill is always admitted.
- `VLLM_ADAPTIVE_PREFILL_WAITING_HWM` — if the waiting backlog reaches this, bypass the
  cadence and admit prefill this step (closed-loop TTFT drain). `0` = off.

All default to `0` → byte-for-byte stock behavior.

**NOT wired into `apply_vllm_patches.sh`** on purpose: the Offline/Server YAMLs
already set non-zero cadence values that are currently inert; auto-applying would
suddenly activate them and could regress those *already-validated* runs. Apply
deliberately, per scenario, and re-validate.

Measured on Llama3.1-8B **Interactive** (MI355X 8x, qps=790; TTFT p99 / result;
constraint = TTFT p99 ≤ 500 ms, TPOT p99 ≤ 30 ms):

| `SUBSEQUENT_DECODE_STEPS` | adaptive HWM | TTFT p99 | TPOT p99 | result |
|---|---|---|---|---|
| 0 (off / stock) | – | **456 ms** | 15.9 ms | **VALID** |
| 11 | off | 640 ms | 13.9 ms | INVALID |
| 11 | 64 | 650 ms | 14.6 ms | INVALID |

Conclusion: Interactive is **TTFT-bound with TPOT headroom**, so throttling prefill
is the wrong direction — keep the cadence **off** (`=0`, the stock default). The
cadence is only useful for **TPOT-bound** scenarios (Offline/Server with long decode),
where it should now actually take effect.

### Manual apply / revert

```bash
python3 code/patches/apply_scheduler_patch.py          # apply (idempotent)
VLLM_DIR=$(python3 -c 'import os,vllm;print(os.path.dirname(vllm.__file__))')
cp "$VLLM_DIR/v1/core/sched/scheduler.py.bak_sched" \
   "$VLLM_DIR/v1/core/sched/scheduler.py"              # revert to stock
```

## 0004 — Decode-batch cap under prefill backlog (INVERSE of cadence, opt-in)

`apply_decode_cap_patch.py` adds the *opposite* trade to the cadence: when a
prefill backlog exists it caps the per-step decode batch
(`VLLM_DECODE_CAP_BATCH`, activated at `VLLM_DECODE_CAP_WAITING_HWM`) so per-step
compute is reallocated decode→prefill, spending **TPOT headroom to lower TTFT**.
Running requests are rotated for fairness. Both env knobs default 0 → stock.

This is the correct *direction* for the TTFT-bound MLPerf scenarios, and it does
measurably move TTFT — but on this stack the gain is within run-to-run noise and
does not lift the valid QPS ceiling. Interactive @qps810 (stock cliff = 572ms):

| decode cap / hwm | TTFT p99 | TPOT p99 |
|---|---|---|
| off (stock) | 572 ms | 15.9 ms |
| 1536 / 32 | 531 ms | 15.6 ms |
| 1024 / 32 | 559 ms | 16.4 ms |
| 2048 / 16 | 550 ms | 17.0 ms |

At qps=800 (2 reps each) it did not beat stock: stock 465/536 ms vs cap1536
500/582 ms. Kept OFF.

### Manual revert (decode-cap)

```bash
VLLM_DIR=$(python3 -c 'import os,vllm;print(os.path.dirname(vllm.__file__))')
cp "$VLLM_DIR/v1/core/sched/scheduler.py.bak_decodecap" \
   "$VLLM_DIR/v1/core/sched/scheduler.py"
```

## 0005 — Scheduler instrumentation (READ-ONLY, diagnostic)

`apply_sched_stats_patch.py` records, per scheduler step, the prefill-token count,
new-prefill count and waiting-queue depth (histograms flushed to
`/tmp/sched_stats_<pid>.txt` every `VLLM_SCHED_STATS_EVERY` steps). Enabled only
when `VLLM_SCHED_STATS=1`; it **never changes scheduling** (default off = stock).
Wired into `apply_vllm_patches.sh` (harmless when disabled).

This is what localized the TTFT tail. On Interactive @790 (per engine, ~11k steps):

- prefill tokens/step: **mean ~1,100, max ~73,726** (full budget)
- new prefills/step: **mean ~2, max ~1,366**
- waiting depth: **mean ~65, max ~7,433**

i.e. steady state is trivial, but rare **bursts pack up to ~1,366 requests /
73,728 prefill tokens into a single ~50–80 ms "monster" step**. TTFT is
single-step-sensitive (everything queued behind that step waits the whole step),
while TPOT is averaged over ~100 output tokens so the same slow step barely moves
it — which is exactly why TTFT p99 tails (~570 ms) while TPOT p99 stays flat
(~16 ms, vs the 30 ms limit).

## 0006 — Prefill-tokens-per-step cap (TTFT p99 fix, ENABLED)

`apply_prefill_cap_patch.py` caps the **aggregate new-prefill tokens admitted per
step** to `VLLM_MAX_PREFILL_TOKENS_PER_STEP` (remaining waiting reqs go next step),
splitting the monster steps. Decode scheduling and the global token budget are
untouched; default `0` = stock. Wired into `apply_vllm_patches.sh`.

Why this and not the alternatives:
- **`enable_chunked_prefill`** (already on) only chunks a *single long prompt*; here
  prompts are short (`max_model_len` 2668) and the monster is *many* prompts, so it
  doesn't cap the aggregate.
- **`max_num_batched_tokens`** would cap it but also shrinks the shared decode budget
  and CUDA-graph sizing (mnbt=32768 gave ~no change); this cap is surgical.
- **decode-cap (0004)** can't help: decode is ~0.2 % of a monster step.

Measured on **Interactive** (MI355X 8x; constraint TTFT p99 ≤ 500 ms, TPOT p99 ≤ 30 ms):

| target qps | cap off | 12288 | 16384 |
|---|---|---|---|
| 790 | 455–499 (borderline) | 392 | 405–425 |
| 810 | **600 INVALID** | **372 VALID** | 429 VALID |
| 830 | – | 490 VALID (tight) | 447 VALID |
| 850 | – | 517 INVALID | – |

TPOT p99 stays ~15–18 ms throughout (nearly free). **`12288` lifts the valid
Interactive ceiling ~790 → 810** with a robust ~128 ms margin (baseline is INVALID
at 810); 830 is reachable (16384 → 447 ms) with less margin. Interactive YAML is set
to `VLLM_MAX_PREFILL_TOKENS_PER_STEP: 12288` and `target_qps` raised 790 → 810.

### Manual revert (prefill-cap / sched-stats)

```bash
VLLM_DIR=$(python3 -c 'import os,vllm;print(os.path.dirname(vllm.__file__))')
cp "$VLLM_DIR/v1/core/sched/scheduler.py.bak_prefillcap" \
   "$VLLM_DIR/v1/core/sched/scheduler.py"   # or .bak_schedstats
```

## 0007 — `merge_attn_states_kernel` recompile storm (Offline; warmup fix, NOT the kernel patch)

Symptom: on a cold Triton cache, **Offline** throughput dropped (~1020 vs ~1226
samples/s) with a rotating-straggler GPU (one GPU at ~63 % util) and a
0↔100 % utilization sawtooth, while Server/Interactive were unaffected.

Root cause: `vllm/v1/attention/ops/triton_merge_attn_states.py` declares
`prefill_tokens_with_context: tl.constexpr`. In the `rocm_aiter_fa` backend this
value defaults to `num_tokens` (the chunked-prefill query-token count of the
step), so Triton compiles a **separate kernel specialization per distinct value**.
Instrumentation showed ~1.5k distinct values (densely spanning 2..~2420). These
JIT-compiled **during the timed run** (measured: **28,360 Triton files / 422
`merge_attn_states` specializations per GPU** created between warmup-end and
run-end), invisible to `jit_monitor` (it dedups warnings by kernel name). Only
Offline hits this because it submits its whole 16.7k-sample shard at once (huge,
varied chunked-prefill steps); Server/Interactive are low-concurrency streams
that touch only a handful of values. Offline time = slowest engine, so each
engine's independent recompiles turned into the rotating straggler.

**What did NOT work — de-constexpr the kernel (`apply_merge_attn_states_patch.py`,
DISABLED).** Making `prefill_tokens_with_context` a runtime arg collapsed the
~422 specializations to a handful and gave the best perf (cold **1020 → 1242
samples/s**, 100 % stable util) — but it **broke accuracy**: ROUGE collapsed
(rouge1 ~10.2 / rougeL ~7.9 vs ~38.5 / ~24.4). Although the arg is only used as a
runtime threshold, lowering it from `tl.constexpr` hits a ROCm/Triton codegen path
that computes wrong attention merges here. The patch is left in-tree for reference
with its `apply_vllm_patches.sh` call **commented out**; do not re-enable without a
ROUGE pass.

**The fix that shipped — warmup precompilation (keeps the kernel byte-for-byte).**
`harness_llm/backends/common/warmup_merge_attn_states.py` precompiles the kernel
for `num_tokens = 1..max_model_len` (both `OUTPUT_LSE` variants) into the
per-device Triton cache **before** the engine starts, so nothing JITs on the timed
hot path. It runs in an **isolated subprocess** (invoked from
`vllm_engine.py::_run_merge_attn_states_warmup`) because the compile CUDA context
must be released before the engine allocates VRAM at `gpu_memory_utilization`
~0.97 — an in-process sweep OOMs EngineCore startup. Since it only populates the
cache, numerics are identical.

- Offline-only: called from `initialize_engine_and_generate`; Server/Interactive
  never invoke it. Default on; disable with `HARNESS_WARMUP_MERGE_ATTN_KERNELS=0`.
  Sweep cap via `HARNESS_MERGE_ATTN_MAX_TOKENS`, compile threads via
  `HARNESS_MERGE_ATTN_THREADS` (default 8).
- Cost: ~2.5 min untimed warmup on a cold container (5336 kernels/GPU, in
  parallel), then reused from the on-disk cache.

Validated on Llama3.1-8B **Offline** (MI355X 8x), cold cache:
- **Perf:** 1226–1231 samples/s (~157k tok/s), VALID, 99–100 % stable util
  (vs ~1020 cold without it) — 0 `merge_attn_states` recompiles during the timed run.
- **Accuracy:** rouge1 38.55 / rougeL 24.44 — PASS (identical to baseline; the
  kernel is unchanged).

> Note: an earlier **shape-diverse warmup** attempt (varying warmup prompt lengths
> in `offline_sut.py` / `async_server.py` / `sync_server.py` / `constants.py`) also
> broke accuracy the same way and was **reverted**; those files are at baseline. The
> merge-kernel precompile above is the only change kept, and only for Offline.

### Manual revert (merge-warmup)

```bash
# It is not a site-packages patch; disable via env or remove the call:
export HARNESS_WARMUP_MERGE_ATTN_KERNELS=0   # skip the precompile
```

## Overall finding (llama3.1-8b, vLLM v0.22, MI355X 8x)

| scenario | binding limit | prefill-cap (split monster steps) | cadence / decode-cap | best |
|---|---|---|---|---|
| Offline | throughput | n/a (no latency bound) | cadence −1.7…−8.6% | **stock (0)** |
| Server | TTFT@1031 (saturated) | *under test* | worse / no gain | tbd |
| Interactive | TTFT cliff (500 ms) | **valid ceiling 790 → 810 (+~3%), TTFT p99 −17%** | worse / noise | **cap=12288 @810** |

Net: the cadence knobs the YAMLs shipped were **inert** on v0.22 and don't help once
live (`VLLM_SUBSEQUENT_DECODE_STEPS: 0` everywhere). The real TTFT lever is the
**prefill-tokens-per-step cap (0006)**, which splits burst "monster" prefill steps
and converts spare TPOT into a lower TTFT p99, lifting the valid Interactive QPS
ceiling. Server tuning of the cap is in progress.
