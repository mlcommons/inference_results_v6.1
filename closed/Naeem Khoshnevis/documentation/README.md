# Documentation — Naeem Khoshnevis, Individual Submitter — MLPerf Inference v6.1

Closed division, Datacenter category. **2 results, 1 benchmark (llama3.1-8b), 1 system
(H200, 1x NVIDIA H200).**

| File | Contents |
|---|---|
| `calibration.md` | Quantization and calibration for llama3.1-8b (FP8 W8A8 + FP8 KV cache, NVIDIA ModelOpt static per-tensor PTQ), including a disclosure about how the calibration inputs were preprocessed. |

## Where the rest of the disclosure lives

- `results/H200/llama3.1-8b/<scenario>/README.md` — what that run was, its result, its
  accuracy, and anything a reviewer should know about it.
- `results/.../measurements.json` — starting weights, data types, weight transformations,
  retraining status.
- `systems/H200.json` — system description, including the disclosure that the run uses a
  **subset** of a 4-GPU node (1 GPU, with host CPU and RAM also cgroup-limited).
- `src/README.md` — what each shipped script does, what it depends on that is not shipped, and the
  KV-cache-reuse compliance note.

## Compliance tests

**TEST06 is not included, and we are raising that rather than leaving it to be discovered.**

v6.1 lists TEST06 as required for llama3.1-8b. Both results here are produced by the MLCommons
"endpoints" (LoadGen++) harness, and TEST06 cannot be generated from that harness:

- `compliance/TEST06/run_verification.py` reads `mlperf_log_accuracy.json`, a LoadGen artifact.
- The test also requires an `audit.config` that LoadGen ingests, verified through
  `mlperf_log_detail.txt`.

The endpoints harness emits neither file, so the test is not applicable to these runs rather than
omitted from them. The v6.1 submission checker reflects this: it disables all compliance checks for
endpoints results, so a clean checker run does not evidence TEST06 either way.

llama3.1-8b is listed in the checker's set of models permitted on the endpoints harness. We would
welcome guidance from the working group on how TEST06 should be satisfied for endpoints results, or
confirmation that it does not apply to them.

**What TEST06 exists to catch, and the evidence we can offer instead.** TEST06 guards against a
system reporting more tokens than it generated. In these runs token counts are measured
**client-side by the harness tokenizer**, not self-reported by the system under test, and both
scenarios report 13,368 of 13,368 responses scored with 0 empty and 0 missing. Per-sample output
sequence length statistics are recorded in each scenario's README and in
`accuracy/accuracy_results.json`.

## Note on reported units

For endpoints results the v6.1 checker publishes the **samples/s** field under a `Tokens/s` label —
a result of roughly 7,800 tokens/s appears as roughly 61. The shipped artifacts carry both values
correctly (`qps` and `tps` / `result_completed_tokens_per_second`). The mechanism is general to
endpoints submissions, but for this benchmark it affects only these two rows, which are the only
llama3.1-8b results produced on the endpoints harness. It is checker behavior, not a property of
these results.
