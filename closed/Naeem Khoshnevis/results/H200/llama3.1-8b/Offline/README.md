# llama3.1-8b — Offline — 1x NVIDIA H200

Model: meta-llama/Llama-3.1-8B-Instruct
Precision: **FP8 (W8A8) + FP8 KV cache**, NVIDIA ModelOpt static per-tensor PTQ
Engine: **TensorRT-LLM 1.0.0** (`trtllm-build` -> `trtllm-serve`), TP1
Harness: MLCommons endpoints (LoadGen++), 8 client workers, `max_new_tokens 128`, greedy

Result: **7,796.59 tokens/s on a single H200** (qps 60.91655307).

| | |
|---|---|
| Samples issued / completed | 40,104 / 40,104 (3 epochs of the 13,368-sample pool) |
| Failed samples | **0** |
| Measured duration | **658.34 s** (minimum is 600 s) |

Accuracy, measured in this same run over the full 13,368-sample dataset:

| Metric | Achieved | Gate (99%) | Margin |
|---|---|---|---|
| ROUGE1 | 38.6405 | 38.391408 | +0.249 |
| ROUGE2 | 15.9282 | 15.748425 | +0.180 |
| ROUGEL | 24.5018 | 24.250743 | +0.251 |
| ROUGELSUM | 35.7399 | 35.435070 | +0.305 |
| gen_len | 8,222,392 | window [7,350,880 , 8,984,408] | inside |

Responses: 13,368 issued, 13,368 scored, **0 empty, 0 missing**.

Output sequence lengths, measured client-side by the harness tokenizer: total 1,710,943,
min 111, max 130, mean 127.9880, std dev 0.2106. The max slightly exceeds the 128-token
`max_new_tokens` budget because the client re-encodes the returned text for its own token
accounting; the server-side generation budget itself was 128.

**Run duration.** The Offline scenario terminates on `n_samples_to_issue`, so a single epoch of
13,368 samples completes well inside the declared `min_duration_ms` and would leave the 600 s
minimum unexercised. This run issues **three** epochs (`EPOCHS=3` in
`src/llama3.1-8b/l31_8b_fp8_pipeline.sbatch`), giving 658.34 s of measured wall time against the
600 s floor. The minimum is satisfied on measured time, not merely declared in the config.

Compliance: KV-cache block reuse is explicitly disabled via `--extra_llm_api_options`
(`enable_block_reuse: false`, `enable_partial_reuse: false`, `copy_on_partial_reuse: false`),
because TensorRT-LLM 1.0.0 enables prefix-KV sharing across requests by default, which the
inference rules prohibit as cross-query caching. The memory fraction is set in the same YAML rather
than on the CLI, since `--extra_llm_api_options` replaces the whole `kv_cache_config` object.

Calibration: see `documentation/calibration.md`, including a disclosure about how the calibration
inputs were preprocessed.

Compliance tests: see `documentation/README.md`.
