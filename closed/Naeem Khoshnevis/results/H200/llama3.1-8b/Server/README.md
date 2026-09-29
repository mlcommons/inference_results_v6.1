# llama3.1-8b — Server — 1x NVIDIA H200

Model: meta-llama/Llama-3.1-8B-Instruct
Precision: **FP8 (W8A8) + FP8 KV cache**, NVIDIA ModelOpt static per-tensor PTQ
Engine: **TensorRT-LLM 1.0.0** (`trtllm-serve`), TP1, GUARANTEED_NO_EVICT
Harness: MLCommons endpoints (LoadGen++), Poisson arrivals, streaming on, 16 client workers

Result: **7,376.60 tokens/s on a single H200** at `target_qps = 58` (achieved qps 57.63080321).

| | |
|---|---|
| Samples issued / completed | 40,104 / 40,104 |
| Failed samples | **0** |
| Measured duration | **695.88 s** (minimum is 600 s) |

Latency. Both figures are reported, and the gate is judged on the **early-stopping** percentile —
the statistically conservative one, which is what a LoadGen Server run is adjudicated on:

| Gate | Raw p99 | **Early-stopping p99 (binding)** | Limit | Headroom |
|---|---|---|---|---|
| TTFT | 1,463.20 ms | **1,495.94 ms** | 2,000 ms | **25.2%** |
| TPOT | 61.11 ms | **61.22 ms** | 100 ms | **38.8%** |

Sampling is greedy (`temperature 0`, `top_k 1`); `top_k = -1` is not accepted by the TensorRT-LLM
backend, and at temperature 0 the two are identical argmax.

**Accuracy was measured in the Server scenario, in this same run.** The performance and accuracy
figures come from one self-consistent online pass — `type: online`, `streaming: 'on'`,
`load_pattern: poisson, target_qps: 58.0` — so the accuracy below describes the exact serving path
that produced the throughput above.

| Metric | Achieved | Gate (99%) | Margin |
|---|---|---|---|
| ROUGE1 | 38.6354 | 38.391408 | +0.244 |
| ROUGE2 | 15.9256 | 15.748425 | +0.177 |
| ROUGEL | 24.4991 | 24.250743 | +0.248 |
| ROUGELSUM | 35.7316 | 35.435070 | +0.297 |
| gen_len | 8,222,691 | window [7,350,880 , 8,984,408] | inside |

Responses: 13,368 issued, 13,368 scored, **0 empty, 0 missing**.

Output sequence lengths, measured client-side by the harness tokenizer: total 1,711,025,
min 111, max 128, mean 127.9941, std dev 0.1769.

Compliance: KV-cache block reuse explicitly disabled, with **all three** flags pinned false
(`enable_block_reuse`, `enable_partial_reuse`, `copy_on_partial_reuse`). Server is the more
reuse-exposed scenario, since requests arrive over a 600 s+ window and later requests could
otherwise hit blocks left by earlier ones.

Compliance tests: see `documentation/README.md`.
