# Lambda 8x B200 — OPEN division — Edge-Agentic (BFCL v4), model = Kimi-K2.6

## What this is
The MLPerf Inference **edge-agentic** reference workload
(`mlcommons/endpoints`, example `11_Edge_Agentic_Example`), run **unchanged** except that
the served model is **moonshotai/Kimi-K2.6** (Hugging Face) instead of the reference
`Qwen/Qwen3.6-27B`. Changing the reference model is what makes this an **open-division**
submission; every other parameter is the reference configuration.

- Model: **moonshotai/Kimi-K2.6** (`kimi_k25`, fp8), served with SGLang TP=8 on 8x B200.
- Reasoning disabled server-side (`chat_template thinking=false`) — the analog of the
  reference server's `--reasoning off`.
- Unchanged from the reference: `temperature 0`, `max_new_tokens 1024`, single-stream
  (`target_concurrency 1`), BFCL v4 category mix / sampling (`non_live 62%`, `live 10%`,
  `hallucination 10%`, `subset_floor 25`), agentic-coding performance replay (20 trajectories).

## Accuracy — BFCL v4 single-turn (995 samples)
- overall: **86.83%**  (reference Qwen3.6-27B = 86.23%; open division is not gated on this)
- normalized single-turn: **87.27%**
- by category: non_live **82.09%**, live **88.24%**, hallucination **91.48%**
- 995/995 scored, 0 empty, 0 missing.

## Performance — SingleStream agentic-coding replay (inline checker)
- inline IoU score: **0.6158**
- 1007 turns issued/completed, **0 failed, 0 missing** → valid run; ~776 s wall clock.
- See `performance/result_summary.json` and `performance/report.txt`.
  (Client-side token-rate metrics TPOT/OSL are empty: Kimi's tiktoken tokenizer is not a
  drop-in HF tokenizer for the client metrics aggregator, so it was omitted. SingleStream's
  primary metric is mean latency, which is present.)

## Reproduce
Serve Kimi-K2.6 (SGLang TP=8), then from the endpoints repo root:
`inference-endpoint benchmark from-config --config src/kimi-k2.6-edge-agentic/kimi_edge_full.yaml`
(BFCL scorer needs `pip install -e ".[bfcl]"`.)
