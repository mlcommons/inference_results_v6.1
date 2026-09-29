# src — Kimi-K2.6 on the MLPerf edge-agentic reference workload (open division)

`kimi_edge_full.yaml` is the edge-agentic reference config
(`mlcommons/endpoints` example `11_Edge_Agentic_Example/online_edge_full_run.yaml`)
with only the model-facing fields repointed at **moonshotai/Kimi-K2.6**:

- `model_params.name: kimi-k2.6` (served model), `chat_template_kwargs.thinking: false`
  (reference "reasoning off" analog).
- endpoint → local SGLang server (TP=8) serving Kimi-K2.6.
- RNG seeds set to the v6.1 submission_checker's required values
  (`sample_index=2747215439041700203`, `schedule=16159082839903944936`) and
  `min_duration_ms=600000`; the shipped reference example still carries the edge-device
  default `seed 42` / `min_duration_ms 0`, which the v6.1 checker rejects.

Everything else (datasets, BFCL category mix + sampling, single-stream load pattern,
`temperature 0`, `max_new_tokens 1024`) is the reference configuration, unchanged.

Run from the endpoints repo root:
`inference-endpoint benchmark from-config --config src/kimi-k2.6-edge-agentic/kimi_edge_full.yaml`
