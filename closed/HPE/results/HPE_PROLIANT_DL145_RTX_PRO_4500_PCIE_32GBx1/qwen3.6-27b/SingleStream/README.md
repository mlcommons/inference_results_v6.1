To run this benchmark, first follow the setup and model-serving steps in
`endpoints/examples/11_Edge_Agentic_Example/README.md`.

Run the combined benchmark (performance + accuracy) with:

```bash
cd endpoints
inference-endpoint benchmark from-config \
	--config examples/11_Edge_Agentic_Example/online_edge_full_run.yaml
```

Before running, set `model_params.name` in
`examples/11_Edge_Agentic_Example/online_edge_full_run.yaml` to match the
served model name (for this run: `Qwen3.6-27B-Q4_K_M`), and ensure
`endpoint_config.endpoints` points to your OpenAI-compatible server.

This result is SingleStream on HPE ProLiant DL145 Gen11 with NVIDIA RTX PRO
4500, using Qwen3.6-27B (Q4_K_M), with BFCL v4 single-turn accuracy
`86.13%` over `995` samples.

Result files:

* `performance/run_1/config.yaml`
* `performance/result_summary.json`
* `accuracy/config.yaml`
* `accuracy/accuracy_results.json`
* `measurements.json`

For more details, please refer to
`endpoints/examples/11_Edge_Agentic_Example/README.md`.
