# Benchmark configs

`staged/` holds the files a reviewer copies into the endpoints checkout before running. They are
inputs, not a record of what ran: each sbatch patches a handful of fields at launch.

The **effective** configuration for each submitted row is that row's own `config.yaml`:

| Row | staged input | effective config as run |
|---|---|---|
| Offline | `staged/submission_llama3_8b_offline.yaml` | `results/H200/llama3.1-8b/Offline/config.yaml` |
| Server | `staged/submission_llama3_8b_server_q58.yaml` | `results/H200/llama3.1-8b/Server/config.yaml` |

The effective configs are not duplicated here; the submitted `config.yaml` is the single copy.

## Staging step

Neither sbatch reads this directory. Both read
`$BASE/endpoints/examples/05_Llama3.1-8B_Example/`, so copy the staged files there first:

```
cp staged/*.yaml $BASE/endpoints/examples/05_Llama3.1-8B_Example/
```

## What the scripts patch at launch

| Field | staged value | effective value | Why |
|---|---|---|---|
| `model_params.name` | model path | served model id | Taken from the running server's `/v1/models`. |
| `model_params.tokenizer_name` | unset | Llama-3.1-8B-Instruct path | The engine id has no tokenizer on the hub, so client-side token metrics come back empty without it. |
| `model_params.top_k` | `-1` | `1` | TensorRT-LLM rejects `-1` (`samplingConfig.cpp:285`). At `temperature 0` the two are identical argmax. |
| `endpoint_config.endpoints` | `localhost:8000` | the job's port | Chosen per job below the ephemeral range. |
| `settings.runtime.n_samples_to_issue` | `13368` | `40104` (Offline) | Multiplied by `EPOCHS` (3 for the submitted Offline run), so measured duration clears the 600 s minimum. |
| `settings.load_pattern.target_qps` | `58` (Server) | `58` | Set from `$QPS`; the staged value is not read. |
| `settings.client.max_connections` | unset | `2048` | Unbounded pooling exhausts the ephemeral port range on multi-epoch runs. |
| `settings.client.warmup_connections` | unset | `256` | Same reason. |
| `report_dir` | placeholder | per-job output directory | |
| `timeout` | unset | `2400.0` (Offline), `3600.0` (Server) | Passed on the command line as `--timeout`, not through the YAML. |

Two further fields appear in the effective config without being patched. `model_params.streaming`
is resolved by the client from `type`: `offline` gives `off`, `online` gives `on`.
`settings.load_pattern.use_legacy_loadgen_qps_metrics` is a client default for the Poisson load
pattern. Both come from the endpoints package at the pinned commit rather than from anything in
this directory or in the scripts.

`staged/submission_llama3_8b_server_q58.yaml` was named and valued `q30` in the original
submission. `target_qps` is overridden from `$QPS` at launch, so the staged value was never used
and the submitted run was always 58; the file has been renamed and revalued so its name, comment
and value agree with the row it produces.
