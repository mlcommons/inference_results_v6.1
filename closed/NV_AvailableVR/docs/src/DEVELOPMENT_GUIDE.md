# Code Development Guide

Internal reference for the `nv_mlpinf` package — structure, patterns, and conventions.

## Package Structure


| Directory     | Purpose                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| `benchmarks/` | Per-benchmark implementation and README                                 |
| `common/`     | Shared utilities: path resolution, system detection, constants, logging |
| `ops/`        | Pipeline operations (harness, server, accuracy checker)                 |
| `fields/`     | Typed configuration fields that users can specify in CLI                |
| `llmlib/`     | LLM-specific server management, config, and harness ops                 |
| `configs/`    | Per-(benchmark, system, scenario) configuration files                   |


## Source Layout

```
src/nv_mlpinf/
├── main.py                  # Entry point (MainRunner)
├── __init__.py              # G_BENCHMARK_MODULES registry
├── benchmarks/              # Per-benchmark implementations
│   ├── llama2_70b/
│   ├── deepseek_r1/
│   ├── gpt_oss_120b/
│   ├── whisper/
│   ├── wan22_a14b/
│   ├── q3vl/
│   └── ...
├── fields/                  # nvmitten Field definitions (CLI args + config keys)
│   ├── meta.py              # action, benchmarks, scenarios
│   ├── harness.py           # harness-specific params
│   ├── loadgen.py           # LoadGen params (QPS, duration)
│   ├── models.py            # model params (batch size, precision)
│   ├── gen_engines.py       # engine build params
│   └── general.py           # misc (verbose, log_dir, config_dir)
├── ops/                     # Shared pipeline operations
│   ├── generate_engines.py  # EngineBuilderOp
│   ├── harness.py           # BenchmarkHarnessOp
│   └── loadgen.py           # LoadgenConfFilesOp
├── llmlib/                  # Shared LLM infrastructure
│   ├── config.py            # HarnessConfig, TrtllmEndpointConfig
│   ├── server.py            # TRT-LLM server management
│   ├── builder.py           # LLM engine building
│   ├── fields.py            # LLM-specific fields (TP, PP, trtllm_build_flags, etc.)
│   └── ...
├── scripts/                 # User-facing utilities
│   └── result_display.py    # Tabular result summary from logs
├── nv_mlpinf_paths.yml      # Bundled paths config template
└── common/                  # Shared utilities
    ├── constants.py         # Benchmark, Scenario, Action, HarnessType enums
    ├── paths.py             # Runtime path resolution (YAML config + env var + defaults)
    ├── workload.py          # WorkloadSetting
    └── systems/             # System detection and hardware definitions
        ├── system_list.py   # DETECTED_SYSTEM, known system configs
        └── known_hardware.py
```

## Entry Point

`src/nv_mlpinf/main.py` → `MainRunner` class (uses nvmitten configuration framework).

Detects system → loads benchmark module → loads system-specific config → executes requested action. Invoked via `nv-mlpinf` CLI (or `python -m nv_mlpinf`).

**Available actions:** `run_llm_server`, `run_harness`, `run_audit_harness`, `show_paths`, `display_results`

### System Name Resolution

Two-phase approach:

1. `SYSTEM_NAME` env var is read at module import time (`system_list.py`)
2. `--system_name` CLI arg overrides it after argparse, before `MainRunner` instantiation

## Fields System

Configuration parameters are defined as `nvmitten.configurator.Field` objects. Fields serve dual purpose: CLI argument parsing (via `@bind` decorators on `MainRunner`) and config file keys.


| Module              | Contents                                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `fields/meta.py`    | Action, benchmark, scenario selection                                                                            |
| `fields/harness.py` | Harness runtime params (`tensor_path`, `use_graphs`, `vboost_slider`)                                            |
| `fields/loadgen.py` | LoadGen params (`min_duration`, `offline_expected_qps`, `server_target_qps`)                                     |
| `fields/models.py`  | Model params (`gpu_batch_size`, `precision`, `input_dtype`)                                                      |
| `llmlib/fields.py`  | LLM-specific params (`tensor_parallelism`, `pipeline_parallelism`, `trtllm_build_flags`, `trtllm_runtime_flags`) |


## Code Conventions


| Convention        | Rule                                                                                                                                                            |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Branch naming     | `<type>-<username>-<details>` (e.g., `feat-pohanh-new-benchmark`, `fix-zhihanj-config-cleanup`)                                                                 |
| Copyright headers | Apache v2.0 required on all `.py`, `.cpp`, `.h`, `.sh`, `Makefile`, `Dockerfile` files. Configured in root `pyproject.toml` under `[tool.nvcopyright_headers]`. |
| Max line length   | 120 characters (pylint config in root `pyproject.toml`)                                                                                                         |


