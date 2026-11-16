# NVIDIA MLPerf Inference Harness package

`nv_mlpinf` is a pip-installable package that runs NVIDIA-optimized MLPerf Inference benchmarks.
It exposes a single CLI entry point, `nv-mlpinf`, for all benchmark actions. 

## Installation

Install from the repo root (`closed/NVIDIA/`) inside your container:

```bash
pip install ".[llm]"
```

Only one extras group is needed per container -- install the one matching your benchmark.

## CLI Usage

```
nv-mlpinf <action> [--benchmarks=<>] [--scenarios=<>] [options...]
```

### Actions


| Action              | Description                                      |
| ------------------- | ------------------------------------------------ |
| `run_harness`       | Run the MLPerf benchmark harness                 |
| `run_llm_server`    | Launch a TRT-LLM serving backend                 |
| `run_audit_harness` | Run compliance/audit tests                       |
| `generate_engines`  | Build TensorRT engines (Whisper only)            |
| `show_paths`        | Print all active runtime paths and their sources |


### Common options


| Option                     | Description                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `--benchmarks=<name>`      | Benchmark name (e.g. `deepseek-r1`). Run one benchmark at a time.                   |
| `--scenarios=<name>`       | Scenario: `Offline`, `Server`, `Interactive`, or `SingleStream`. Run one at a time. |
| `--system_name=<name>`     | Override auto-detected system ID (e.g. `GB200-NVL72_GB200-186GB_aarch64x72`)        |
| `--test_mode=AccuracyOnly` | Run in accuracy-only mode                                                           |
| `--test_run`               | Shorten minimum runtime from 10 min to 1 min (development use)                      |
| `--accuracy_target=.999`   | Use 99.9% accuracy target (high-accuracy variant, llama2 only)                      |


> **Tip:** Always specify exactly one `--benchmarks` and one `--scenarios`. Running multiple benchmarks or scenarios in a single invocation is not recommended

### Examples

```bash
# Start TRT-LLM server for DeepSeek-R1 Offline in Docker
nv-mlpinf run_llm_server --benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint

# Run harness against a running server in Docker
nv-mlpinf run_harness --benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint

# Accuracy run
nv-mlpinf run_harness --benchmarks=llama2-70b --scenarios=Offline --test_mode=AccuracyOnly

# Quick sanity check (1-minute run)
nv-mlpinf run_harness --benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint --test_run

# Override system name explicitly
nv-mlpinf run_harness --benchmarks=deepseek-r1 --scenarios=Offline \
    --system_name=GB200-NVL72_GB200-186GB_aarch64x72 --core_type=trtllm_endpoint

# Show active runtime paths
nv-mlpinf show_paths
```

## Path Configuration

Runtime paths (scratch space, model dir, build dir, etc.) are resolved from, in order:

1. Per-path environment variable (e.g. `MLPERF_SCRATCH_PATH`)
2. Config file at `~/.config/nv_mlpinf/paths.yml` (created on first run from a bundled template)
3. Hardcoded defaults (all rooted at `/work`)

To inspect or change active paths:

```bash
# Show current paths and their sources
nv-mlpinf show_paths

# Edit the config file
$EDITOR ~/.config/nv_mlpinf/paths.yml

# Or point to a custom config
NV_MLPINF_PATHS_CONFIG=/my/config.yml nv-mlpinf show_paths
```

The config file format:

```yaml
# ~/.config/nv_mlpinf/paths.yml
project_base_dir: /work
build_dir: /work/build
mlperf_scratch_path: /home/mlperf_inference_storage
trtllm_dir: /work/3rdparty/trtllm
mlcommons_inf_repo: /work/3rdparty/mlc-inference
results_submission_dir: /work/build/artifacts
results_staging_dir: /work/build/submission-staging
```

## System Configuration In Docker Environment

For any **Actions** defined in thie package, it has to run based on a detected system either via detection, or manual override. Ultimately, it is used to select the matching config directory under `configs/<benchmark name>/<system id>`. 

To override (e.g. for an unregistered system, or borrowing from any other configs):

```bash
# Via CLI (preferred)
nv-mlpinf run_harness --system_name=GB200-NVL72_GB200-186GB_aarch64x72 ...

# Via environment variable
SYSTEM_NAME=GB200-NVL72_GB200-186GB_aarch64x72 nv-mlpinf run_harness ...
```

## Code Development Guide

For internal package structure, architecture patterns, and code conventions, see [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md).