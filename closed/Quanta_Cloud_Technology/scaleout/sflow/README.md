# NV-sflow Orchestrator for Launching Endpoint-Based Multi-Node Benchmarks

This document covers how to launch endpoint-based multi-node benchmarks using `nv-sflow`, in both interactive and batch modes.

---

## Dependency

A SLURM-managed cluster with **Pyxis** and **Enroot** support is required. `nv-sflow` relies on Pyxis/Enroot to launch containerized job steps via `srun` across multiple nodes.

---

## Section 1: NV-SFlow Installation (on the slurm login node)

Install `uv` and `sflow` once:

```bash
# Download uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the virtual environment and install sflow
uv venv
uv pip install "sflow @ git+https://github.com/NVIDIA/nv-sflow.git@6efbf59c4a835473b08362eb623358e3ace31d6e"
source .venv/bin/activate
```

---

## Section 2: Container Image Selection

Each benchmark/scenario config pair carries the container image for that run: load the chosen `<benchmark>_config_sflow.yaml` together with its colocated `slurm_env_sflow.yaml`, and the SFlow template uses the pinned `CONTAINER_IMAGE` from those config files.

To inspect which image a run will use, open the `slurm_env_sflow.yaml` next to the selected `<benchmark>_config_sflow.yaml` and check `variables.CONTAINER_IMAGE`.


---

## Section 3.1: Quickstart — Interactive Launch on Login Node

The example below reuses the `gpt-oss-120b` GB200×72 Offline config but **scales it down to 16 GPUs / 4 nodes** for a quick smoke test, by overriding `DP_MULTIPLICITY` (and the loadgen target QPS in `HARNESS_EXTRA_ARGS`) on the command line. The config files themselves are unchanged.

```bash
sflow run \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set DP_MULTIPLICITY=16 \
  --set TEST_MODE=PerformanceOnly \
  --set HARNESS_EXTRA_ARGS="--offline_expected_qps=156" \
  --tui
```

What this does:

- Loads the GB200×72 Offline `gptoss_config_sflow.yaml` (DP_MULTIPLICITY default = 72) and overrides it to 16 via `--set DP_MULTIPLICITY=16`. With `GPUS_PER_DP_RANK = 1`, that's **16 GPUs across 4 nodes** (`16 / GPUS_PER_NODE(=4)`).
- Loads `slurm_env_sflow.yaml`, which provides the pinned container image and auto-derives `SLURM_NODES = ceil(TOTAL_GPUS / GPUS_PER_NODE)`, so dropping `DP_MULTIPLICITY` automatically cuts the allocation from 4 to 18. The command only needs to supply `WORK_DIR`.
- Overrides `HARNESS_EXTRA_ARGS` to scale the offline target QPS from 702 (full x72) to ~156 (= 702 × 16/72), keeping per-GPU load roughly constant so the run still finishes within `min_duration` without being bottlenecked.
- `--tui` shows the live workflow DAG / task status from the login-node sflow allocator.

For the Server scenario, swap `Offline` → `Server` in both yaml paths and replace the QPS override with `--server_target_qps=156` (server config defines `server_target_qps`, not `offline_expected_qps`).

For Interactive (disaggregated), use the disagg template and override the per-worker counts instead of `DP_MULTIPLICITY`:

```bash
sflow run \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Interactive/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Interactive/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_disagg_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set NUM_CTX_SERVERS=2 --set NUM_GEN_SERVERS=1 --set NUM_FRONTEND_SERVERS=1 \
  --set TEST_MODE=PerformanceOnly \
  --set HARNESS_EXTRA_ARGS="--server_target_qps=40" \
  --tui
```

That yields **2 CTX × 1 GPU + 1 GEN × 4 GPU = 6 GPUs** / 2 nodes (2 GPUs idle on the second node). The x72 baseline is 24 CTX + 12 GEN @ 480 QPS, so scaling the GEN side down by 12× gives `480 / 12 = 40 QPS` — which also matches the per-worker sustainability formula `min(2·(480/24), 1·(480/12)) = min(40, 40) = 40`.

---

## Section 3.2: Quickstart — Asynchronous Sbatch Job Launch

Same config files as Section 3.1, same `--set` overrides for the 16-GPU scaled run, but submitted as a non-blocking sbatch job. `sflow batch` writes the generated sbatch script under `-o` and (with `--submit`) hands it to `sbatch`.

```bash
sflow batch \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set DP_MULTIPLICITY=16 \
  --set TEST_MODE=PerformanceOnly \
  --set HARNESS_EXTRA_ARGS="--offline_expected_qps=156" \
  --nodes=4 \
  --partition=gb200 \
  --account=coreai_mlperf_inference \
  --time=04:00:00 \
  --job-name=gptoss-x72-offline-16gpu \
  -o build/sbatch_scripts_sflow/gptoss_x72_offline_16gpu.sh \
  --submit
```

Notes:

- `--nodes`, `--partition`, `--account`, `--time` must be passed on the CLI for `sflow batch` (the values in the `backends:` block of `slurm_env_sflow.yaml` are not auto-promoted to sbatch directives). Pick `--nodes` to match the scaled `TOTAL_GPUS / GPUS_PER_NODE`.
- The generated sbatch script handles everything else — including bootstrapping a compute-node sflow venv on first run — so subsequent submissions land much faster.
- For Interactive (2 CTX + 1 GEN = 6 GPUs / 2 nodes), swap the config paths to `.../Interactive/` and the template to `trtllm_disagg_loadgen.yaml`, pass `--set NUM_CTX_SERVERS=2 --set NUM_GEN_SERVERS=1 --set NUM_FRONTEND_SERVERS=1 --set TEST_MODE=PerformanceOnly --set HARNESS_EXTRA_ARGS="--server_target_qps=40"`, and set `--nodes=2`.

---

## Deep Dive

### nv-sflow Config Format

To kick off a run, several YAML files are needed — each configures a different component of the benchmarking infrastructure:


| Category                                          | Files                                                                                                        | Description                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **nv-sflow orchestrator configs**                 | `<benchmark>_config_sflow.yaml`, `slurm_env_sflow.yaml`                                                      | Defines benchmark topology, model paths, harness arguments, the pinned container image, and SLURM-environment variables such as number of nodes, partition, account, and time. |
| **Inference endpoint client configs**             | `<endpoint.yaml>`                                                                                            | Configures the benchmarking client which is used to generate a valid MLPerf endpoint submission run.                                                                                                                                                                                                        |
| **Backend-specific configs**                      | `<trtllm-serve-disagg-ctx ...>`, `<trtllm-serve-disagg-ctx-env ...>`, `<dynamo_config.yaml ...>`             | Configures each component used to launch the LLM server (ctx server, gen server, front-end server, router server, ...).                                                                                                                                                                                     |
| **Backend recipe templates in (sflow/templates)** | `<trtllm_disagg_endpoints.yaml>`, `<trtllm_ifb_endpoints.yaml>`, `<trtllm_disagg_dynamo_endpoints.yaml>` ... | The recipe template that starts a specific type of server for benchmarking: `trtllm_disagg`, `trtllm_ifb`, `trtllm_disagg_dynamo`, `vllm_ifb_dynamo`, ...                                                                                                                                                   |


### How the Configs Work Together

The nv-sflow orchestrator template (e.g. `trtllm_disagg_endpoints.yaml`) defines a series of job steps. Each step consists of a recipe to:

1. Launch the **server**
  1. depending on server types and complexity, they are launched differently.
2. Launch the **measuring client** that sends queries to the backend

Variables are aggregated by sflow command via `sflow run -f <benchmark_config_sflow.yaml> -f <slurm_env_sflow.yaml> -f <template.yaml> ....`:

At launch time, the experiment variables, SLURM variables, and recipe templates are supplied together. They combine into a complete benchmark run definition — specifying how to start the servers and how to start the client.
