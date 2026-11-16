# Config Authorship - Agent Guidelines

This document is the source of truth for AI agents creating or editing configs in
this directory.

---

## Description

Per-model, per-system, per-scenario serving configurations for MLPerf Inference
benchmarks. Each config entry maps to one combination of
`<model_name>/<system_config>/<framework>/<scenario>/`.

The repo supports two launch paths:

- **Single-node Docker** uses `make run_llm_server` and `make run_harness`
  through `nv-mlpinf`. Add `server.py` when this path is supported.
- **Multi-node SLURM** uses nv-sflow. A run combines the scenario's
  `<benchmark_name>_config_sflow.yaml`, the colocated `slurm_env_sflow.yaml`,
  and one template from `scaleout/sflow/templates/`.

Both launch paths use the same `harness.py` and the same TRT-LLM YAML files.
Existing configs may still contain `server-topology.json` for legacy scaleout
compatibility; nv-sflow does not use it as the primary config source.

## Directory Structure

```text
configs/
  <model_name>/
    <system_config>/
      <framework>/
        <scenario>/
          harness.py
          server.py                         # add for single-node Docker
          <benchmark_name>_config_sflow.yaml # add for multi-node SLURM nv-sflow
          slurm_env_sflow.yaml              # add for multi-node SLURM nv-sflow
          trtllm-serve-<mode>-<parallelism>.yaml
          trtllm-serve-<mode>-<parallelism>-env.yaml
```

A scenario directory may support Docker, nv-sflow, or both. Having both
`server.py` and the two nv-sflow YAML files means the config supports both launch
paths.

---

## Naming Rules - Follow Exactly

`<model_name>` - lowercase, underscores only.

- `gpt_oss_120b`

`<system_config>` - must match the MLPerf system ID exactly, including case and
hyphens.

- `GB200-NVL72_GB200-186GB_aarch64x4`
- `GB200-NVL72_GB200-186GB_aarch64x72`

`<scenario>` - one of: `Offline`, `Server`, `Interactive`

`<framework>` - use `TRTLLM` for TensorRT-LLM configs.

**TRT-LLM YAML filenames** use this pattern:

```text
trtllm-serve-<mode>-<parallelism>.yaml
trtllm-serve-<mode>-<parallelism>-env.yaml
```

| `<mode>` value | When to use                                              |
| -------------- | -------------------------------------------------------- |
| `ifb`          | In-flight batching serving strategy                      |
| `disagg-ctx`   | Disaggregated serving strategy, context/prefill worker   |
| `disagg-gen`   | Disaggregated serving strategy, generation/decode worker |

| `<parallelism>` examples | Meaning                                    |
| ------------------------ | ------------------------------------------ |
| `1gpu`                   | 1 GPU per rank                             |
| `tp1pp4`                 | Non-MoE: TP=1, PP=4                        |
| `dep8`                   | MoE with attention DP enabled, EP=8        |
| `tep8`                   | MoE with attention DP disabled, EP=8       |
| `4gpu`                   | 4 GPUs per rank                            |

**MoE config naming:**

- Use `dep<N>` if `enable_attention_dp: true`, where `<N>` is
  `moe_expert_parallel_size`.
- Use `tep<N>` if `enable_attention_dp: false`, where `<N>` is
  `moe_expert_parallel_size`.

Every main YAML must have a paired `-env.yaml`.

**Non-MoE IFB example:**

```text
trtllm-serve-ifb-tp1pp4.yaml
trtllm-serve-ifb-tp1pp4-env.yaml
```

**MoE IFB example:**

```text
trtllm-serve-ifb-dep8.yaml
trtllm-serve-ifb-dep8-env.yaml
```

**Disaggregated Interactive example:**

```text
trtllm-serve-disagg-ctx-1gpu.yaml
trtllm-serve-disagg-ctx-1gpu-env.yaml
trtllm-serve-disagg-gen-4gpu.yaml
trtllm-serve-disagg-gen-4gpu-env.yaml
```

---

## nv-sflow Benchmark Config

Use `<benchmark_name>_config_sflow.yaml` for multi-node SLURM benchmarking with
nv-sflow. This file supplies workflow variables consumed by
`scaleout/sflow/templates/trtllm_ifb_loadgen.yaml` or
`scaleout/sflow/templates/trtllm_disagg_loadgen.yaml`.

For IFB `Offline` and `Server` configs, include the DP topology, benchmark
metadata, system name, model path, and TRT-LLM YAML paths:

```yaml
variables:
  DP_MULTIPLICITY:
    type: integer
    value: <int>
  GPUS_PER_DP_RANK:
    type: integer
    value: <int>
  TOTAL_GPUS:
    type: integer
    value: ${{ variables.DP_MULTIPLICITY * variables.GPUS_PER_DP_RANK }}

  BASE_PORT:
    value: 8336

  MODEL_PATH:
    value: /home/mlperf_inference_storage/models/<model_dir>
  TRTLLM_YAML:
    value: /work/configs/<model>/<system>/TRTLLM/<scenario>/trtllm-serve-ifb-<parallelism>.yaml
  TRTLLM_ENV_YAML:
    value: /work/configs/<model>/<system>/TRTLLM/<scenario>/trtllm-serve-ifb-<parallelism>-env.yaml

  BENCHMARK:
    value: <benchmark-cli-name>
  SCENARIO:
    value: <scenario>
  SYSTEM_NAME:
    value: <system>
  TEST_MODE:
    value: PerformanceOnly
  HARNESS_EXTRA_ARGS:
    value: ""
```

For disaggregated `Interactive` configs, use separate context, generation, and
frontend counts:

```yaml
variables:
  NUM_CTX_SERVERS:
    type: integer
    value: <int>
  GPUS_PER_CTX_RANK:
    type: integer
    value: <int>
  NUM_GEN_SERVERS:
    type: integer
    value: <int>
  GPUS_PER_GEN_RANK:
    type: integer
    value: <int>
  NUM_FRONTEND_SERVERS:
    type: integer
    value: <int>
  TOTAL_GPUS:
    type: integer
    value: ${{ variables.NUM_CTX_SERVERS * variables.GPUS_PER_CTX_RANK + variables.NUM_GEN_SERVERS * variables.GPUS_PER_GEN_RANK }}

  BASE_PORT:
    value: 8336
  FRONTEND_PORT:
    value: 8000

  MODEL_PATH:
    value: /home/mlperf_inference_storage/models/<model_dir>
  TRTLLM_YAML_CTX:
    value: /work/configs/<model>/<system>/TRTLLM/Interactive/trtllm-serve-disagg-ctx-<parallelism>.yaml
  TRTLLM_ENV_YAML_CTX:
    value: /work/configs/<model>/<system>/TRTLLM/Interactive/trtllm-serve-disagg-ctx-<parallelism>-env.yaml
  TRTLLM_YAML_GEN:
    value: /work/configs/<model>/<system>/TRTLLM/Interactive/trtllm-serve-disagg-gen-<parallelism>.yaml
  TRTLLM_ENV_YAML_GEN:
    value: /work/configs/<model>/<system>/TRTLLM/Interactive/trtllm-serve-disagg-gen-<parallelism>-env.yaml

  BENCHMARK:
    value: <benchmark-cli-name>
  SCENARIO:
    value: Interactive
  SYSTEM_NAME:
    value: <system>
  TEST_MODE:
    value: PerformanceOnly
  HARNESS_EXTRA_ARGS:
    value: ""
```

**Invariants to verify before saving:**

- IFB: `DP_MULTIPLICITY * GPUS_PER_DP_RANK` equals the GPU count in
  `<system_config>`.
- Disaggregated: `NUM_CTX_SERVERS * GPUS_PER_CTX_RANK + NUM_GEN_SERVERS *
  GPUS_PER_GEN_RANK` equals the GPU count in `<system_config>`.
- `TRTLLM_YAML` and `TRTLLM_ENV_YAML`, or the ctx/gen variants, point to files
  in the same scenario directory.
- `SYSTEM_NAME` matches the folder name unless intentionally overriding it for a
  custom system.
- `BENCHMARK` uses the CLI benchmark name, such as `deepseek-r1` or
  `gpt-oss-120b`.

---

## nv-sflow SLURM Environment Config

Use `slurm_env_sflow.yaml` for cluster and container settings. It defines the
container image, mounts, SLURM account/partition/time, GPU shape, and the
`test_container` operator used by the nv-sflow templates.

Key fields to verify:

- `CONTAINER_IMAGE` points to the intended release or development image.
- `WORK_DIR` is supplied at launch with `--set WORK_DIR=$PWD`.
- `SCRATCH_DIR` points to the host model/dataset root mounted at
  `/home/mlperf_inference_storage`.
- `GPUS_PER_NODE` matches the cluster node shape.
- `SLURM_NODES` is computed from `TOTAL_GPUS` and `GPUS_PER_NODE`, or is set
  explicitly when needed.
- `SLURM_ACCOUNT`, `SLURM_PARTITION`, and `SLURM_TIME` match the target cluster.

Example launch shape:

```bash
sflow batch \
  -f configs/<model>/<system>/TRTLLM/<scenario>/<benchmark_name>_config_sflow.yaml \
  -f configs/<model>/<system>/TRTLLM/<scenario>/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set TEST_MODE=PerformanceOnly \
  --set SLURM_ACCOUNT=<account> \
  --set SLURM_PARTITION=<partition> \
  --nodes=<nodes> \
  --partition=<partition> \
  --account=<account> \
  --submit
```

Use `scaleout/sflow/templates/trtllm_disagg_loadgen.yaml` for disaggregated
Interactive configs.

---

## Single-node Docker Server Config

Use `server.py` for single-node Docker benchmarking only. It is loaded by
`nv-mlpinf run_llm_server`.

Reference YAML files through `trtllm_yml_override` and `env_yml_override` using
`paths.PROJECT_BASE_DIR`.

```python
import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.llmlib.fields as llm_fields

# Used in single-node benchmark with Docker environment only.

ifb_config = {
    llm_fields.trtllm_yml_override: paths.PROJECT_BASE_DIR / 'configs/<model>/<system>/TRTLLM/<scenario>/trtllm-serve-ifb-<parallelism>.yaml',
    llm_fields.env_yml_override: paths.PROJECT_BASE_DIR / 'configs/<model>/<system>/TRTLLM/<scenario>/trtllm-serve-ifb-<parallelism>-env.yaml',
    model_fields.precision: '<precision>',
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): ifb_config,
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": ifb_config,
    },
}
```

`server.py` should reference YAML files under the same scenario directory.

---

## LLM Benchmarking Harness Config

Use `harness.py` for LoadGen and harness settings. It is required for both
Docker and nv-sflow paths.

### Required QPS Fields

| Scenario  | Required field                        |
| --------- | ------------------------------------- |
| `Offline` | `loadgen_fields.offline_expected_qps` |
| `Server`  | `loadgen_fields.server_target_qps`    |

### `server_instance_size`

- Single-node Docker workflow: set `llm_fields.server_instance_size` to the GPU
  count for the instance.
- Multi-node SLURM nv-sflow workflow: optional.

Reference example:
`configs/gpt_oss_120b/B300-SXM-270GBx8/TRTLLM/Server/harness.py`

### `max_concurrency`

Represents each endpoint's maximum supported concurrency.

| Serving mode                  | Scope of `max_concurrency`         |
| ----------------------------- | ---------------------------------- |
| IFB multiple server instances | Per individual IFB server instance |
| Disaggregated serving         | Per disagg frontend/master server  |

---

## Checklist: Adding a New Config Entry

Complete every applicable item before considering the entry done.

- Directory created:
  `configs/<model_name>/<system_config>/<framework>/<scenario>/`
- `harness.py` added with LoadGen parameters and QPS targets for the scenario.
- Main TRT-LLM YAML added:
  `trtllm-serve-<mode>-<parallelism>.yaml`
- Env YAML added:
  `trtllm-serve-<mode>-<parallelism>-env.yaml`
- If single-node Docker is supported, `server.py` added with correct
  `trtllm_yml_override` and `env_yml_override`.
- If multi-node SLURM nv-sflow is supported,
  `<benchmark_name>_config_sflow.yaml` and `slurm_env_sflow.yaml` added.
- If IFB nv-sflow is supported, use
  `scaleout/sflow/templates/trtllm_ifb_loadgen.yaml`.
- If disaggregated nv-sflow is supported, all ctx and gen TRT-LLM YAML pairs are
  present and use `scaleout/sflow/templates/trtllm_disagg_loadgen.yaml`.
- GPU-count invariants are verified against the system name.

---

## Checklist: Creating Test Configs from an Existing Config

Always check `configs/` for the latest supported configurations and use the
closest current config as the template.

Use case 1: create a multi-node config with fewer GPUs than the production
config, such as `GB300x72` to `GB300x16`.

Use case 2: create a similar config between same-generation hardware, such as
`B200` to `B300` or `GB300` to `GB200`.

- Create the new folder:
  `configs/<model>/<system_with_new_gpu_count>/<framework>/<scenario>/`
- Update the GPU count in the system name, such as `x72` to `x16`.
- Copy all TRT-LLM YAML files first.
- Copy `harness.py` without modifying non-LoadGen fields.
- If the original config has `server.py`, copy it and update
  `trtllm_yml_override` and `env_yml_override` paths.
- If the original config has `<benchmark_name>_config_sflow.yaml`, copy it and
  update `SYSTEM_NAME`, the TRT-LLM YAML paths, and the topology variables.
- If the original config has `slurm_env_sflow.yaml`, copy it and update
  `GPUS_PER_NODE`, `SLURM_NODES`, account, partition, time, image, and mount
  settings as needed for the target cluster.
- Edit `harness.py` LoadGen targets only:
  `offline_expected_qps`, `server_target_qps`, and `min_query_count`.
- Use this scaling formula:
  `new_value = original * (new_gpu_count / original_gpu_count)`.
- Do not change non-LoadGen parameters unless the new hardware or serving
  topology requires it.

For the same GPU count with only a different system name, see
`docs/configs/CUSTOM_SYSTEM.md`.

---

## Additional Resources

When adding or adapting configs, refer to existing configs for examples:

- Support matrices: [Docker](../../configs/DOCKER_SUPPORT.md) and
  [SLURM](../../configs/SLURM_SUPPORT.md)
- Environment setup: [ENV_SETUP.md](../ENV_SETUP.md)
- Example configs: `configs/<benchmark>/<system>/<framework>/<scenario>/`
- Latest official submissions:
  [MLPerf Inference v6.0 results](https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA)

---

## Rules and Restrictions

- Do not describe new multi-node configs in terms of `run_scaleout.sh` or
  `run_scaleout_disagg.py`; use nv-sflow files and templates.
- Every main TRT-LLM YAML must have a paired `-env.yaml`.
- nv-sflow TRT-LLM YAML paths should use the container path
  `/work/configs/<model>/<system>/...`.
- Docker `server.py` paths should use `paths.PROJECT_BASE_DIR` and point to
  files under the same scenario directory.
- GPU counts must match the system name and the nv-sflow topology variables.
- Do not inline TRT-LLM YAML content into `server.py` or
  `<benchmark_name>_config_sflow.yaml`; reference files by path.
- Test configs are for development only, not official submissions. Do not commit
  throwaway test configs to the repo.
