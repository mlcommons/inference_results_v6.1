# Cisco B300 configurations

This directory contains the selected TensorRT-LLM Serve configurations for the
`Cisco_UCS_B300x16_G200_4x2_RO` and `Cisco_UCS_B300x64_G200_4x2_RO` systems.
Each profile README defines its topology, target rate, and required audit.

Set the common runtime variables:

```bash
export CISCO_SUBMISSION_ROOT=/path/to/submission/closed/Cisco
export CISCO_RUN_ROOT="${CISCO_SUBMISSION_ROOT}"
export MLPERF_CISCO_ROOT="${CISCO_SUBMISSION_ROOT}"
export MLPERF_DATA_DIR=/path/to/mlperf-data
export MLPERF_CONTAINER_IMAGE=/path/to/mlperf-inference-v6.1-loadgen-trtllm-x86.sqsh
export MLPERF_CONTAINER_SHA256=5112cac5674c92ac1e6e7db69d0fc9e8882bd5211d1d97636229facf153f30dc
export MLPERF_SFLOW_BIN=/path/to/sflow
export MLPERF_SLURM_ACCOUNT=your-account
export MLPERF_SLURM_PARTITION=your-partition
export MLPERF_OUTPUT_DIR=/path/to/output
```

The collected squashfs came from the immutable OCI manifest
`nvcr.io/nvidia/mlperf/mlperf-inference@sha256:54c5995ded61b0b5640c641ca79248b14422d0720ef8b293aba8f5fab2e5243d`.
Before creating the environment or launching a run, verify the exact local image:

```bash
printf '%s  %s\n' "${MLPERF_CONTAINER_SHA256}" "${MLPERF_CONTAINER_IMAGE}" |
  sha256sum --check --strict -
```

Provide routable endpoint addresses in replica order. IFB profiles use one
comma-separated list; PD profiles use separate context, generation, and
frontend lists. Addresses may repeat when multiple replicas share a host.

```bash
# IFB:
export MLPERF_SERVER_HOSTS=host01,host01,host02,host02

# PD:
export MLPERF_CTX_HOSTS=host01,host01
export MLPERF_GEN_HOSTS=host02,host02
export MLPERF_FRONTEND_HOSTS=host03

# Optional; leave empty to use the operating system routing table.
export MLPERF_HTTP_INTERFACE=

# Optional; leave empty to let Gloo and NCCL auto-select their bootstrap route.
export MLPERF_BOOTSTRAP_INTERFACE=
```

Inside that image, install the shared harness from `closed/Cisco`. The runtime
constraints match the collected image and preserved harness environment:

```bash
test "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "3.12"
python3 -m venv --system-site-packages "${CISCO_RUN_ROOT}/.venv"
source "${CISCO_RUN_ROOT}/.venv/bin/activate"
python -m pip install "pip==26.1.2"
python -m pip install \
  --constraint "${CISCO_RUN_ROOT}/runtime-constraints.txt" \
  --no-build-isolation \
  --editable "${CISCO_RUN_ROOT}[llm]"
python -m pip check
python -c 'import mlperf_loadgen, nv_mlpinf, nvmitten; print("harness imports OK")'
```

For an x16 profile, export the `PROFILE`, `MAIN_CONFIG`, and `WORKFLOW` values
shown in its README, then run directly from the checked-in profile:

```bash
export CONFIG_DIR="${PROFILE}"
mkdir -p "${MLPERF_OUTPUT_DIR}"

SFLOW_ENDPOINT_ARGS=(--set "HTTP_INTERFACE=${MLPERF_HTTP_INTERFACE:-}")
case "${WORKFLOW}" in
  trtllm_ifb_portable.yaml)
    : "${MLPERF_SERVER_HOSTS:?Set one comma-separated host per IFB replica}"
    SFLOW_ENDPOINT_ARGS+=(--set "SERVER_HOSTS=${MLPERF_SERVER_HOSTS}")
    ;;
  trtllm_disagg_portable.yaml)
    : "${MLPERF_CTX_HOSTS:?Set one comma-separated host per context replica}"
    : "${MLPERF_GEN_HOSTS:?Set one comma-separated host per generation replica}"
    : "${MLPERF_FRONTEND_HOSTS:?Set one comma-separated host per frontend replica}"
    SFLOW_ENDPOINT_ARGS+=(
      --set "CTX_HOSTS=${MLPERF_CTX_HOSTS}"
      --set "GEN_HOSTS=${MLPERF_GEN_HOSTS}"
      --set "FRONTEND_HOSTS=${MLPERF_FRONTEND_HOSTS}"
    )
    ;;
esac

"${MLPERF_SFLOW_BIN}" run \
  --file "${CONFIG_DIR}/${MAIN_CONFIG}" \
  --file "${CISCO_SUBMISSION_ROOT}/src/nv_mlpinf/scaleout/templates/slurm_env_sflow.example.yaml" \
  --file "${CISCO_SUBMISSION_ROOT}/src/nv_mlpinf/scaleout/templates/${WORKFLOW}" \
  --set WORK_DIR="${CISCO_RUN_ROOT}" \
  --set SCRATCH_DIR="${MLPERF_DATA_DIR}" \
  --set CONTAINER_IMAGE="${MLPERF_CONTAINER_IMAGE}" \
  --set SLURM_ACCOUNT="${MLPERF_SLURM_ACCOUNT}" \
  --set SLURM_PARTITION="${MLPERF_SLURM_PARTITION}" \
  --set TEST_MODE="${MLPERF_TEST_MODE:-PerformanceOnly}" \
  --set LOADGEN_MODE=full \
  --set "BOOTSTRAP_INTERFACE=${MLPERF_BOOTSTRAP_INTERFACE:-}" \
  --set "HARNESS_EXTRA_ARGS=${MLPERF_HARNESS_EXTRA_ARGS:-}" \
  "${SFLOW_ENDPOINT_ARGS[@]}" \
  --workspace-dir "${MLPERF_OUTPUT_DIR}/workspace" \
  --output-dir "${MLPERF_OUTPUT_DIR}"
```

The submission configurations are consumed in place. Shared SFlow workflows and
the portable scheduler template live under `src/nv_mlpinf/scaleout/templates`.

Use `MLPERF_TEST_MODE=AccuracyOnly` for accuracy. Keep
`MLPERF_TEST_MODE=PerformanceOnly` for compliance and set the audit argument
documented by the profile. The x64 profiles provide their own portable Slurm
launchers because their layouts span eight nodes.
