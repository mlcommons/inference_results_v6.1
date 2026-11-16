# Environment Setup — Agent Guidelines

This document is the source of truth for AI agents setting up the MLPerf Inference benchmark environment.

---

## Overview

MLPerf Inference supports a wide range of deep learning workloads (LLMs, Diffusion Models, RecSys, etc.) across two launching modes:


| Mode               | When to Use                                                                                                                                                       |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Single-node Docker | Benchmarking on **1 node only** with an interactive bash shell. The node must have Docker installed and be able to launch containers. Example: B200 on computelab |
| Multi-node SLURM   | Benchmarking with **multiple nodes** in a SLURM cluster environment. Commands are launched from a login node using `sbatch` / `srun`. Example: GB200 NVL72        |


---

## Single-node Docker Environment Setup

### 1. Set Up Your Scratch Directory

Choose the host path that stores your model weights, datasets, and
preprocessed data, then export it before launching Docker:

```bash
export MLPERF_SCRATCH_PATH=/path/to/mlperf_inference_storage
mkdir -p "$MLPERF_SCRATCH_PATH"/{data,models,preprocessed_data}
```

Benchmark code expects the scratch layout to contain `data/`, `models/`, and
`preprocessed_data/`. The Docker launch passes `MLPERF_SCRATCH_PATH` into the
container so single-node recipes can find weights and datasets consistently.

### 2. Pull Base Image

Each benchmark requires a specific base image. Check the benchmark's README for the correct image.

**Example (LLM benchmarks):**

```bash
docker pull nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
```

### 3. Attach to Container

Navigate to the working directory `closed/NVIDIA` and launch a container:

```bash
cd closed/NVIDIA

# Option A: Interactive mode (for manual work)
make attach_docker MLPERF_IMAGE=<image>

# Option B: Detached mode (for agent automation)
make attach_docker_detached MLPERF_IMAGE=<image>
```

**Notes:**

- **Interactive mode**: Launches an interactive bash shell inside the container. Use this for hands-on debugging and exploration.
- **Detached mode**: Launches the container in the background. Use `docker exec` to run commands. Recommended for agent automation.

### 4. Install nv-mlpinf Package in Benchmark Specific Containers

Inside the container, install the `nv-mlpinf` package inside `closed/NVIDIA`.

**For detailed installation instructions for each benchmark**, see [src/README.md](src/README.md).

---

## Multi-node SLURM Environment Setup

Use `nv-sflow` for SLURM launches. These runs are submitted from the SLURM login node and executed through Pyxis/Enroot on the allocated compute nodes.

### 1. Set Up Your Scratch Directory

For each nv-sflow run config, edit the colocated `slurm_env_sflow.yaml` so
`variables.SCRATCH_DIR` points to the host path where your model weights,
datasets, and preprocessed data live. nv-sflow mounts that host path at
`/home/mlperf_inference_storage` inside the Enroot container.

For example:

```yaml
variables:
  SCRATCH_DIR:
    value: /path/to/mlperf_inference_storage
  CONTAINER_MOUNTS:
    value: "${{ variables.WORK_DIR }}:/work,${{ variables.SCRATCH_DIR }}:/home/mlperf_inference_storage"
```

Make this change in every `slurm_env_sflow.yaml` you plan to use, such as
[configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Interactive/slurm_env_sflow.yaml](../configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Interactive/slurm_env_sflow.yaml).
The benchmark configs and harnesses then refer to the stable in-container paths
under `/home/mlperf_inference_storage`.

### 2. Confirm Published MLPerf Image Access

The nv-sflow configs pull benchmark containers from the published MLPerf images on NGC. Before launching a benchmark, confirm that your user and SLURM compute nodes can pull from `nvcr.io/nvidia/mlperf/mlperf-inference`.

If your cluster requires registry credentials for Enroot/Pyxis, configure those credentials before running `sflow`. A missing registry login usually fails before benchmark code starts, during the container import or first `srun` task.

### 3. Confirm SLURM Support for the Benchmark

Check [configs/SLURM_SUPPORT.md](../configs/SLURM_SUPPORT.md) and choose an exact benchmark, system, and scenario that is supported for SLURM. Use rows marked `✅` as the current supported path; rows marked pending or WIP require additional owner confirmation before treating them as supported.

### 4. Inspect the Container Image in the nv-sflow Config

Each nv-sflow run loads a benchmark config together with its colocated `slurm_env_sflow.yaml`. The container image is pulled directly from `variables.CONTAINER_IMAGE` in that environment file.

For example, the DeepSeek-R1 B300 Offline SLURM config pins the image in [configs/deepseek_r1/B300-SXM-270GBx8/TRTLLM/Offline/slurm_env_sflow.yaml](../configs/deepseek_r1/B300-SXM-270GBx8/TRTLLM/Offline/slurm_env_sflow.yaml):

```yaml
variables:
  CONTAINER_IMAGE:
    value: "nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86"
```

Confirm that your SLURM cluster can pull the exact image referenced by the config you plan to run. B200/B300 systems use the x86 image, while GB200/GB300 systems use the aarch64 image. Do this before a full benchmark launch, because nv-sflow uses the config value directly unless you override `CONTAINER_IMAGE` at launch time.

### 5. Set Cluster-Specific SLURM Values

Update or override the SLURM variables to match your cluster naming convention:

- `SLURM_ACCOUNT`: your charge account or allocation name
- `SLURM_PARTITION`: the target partition or queue
- `SLURM_TIME`: the requested walltime, if the default is not appropriate

You can edit the relevant `slurm_env_sflow.yaml`, or override values when launching:

```bash
--set SLURM_ACCOUNT=<your-account> \
--set SLURM_PARTITION=<your-partition> \
--set SLURM_TIME=<hh:mm:ss>
```

When using `sflow batch`, also pass the scheduler directives expected by your cluster, such as `--account`, `--partition`, `--nodes`, and `--time`, as shown in the nv-sflow guide.

### 6. Launch with nv-sflow

Follow [scaleout/sflow/README.md](../scaleout/sflow/README.md) for installation and launch details. A run normally combines three files:

1. The benchmark/scenario config, such as `configs/deepseek_r1/B300-SXM-270GBx8/TRTLLM/Offline/deepseek_config_sflow.yaml`
2. The colocated SLURM environment config, such as `configs/deepseek_r1/B300-SXM-270GBx8/TRTLLM/Offline/slurm_env_sflow.yaml`
3. The matching nv-sflow template under `scaleout/sflow/templates/`

Launch the selected config with `sflow run` for interactive debugging or `sflow batch --submit` for a queued SLURM job. Always run these commands from `closed/NVIDIA` so relative config paths and `/work` mounts resolve correctly.

### FAQ: `MPI_Init_thread` fails with PMIx or SLURM PMI support errors

If a multi-node SLURM run fails during MPI initialization with an error like:

```text
PMIX ERROR: ERROR in file gds_ds12_lock_pthread.c
OPAL ERROR: Unreachable in file pmix3x_client.c
The application appears to have been direct launched using "srun",
but OMPI was not built with SLURM's PMI support and therefore cannot execute.
*** An error occurred in MPI_Init_thread
```

the issue is usually in the SLURM PMIx/Open MPI bootstrap path, before
TRT-LLM model execution begins. A common cause is that the cluster PMIx/Open MPI
stack is not compatible with the NVIDIA-validated runtime stack.

Run the PMIx smoke test from `closed/NVIDIA`:

```bash
sbatch scaleout/sflow/tools/sbatch_pmix_tensorrtllm_runtime_test.sbatch
```

Then inspect the generated `pmix_tensorrtllm_<jobid>.out` file:

```bash
grep -E 'PMIX_VERSION|mpi4py rank|PASS' pmix_tensorrtllm_<jobid>.out
```

In the NVIDIA reference environment, the host-side PMIx test reports
`PMIX_VERSION=5.0.1a1`, and the container-side `mpi4py` `COMM_WORLD` smoke test
prints one rank line per task followed by `PASS`. If `PMIX_VERSION` differs or
the `mpi4py` test fails, align the cluster SLURM PMIx/Open MPI integration with
the validated runtime environment, or rebuild the container MPI/`mpi4py` stack
with compatible SLURM PMI/PMIx support.

## Benchmark-Specific Setup Instructions

**For both Docker and SLURM workflows**, each benchmark has unique requirements and up-to-date setup instructions. Always refer to the benchmark's README for the most current information on:

- Base image requirements
- Package installation commands (with specific flags like `--no-build-isolation` for WAN2.2)
- Data preprocessing steps
- Model downloads
- nv-sflow config selection, container image access, and SLURM account/partition overrides
- Benchmark-specific configuration options

**Benchmark READMEs:**

- DeepSeek-R1: [src/nv_mlpinf/benchmarks/deepseek_r1/README.md](../src/nv_mlpinf/benchmarks/deepseek_r1/README.md)
- GPT-OSS-120B: [src/nv_mlpinf/benchmarks/gpt_oss_120b/README.md](../src/nv_mlpinf/benchmarks/gpt_oss_120b/README.md)
- Llama2-70B: [src/nv_mlpinf/benchmarks/llama2_70b/README.md](../src/nv_mlpinf/benchmarks/llama2_70b/README.md)
- Whisper: [src/nv_mlpinf/benchmarks/whisper/README.md](../src/nv_mlpinf/benchmarks/whisper/README.md)
- Q3VL: [src/nv_mlpinf/benchmarks/q3vl/vllm/README.md](../src/nv_mlpinf/benchmarks/q3vl/vllm/README.md)
- Wan2.2-T2V-A14B: [src/nv_mlpinf/benchmarks/wan22_a14b/README.md](../src/nv_mlpinf/benchmarks/wan22_a14b/README.md)

---

> **⚙️ Advanced: Building TRT-LLM from Source for LLM workloads (Optional)**
>
> If you need a custom TRT-LLM branch instead of the provided release container:
>
> ```bash
> cd closed/NVIDIA/3rdparty/trtllm
> make -C docker release_build CUDA_ARCHS="90-real;100-real;120-real"
> ```
>
> This produces `tensorrt_llm/release:latest` which can be used as the LLM base image.

---

## Additional Resources

- **Configuration Editing**: [docs/configs/EDITING.md](configs/EDITING.md)
- **nv-sflow Guide**: [scaleout/sflow/README.md](../scaleout/sflow/README.md)
- **Docker Support Matrix**: [configs/DOCKER_SUPPORT.md](../configs/DOCKER_SUPPORT.md)
- **SLURM Support Matrix**: [configs/SLURM_SUPPORT.md](../configs/SLURM_SUPPORT.md)
- **Agent Guidelines**: [AGENTS.md](../AGENTS.md) (repository root)

## Rules and Restrictions

- Docker: Never run Docker as root
- SLURM: Use `sbatch` scripts (not `srun` directly), write the scipt under `closed/NVIDIA`
- SLURM: prompt user before launching any sbatch job
- SLURM: When monitoring jobs: avoid `watch`, don't poll `squeue` frequently
- SLURM: Ask user for benchmark duration (don't default to 4 hours)
- SLURM: use the container image pinned by `variables.CONTAINER_IMAGE` unless an owner-provided override is required
- SLURM: always run `sflow run` or `sflow batch` from the `closed/NVIDIA` directory on the cluster
