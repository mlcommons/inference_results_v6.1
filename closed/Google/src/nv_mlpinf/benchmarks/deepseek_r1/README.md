# DeepSeek-R1

## Support Matrix

See [Docker support](../../../../configs/DOCKER_SUPPORT.md) and [SLURM support](../../../../configs/SLURM_SUPPORT.md) for per-system, per-scenario support details.

## Getting Started

### Initialize the MLCommons inference repository

The performance harness imports MLCommons submission-checker constants from
the pinned `3rdparty/mlc-inference` submodule at startup. Initialize it before
submitting a Slurm workflow:

```bash
git submodule update --init 3rdparty/mlc-inference
test -f 3rdparty/mlc-inference/tools/submission/submission_checker/constants.py
```

If the checkout uses GitHub SSH but the controller has no GitHub SSH key, use
the public HTTPS URL without changing `.gitmodules`:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init 3rdparty/mlc-inference
```

AccuracyOnly and TEST06 also require the DeepSeek evaluation submodules:

```bash
git -C 3rdparty/mlc-inference submodule update --init \
  language/deepseek-r1/submodules/LiveCodeBench \
  language/deepseek-r1/submodules/prm800k
```

The DeepSeek accuracy checker creates PRM800k's editable-install metadata from
a private temporary setup directory. It removes PRM800k's setup-time
`import numpy` only in that directory and rewrites a temporary requirements
file to point to it; the evaluator still reads grading code from the pinned
submodule. Accuracy jobs therefore do not modify the bind-mounted checkout,
including when multiple jobs run or environment creation fails.

The IFB sflow template checks `constants.py` before launching model servers so
an incomplete checkout fails before consuming the full GPU allocation. The
canonical dataset/evaluation instructions are in
`3rdparty/mlc-inference/language/deepseek-r1/README.md`.

### Download Model

Download the quantized FP4 checkpoint:

```bash
export CHECKPOINT_PATH=build/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4
git lfs install

git clone https://huggingface.co/centml/DeepSeek-R1-NVFP4-v2-mlpinf ${CHECKPOINT_PATH}

```

#### Checkpoint provenance & reproduction

The FP4 checkpoint is `deepseek-ai/DeepSeek-R1` quantized to NVFP4 with
[NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
using its official `examples/deepseek` recipe: reshard to model-parallel-8,
a single calibration sweep recording per-tensor activation `amax`
(`NVFP4_DEFAULT_CFG`), then a one-shot FP8 → NVFP4 weight conversion. Scheme:
MLP/MoE expert linears + `attn.wo` in NVFP4 (group-16 block scaling), MLA
projections in bf16, KV cache in FP8.

This checkpoint was built with the **MLPerf deepseek-r1 calibration dataset**
(500 samples). Exact Model-Optimizer/DeepSeek-V3 commit pins, the calibration
override, and a runnable end-to-end script (`quantize_dsr1_mlperf_calib.py`) are
documented in
[`scripts/dsr1_nvfp4_ptq/`](../../../../scripts/dsr1_nvfp4_ptq/README.md).
It clears the closed-division accuracy gate (exact_match 80.86 ≥ 80.5446).

### Download and Prepare Data

Download dataset files following the MLCommons README:

```bash
# Follow: 3rdparty/mlc-inference/language/deepseek-r1/README.md
# - mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl
# - mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl

# Move files to expected locations
mv mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl \
    build/data/deepseek-r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl
mv mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl \
    build/data/deepseek-r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl

# Run preprocessing (inside container)
make attach_docker MLPERF_IMAGE=nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
pip install -e ".[llm]"
BENCHMARKS=deepseek-r1 make preprocess_data

# Note: If preprocessing fails, you may need numpy==2.3.0:
# pip install virtualenv
# virtualenv .venv --system-site-packages
# source .venv/bin/activate
# pip install numpy==2.3.0 torch==2.7.0
# BENCHMARKS=deepseek-r1 make preprocess_data
```

**Verify the following files exist:**

1. Model: `build/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4/`
2. Preprocessed data at `build/preprocessed_data/deepseek-r1/`:
  - `input_lens.npy`
  - `input_ids_padded.npy`
  - `mlperf_deepseek_r1_calibration_dataset_500_fp8_calibration/data.parquet`

### Validated a4x Slurm preprocessing

On the a4x GB200 domain, keep the shared model and raw datasets immutable and
write preprocessing results to user-owned Lustre storage. The checked-in
`build/dp_r1_preprocess.sbatch` script implements the validated path:

```bash
sbatch build/dp_r1_preprocess.sbatch
```

The job uses one GPU because importing `nv_mlpinf` performs CUDA system
detection before preprocessing starts. It invokes `preprocess_data.py` directly
to avoid loading unrelated TensorRT-LLM benchmark symbols, and creates a
temporary system-site-packages virtual environment with `numpy==2.3.0` so the
NumPy 2.x pickles can be read. The shared input is mounted read-only; outputs are
written to:

```text
/lustre/alisachen/mlperf_inference_storage/preprocessed_data/deepseek-r1/
├── input_lens.npy
├── input_ids_padded.npy
└── mlperf_deepseek_r1_calibration_dataset_500_fp8_calibration/data.parquet
```

The current preprocessor initially writes the calibration parquet below a
directory ending in `_eval.pkl`; the batch script also publishes the
README-required `_calibration/data.parquet` path.

## Base Image

DeepSeek-R1 uses the same TensorRT-LLM + nv-mlpinf container image for both the server and harness/client tasks. Use the x86 image for B200/B300 systems and the aarch64 image for GB200/GB300 systems.

| System | Arch | Server image | Client image |
| ------ | ---- | ------------ | ------------ |
| B200-SXM-180GBx8, B300-SXM-270GBx8 | x86_64 | `nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86` | Same as server image |
| GB200-NVL72, GB300-NVL72 (single node system or full rack system) | aarch64 | `nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64` | Same as server image |

## Run the Benchmark in Single Node Using Docker

This section covers all single-node Docker work: image pull, container start, performance, accuracy, and compliance. Check [Docker support](../../../../configs/DOCKER_SUPPORT.md) for the supported single-node systems.

### Prepare the image for single node with Docker

```bash
docker pull nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
```

### Start Docker and install `nv-mlpinf`

```bash
cd closed/NVIDIA
make attach_docker MLPERF_IMAGE=nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
pip install -e ".[llm]"
```

### Offline performance and accuracy

```bash
nv-mlpinf run_llm_server --benchmarks=deepseek-r1 --scenarios=Offline

nv-mlpinf run_harness --benchmarks=deepseek-r1 --scenarios=Offline --test_mode=PerformanceOnly
nv-mlpinf run_harness --benchmarks=deepseek-r1 --scenarios=Offline --test_mode=AccuracyOnly
```

### Server performance and accuracy

```bash
nv-mlpinf run_llm_server --benchmarks=deepseek-r1 --scenarios=Server

nv-mlpinf run_harness --benchmarks=deepseek-r1 --scenarios=Server --test_mode=PerformanceOnly
nv-mlpinf run_harness --benchmarks=deepseek-r1 --scenarios=Server --test_mode=AccuracyOnly
```

Exit and re-enter the container to stop a running server before switching scenarios.

### Single-node compliance

DeepSeek-R1 uses `TEST06`. Run compliance in `PerformanceOnly` mode. Set `SCENARIO=Offline` or `SCENARIO=Server`.

```bash
SCENARIO=Offline
nv-mlpinf run_llm_server --benchmarks=deepseek-r1 --scenarios=$SCENARIO
make run_audit_test06 RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=$SCENARIO --test_mode=PerformanceOnly"
```

## Run the Benchmark in Multi Node using Slurm + Enroot

This section covers all multi-node SLURM work through `nv-sflow`: image access, performance, accuracy, and compliance. Check [SLURM support](../../../../configs/SLURM_SUPPORT.md) before choosing a system/scenario.

### Prepare the image for multi node with Slurm + Enroot

Use the container source selected by `variables.CONTAINER_IMAGE` in the chosen
`slurm_env_sflow.yaml`. On the a4x domain, import the pinned image once as a
shared `.sqsh` because its nv-sflow/Pyxis combination cannot express the same
NGC registry URI consistently. For example,
[configs/deepseek_r1/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml](../../../../configs/deepseek_r1/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml)
points to:

```yaml
variables:
  CONTAINER_IMAGE:
    value: "/lustre/alisachen/containers/mlperf-inference_tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64.sqsh"
```

Confirm every Slurm compute node can read and launch the selected image through
Pyxis/Enroot before launching a full run.

### Validate the Slurm MPI mode

Run a two-node container smoke test before generating a benchmark job. On the
a4x domain, direct PMIx launch fails because Slurm provides PMIx 5.0.3 while the
pinned container's Open MPI 4.1.9a1 embeds PMIx 3.2.5. PMI-2 is supported by
both sides and has been validated with container-side `mpi4py` ranks 0 and 1:

```bash
sbatch --export=ALL,MPI_MODE=pmi2 build/dp_r1_mpi_smoke.sbatch
```

For this domain, set `mpi: pmi2` in the selected `slurm_env_sflow.yaml`.
The a4x Pyxis installation requires `nvcr.io#nvidia/...` registry syntax for a
direct `--container-image` argument. nv-sflow 0.2.1 rejects that registry syntax
during validation, while the slash-only `nvcr.io/nvidia/...` form is
misinterpreted as a Docker Hub repository. Import the pinned image once to the
shared `.sqsh` path with `build/dp_r1_import_container.sbatch`, then use that
file for nv-sflow runs.

### Validate the MNNVL fabric

Each `dep8` server replica spans two a4x nodes and four GPUs per node. On a
healthy GB200 NVL72 fabric, TensorRT-LLM should create `MnnvlMemory` and select
the `NVLinkOneSided` communication strategy. Run the dedicated fabric/clique
check when no performance measurement is active:

```bash
sbatch scripts/slurm_llm/deepseek_r1/mnnvl_fabric_smoke.sbatch
```

The smoke test assigns one GPU to each of eight PMI-2 ranks and requires every
GPU to report a distinct UUID plus fabric `State: Completed` and `Status:
Success`, with four distinct GPUs per node. When the driver exposes MNNVL
cluster UUID and clique ID, the checker also requires them to be consistent
across ranks. A real server run provides the data-path confirmation;
its rank logs must contain both of these markers and no transport error:

```text
[MnnvlMemory] creating address
Selected communication strategy: NVLinkOneSided
```

`NixlTransferAgent ... using NIXL backend: UCX` is an additional transfer-path
marker. It does not by itself prove that UCX selected RDMA rather than sockets.
Do not run a transport smoke test concurrently with PerformanceOnly because it
would compete for the same fabric and invalidate the performance comparison.

### Common nv-sflow settings

Run from `closed/NVIDIA`. Replace `ACCT`, `PARTITION`, and `SYSTEM` for your cluster and target system. The examples below use GB200x72.

```bash
ACCT=<your-slurm-account>
PARTITION=gb200
SYSTEM=GB200-NVL72_GB200-186GB_aarch64x72
NODES=18
SFLOW_VERSION=6efbf59c4a835473b08362eb623358e3ace31d6e
SFLOW_DUMP_DIR=build/sbatch_scripts_sflow
mkdir -p "$SFLOW_DUMP_DIR"
```

nv-sflow 0.2.1 performs post-run artifact copies without preserving the return
code from `sflow run`. Each example below patches the generated wrapper and
validates it before submission. This keeps a failed workflow failed at the
parent Slurm-job level while still allowing nv-sflow to copy diagnostics.

### Offline performance and accuracy

```bash
TEST_MODE=PerformanceOnly  # use AccuracyOnly for accuracy
RUN=deepseek_r1_${SYSTEM}_offline_${TEST_MODE}_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/deepseek_r1/${SYSTEM}/TRTLLM/Offline/deepseek_config_sflow.yaml \
  -f configs/deepseek_r1/${SYSTEM}/TRTLLM/Offline/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set TEST_MODE=$TEST_MODE \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  --sflow-version=$SFLOW_VERSION \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh

python3 scripts/slurm_llm/deepseek_r1/fix_sflow_batch_exit.py \
  "${SFLOW_DUMP_DIR}/$RUN.sh"
bash -n "${SFLOW_DUMP_DIR}/$RUN.sh"
```

### Server performance and accuracy

```bash
TEST_MODE=PerformanceOnly  # use AccuracyOnly for accuracy
RUN=deepseek_r1_${SYSTEM}_server_${TEST_MODE}_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/deepseek_r1/${SYSTEM}/TRTLLM/Server/deepseek_config_sflow.yaml \
  -f configs/deepseek_r1/${SYSTEM}/TRTLLM/Server/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set TEST_MODE=$TEST_MODE \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  --sflow-version=$SFLOW_VERSION \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh

python3 scripts/slurm_llm/deepseek_r1/fix_sflow_batch_exit.py \
  "${SFLOW_DUMP_DIR}/$RUN.sh"
bash -n "${SFLOW_DUMP_DIR}/$RUN.sh"
```

### Interactive performance and accuracy

Interactive uses disaggregated serving with the `trtllm_disagg_loadgen.yaml` disagg loadgen template.

```bash
TEST_MODE=PerformanceOnly  # use AccuracyOnly for accuracy
RUN=deepseek_r1_${SYSTEM}_interactive_${TEST_MODE}_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/deepseek_r1/${SYSTEM}/TRTLLM/Interactive/deepseek_config_sflow.yaml \
  -f configs/deepseek_r1/${SYSTEM}/TRTLLM/Interactive/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_disagg_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set TEST_MODE=$TEST_MODE \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  --sflow-version=$SFLOW_VERSION \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh

python3 scripts/slurm_llm/deepseek_r1/fix_sflow_batch_exit.py \
  "${SFLOW_DUMP_DIR}/$RUN.sh"
bash -n "${SFLOW_DUMP_DIR}/$RUN.sh"
```

After LoadGen finishes, the endpoint harness signals each HTTP worker, waits up
to 10 seconds for graceful completion, closes both ZMQ sockets before
terminating their context, and force-reaps only surviving processes. The
health check is synchronous, so the harness does not carry a module-global
uvloop daemon thread across worker forks or into interpreter shutdown. Treat a
nonzero harness or parent Slurm exit as a failed attempt even when result files
and accuracy metrics were written before shutdown.

### Multi-node compliance

DeepSeek-R1 uses `TEST06`. Run compliance in `PerformanceOnly` mode by passing the audit test through `HARNESS_EXTRA_ARGS`. Set `SCENARIO=Offline`, `Server`, or `Interactive`; use `trtllm_ifb_loadgen.yaml` for Offline/Server and `trtllm_disagg_loadgen.yaml` for Interactive.

```bash
SCENARIO=Offline
TEMPLATE=trtllm_ifb_loadgen.yaml
# For Interactive, use:
# SCENARIO=Interactive
# TEMPLATE=trtllm_disagg_loadgen.yaml
RUN=deepseek_r1_${SYSTEM}_${SCENARIO}_TEST06_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/deepseek_r1/${SYSTEM}/TRTLLM/${SCENARIO}/deepseek_config_sflow.yaml \
  -f configs/deepseek_r1/${SYSTEM}/TRTLLM/${SCENARIO}/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/$TEMPLATE \
  --set WORK_DIR=$PWD \
  --set TEST_MODE=PerformanceOnly \
  --set HARNESS_EXTRA_ARGS="--audit_test=TEST06 --server_target_qps_adj_factor=0.92" \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  --sflow-version=$SFLOW_VERSION \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh

python3 scripts/slurm_llm/deepseek_r1/fix_sflow_batch_exit.py \
  "${SFLOW_DUMP_DIR}/$RUN.sh"
bash -n "${SFLOW_DUMP_DIR}/$RUN.sh"
```

The commands above generate sbatch scripts for review and intentionally do not
submit them. After checking the generated script, launch it explicitly with
`sbatch <generated-script>`.

For TEST06 verification, `First token check pass: Skipped` is acceptable only
for Offline. Server and Interactive must report `First token check pass: True`;
all scenarios must also pass the EOS and sample-length checks.

## Benchmark Passing Criteria

A run is valid only if the required accuracy and performance criteria pass. For
Server and Interactive scenarios, both TTFT and TPOT latency must be within the
threshold; reduce target QPS and rerun if either latency metric fails.

| Scenario | Accuracy criteria | Performance criteria |
| --- | --- | --- |
| Offline | `exact_match >= 80.544618`; `3497.60466 <= TOKENS_PER_SAMPLE <= 4274.85014` | LoadGen `VALID`; no TTFT/TPOT constraint |
| Server | `exact_match >= 80.544618`; `3497.60466 <= TOKENS_PER_SAMPLE <= 4274.85014` | LoadGen `VALID`; 99p TTFT < 2 s; 99p TPOT < 80 ms |
| Interactive | `exact_match >= 80.544618`; `3497.60466 <= TOKENS_PER_SAMPLE <= 4274.85014` | LoadGen `VALID`; 99p TTFT < 1.5 s; 99p TPOT < 15 ms |

For a closed v6.1 datacenter submission, Offline is mandatory and the pinned
submission checker also requires at least one of Server or Interactive. Each
submitted scenario needs performance, accuracy, and TEST06 artifacts. Running
both latency scenarios is optional.

For DeepSeek-R1 Offline, the official performance score is LoadGen's final
`result_tokens_per_second`. `offline_expected_qps` is a workload-sizing target,
not an MLPerf validity threshold. PerformanceOnly does not evaluate accuracy or
TEST06; those require separate runs.
