# DeepSeek-R1

## Support Matrix

See [Docker support](../../../../configs/DOCKER_SUPPORT.md) and [SLURM support](../../../../configs/SLURM_SUPPORT.md) for per-system, per-scenario support details.

## Getting Started

### Download Model

Download the quantized FP4 checkpoint:

```bash
export CHECKPOINT_PATH=build/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4
git lfs install

git clone https://huggingface.co/nvidia/DeepSeek-R1-FP4-v2 ${CHECKPOINT_PATH}

```

### Download and Prepare Data

Download dataset files following the MLCommons README:

```bash
# Download files from: https://github.com/mlcommons/inference/tree/master/language/deepseek-r1/README.md
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

Do not build a `.sqsh` manually for the nv-sflow path. Each run pulls the container image from `variables.CONTAINER_IMAGE` in the selected `slurm_env_sflow.yaml`. For example, [configs/deepseek_r1/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml](../../../../configs/deepseek_r1/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml) points to:

```yaml
variables:
  CONTAINER_IMAGE:
    value: "nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64"
```

Confirm your SLURM compute nodes can pull that image through Pyxis/Enroot before launching a full run.

### Common nv-sflow settings

Run from `closed/NVIDIA`. Replace `ACCT`, `PARTITION`, and `SYSTEM` for your cluster and target system. The examples below use GB200x72.

```bash
ACCT=<your-slurm-account>
PARTITION=gb200
SYSTEM=GB200-NVL72_GB200-186GB_aarch64x72
NODES=18
SFLOW_DUMP_DIR=build/sbatch_scripts_sflow
mkdir -p "$SFLOW_DUMP_DIR"
```

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
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
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
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
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
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
```

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
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
```

## Benchmark Passing Criteria

A run is valid only if the required accuracy and performance criteria pass. For
Server and Interactive scenarios, both TTFT and TPOT latency must be within the
threshold; reduce target QPS and rerun if either latency metric fails.

| Scenario | Accuracy criteria | Performance criteria |
| --- | --- | --- |
| Offline | `exact_match >= 0.99 * 81.3582 = 80.544618`; `TOKENS_PER_SAMPLE >= 0.9 * 3886.2274 = 3497.60466` | No TTFT/TPOT constraint |
| Server | `exact_match >= 0.99 * 81.3582 = 80.544618`; `TOKENS_PER_SAMPLE >= 0.9 * 3886.2274 = 3497.60466` | TTFT <= 2 s; TPOT <= 80 ms |
| Interactive | `exact_match >= 0.99 * 81.3582 = 80.544618`; `TOKENS_PER_SAMPLE >= 0.9 * 3886.2274 = 3497.60466` | TTFT <= 1.5 s; TPOT <= 15 ms |
