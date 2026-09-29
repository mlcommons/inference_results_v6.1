# GPT-OSS-120B

## Support Matrix

See [Docker support](../../../../configs/DOCKER_SUPPORT.md) and [SLURM support](../../../../configs/SLURM_SUPPORT.md) for per-system, per-scenario support details.

## Getting Started

Change directory to `closed/Inventec`.

```
cd closed/Inventec
```

### Download Dataset

The GPT-OSS-120B benchmark uses separate datasets for accuracy and performance runs.

Please refer to the reference implementation README on mlcommons/inference for instructions to download datasets and model: [https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b#model-and-dataset-download](https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b#model-and-dataset-download)

```
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d build/data/gpt-oss/v4-test \
  https://inference.mlcommons-storage.org/metadata/gpt-oss-data.uri
```

**Dataset Statistics:**

| Dataset             | Samples | Max ISL | Max OSL | Benchmarks                                             |
| ------------------- | ------- | ------- | ------- | ------------------------------------------------------ |
| Accuracy            | 4,395   | 2,871   | 32,768  | AIME (240), GPQA (990), LiveCodeBench (3,165)          |
| Performance         | 6,396   | 15,330  | 10,240  | pubmed_summarization (synthetic)                       |
| Compliance (TEST07) | 990     | 2,871   | 10,240  | GPQA subset (accuracy verification in perf mode)       |
| Compliance (TEST09) | 6,396   | 15,330  | 10,240  | Same as Performance (output token length verification) |

**Note:** `/work/build/data` is a symlink that points to `$MLPERF_SCRATCH_PATH/data`. Once you have the freshly downloaded data, copy/mv the `acc/` and `perf/` directories into `/work/build/data/gpt-oss/v4/`. The harness reads the `_acc_eval` / `_perf_eval` suffixed files directly based on `--test_mode` — no renaming or symlinking is required.

### Download Model

```
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d build/models/gpt-oss \
  https://inference.mlcommons-storage.org/metadata/gpt-oss-model.uri
```

### Compliance Data Preprocessing

For TEST07 compliance testing, you need to preprocess the GPQA compliance dataset:

```bash
# Preprocess compliance data for TEST07
python src/nv_mlpinf/benchmarks/gpt_oss_120b/preprocess_compliance_data.py \
    --input-file build/data/gpt-oss/v4/acc/acc_eval_compliance_gpqa.parquet \
    --output-dir build/data/gpt-oss/v4/compliance/test07
```

This creates (TEST07 runs in PerformanceOnly mode, so files use the `_perf_eval` suffix the loader expects).  When done, you should have the following in your `build/data/gpt-oss` folder:

```
build/data/gpt-oss/
└── v4
    ├── acc
    │   ├── acc_eval_compliance_gpqa.parquet
    │   ├── acc_eval_ref.parquet
    │   ├── calibration_unique_sampled1024.parquet
    │   ├── input_ids_padded_acc_eval.npy
    │   └── input_lens_acc_eval.npy
    ├── compliance
    │   └── test07
    │       ├── input_ids_padded_perf_eval.npy
    │       └── input_lens_perf_eval.npy
    ├── gpt-oss-data.md5
    └── perf
        ├── input_ids_padded_perf_eval.npy
        ├── input_lens_perf_eval.npy
        └── perf_eval_ref.parquet

6 directories, 11 files
```

### Initialize Accuracy/Compliance Submodules

Accuracy runs and TEST07/TEST09 compliance verification build a Python venv (`gptoss-acc-venv`) from `src/nv_mlpinf/benchmarks/gpt_oss_120b/requirements.accuracy.txt`, which installs `LiveCodeBench` and `prm800k` from local paths. Those paths are git submodules of `3rdparty/mlc-inference` and are **not initialized by default**. Run this once before any accuracy or audit run:

```bash
# From the repository root
cd closed/Inventec/3rdparty/mlc-inference
git submodule update --init --recursive \
    language/deepseek-r1/submodules/LiveCodeBench \
    language/deepseek-r1/submodules/prm800k
```

`closed/Inventec/3rdparty/mlc-inference/language/gpt-oss-120b/submodules/{LiveCodeBench,prm800k}` are symlinks to the deepseek-r1 submodule paths above, so initializing them there populates both locations. Verify with:

```bash
ls 3rdparty/mlc-inference/language/gpt-oss-120b/submodules/LiveCodeBench/pyproject.toml
```

Skipping this step will surface as a verifier failure with: `ERROR: file:///work/3rdparty/mlc-inference/language/gpt-oss-120b/submodules/LiveCodeBench ... does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.`

## Base Image

GPT-OSS-120B uses the same TensorRT-LLM + nv-mlpinf container image for both the server and harness/client tasks. Use the x86 image for B200/B300 systems and the aarch64 image for GB200/GB300 systems.


| System                                                            | Arch    | Server image                                                                                                            | Client image         |
| ----------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------- |
| B200-SXM-180GBx8, B300-SXM-270GBx8                                | x86_64  | `nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86`     | Same as server image |
| GB200-NVL72, GB300-NVL72 (single node system or full rack system) | aarch64 | `nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64` | Same as server image |


## Run the Benchmark in Single Node Using Docker

This section covers all single-node Docker work: container start, performance, accuracy, and compliance. Check [Docker support](../../../../configs/DOCKER_SUPPORT.md) for the supported single-node systems.

### Start Docker and install `nv-mlpinf`

```bash
export MLPERF_SCRATCH_PATH=/hps/franklin/mlperf_scratch
export MLPERF_IMAGE=nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
cd closed/Inventec
make attach_docker MLPERF_IMAGE=$MLPERF_IMAGE
```

The following steps are performed inside the container.  SYSTEM_NAME is either `P9000G7-B300-SXM-270GBx8` or `P5800G7-B300-SXM-270GBx8`.

```
export SYSTEM_NAME=P9000G7-B300-SXM-270GBx8
pip install -e ".[llm]"
```

### Test mode selection

The harness selects the dataset and generation config from `--test_mode`:

- `PerformanceOnly`: performance dataset, `max_output_len=10240`
- `AccuracyOnly`: accuracy dataset, `max_output_len=32768`

### Offline performance and accuracy

```bash
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Offline

nv-mlpinf run_harness --benchmarks=gpt-oss-120b --scenarios=Offline --test_mode=PerformanceOnly
nv-mlpinf run_harness --benchmarks=gpt-oss-120b --scenarios=Offline --test_mode=AccuracyOnly
```

Exit and re-enter the container to stop a running server before switching scenarios.

### Server performance and accuracy

```bash
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Server

nv-mlpinf run_harness --benchmarks=gpt-oss-120b --scenarios=Server --test_mode=PerformanceOnly
nv-mlpinf run_harness --benchmarks=gpt-oss-120b --scenarios=Server --test_mode=AccuracyOnly
```

Exit and re-enter the container to stop a running server before switching scenarios.

### Single-node compliance

GPT-OSS-120B requires `TEST07` and `TEST09`. Both must run in `PerformanceOnly` mode.

**Offline scenario:**

```bash
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Offline
make run_audit_test07 RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --test_mode=PerformanceOnly"

# Restart the server before the next audit test.
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Offline
make run_audit_test09 RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --test_mode=PerformanceOnly"
```

TEST07 has normal run-to-run variance. If it reports an accuracy score near but below the threshold, rerun it once before debugging the setup.

**Server scenario:** Similar process with `--scenarios=Server`

## Run the Benchmark in Multi Node using Slurm + Enroot

This section covers all multi-node SLURM work through `nv-sflow`: image access, performance, accuracy, and compliance. Check [SLURM support](../../../../configs/SLURM_SUPPORT.md) before choosing a system/scenario.

### Prepare the image for multi node with Slurm + Enroot

Do not build a `.sqsh` manually for the nv-sflow path. Each run pulls the container image from `variables.CONTAINER_IMAGE` in the selected `slurm_env_sflow.yaml`. For example, [configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml](../../../../configs/gpt_oss_120b/GB200-NVL72_GB200-186GB_aarch64x72/TRTLLM/Offline/slurm_env_sflow.yaml) points to:

```yaml
variables:
  CONTAINER_IMAGE:
    value: "nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64"
```

Confirm your SLURM compute nodes can pull that image through Pyxis/Enroot before launching a full run.

### Common nv-sflow settings

Run from `closed/Inventec`. Replace `ACCT`, `PARTITION`, and `SYSTEM` for your cluster and target system.

```bash
export ACCT=root
export PARTITION=defq
export SYSTEM=B300-SXM-270GBx8
export NODES=1
export SFLOW_DUMP_DIR=build/sbatch_scripts_sflow
mkdir -p "$SFLOW_DUMP_DIR"
```

### Offline performance and accuracy

```bash
export TEST_MODE=PerformanceOnly  # use AccuracyOnly for accuracy
export RUN=gpt_oss_120b_${SYSTEM}_offline_${TEST_MODE}_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/gpt_oss_120b/${SYSTEM}/TRTLLM/Offline/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/${SYSTEM}/TRTLLM/Offline/slurm_env_sflow.yaml \
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
export TEST_MODE=PerformanceOnly  # use AccuracyOnly for accuracy
export RUN=gpt_oss_120b_${SYSTEM}_server_${TEST_MODE}_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/gpt_oss_120b/${SYSTEM}/TRTLLM/Server/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/${SYSTEM}/TRTLLM/Server/slurm_env_sflow.yaml \
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
export TEST_MODE=PerformanceOnly  # use AccuracyOnly for accuracy
export RUN=gpt_oss_120b_${SYSTEM}_interactive_${TEST_MODE}_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/gpt_oss_120b/${SYSTEM}/TRTLLM/Interactive/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/${SYSTEM}/TRTLLM/Interactive/slurm_env_sflow.yaml \
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

GPT-OSS-120B requires `TEST07` and `TEST09`; both must run in `PerformanceOnly` mode. Set `SCENARIO=Offline`, `Server`, or `Interactive`; use `trtllm_ifb_loadgen.yaml` for Offline/Server and `trtllm_disagg_loadgen.yaml` for Interactive.

TEST07:

```bash
export SCENARIO=Offline
export TEMPLATE=trtllm_ifb_loadgen.yaml
# For Interactive, use:
# SCENARIO=Interactive
# TEMPLATE=trtllm_disagg_loadgen.yaml
export RUN=gpt_oss_120b_${SYSTEM}_${SCENARIO}_TEST07_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/gpt_oss_120b/${SYSTEM}/TRTLLM/${SCENARIO}/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/${SYSTEM}/TRTLLM/${SCENARIO}/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/$TEMPLATE \
  --set WORK_DIR=$PWD \
  --set TEST_MODE=PerformanceOnly \
  --set HARNESS_EXTRA_ARGS="--audit_test=TEST07 --server_target_qps_adj_factor=0.92" \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
```

TEST09:

```bash
export SCENARIO=Offline
export TEMPLATE=trtllm_ifb_loadgen.yaml
# For Interactive, use:
# SCENARIO=Interactive
# TEMPLATE=trtllm_disagg_loadgen.yaml
export RUN=gpt_oss_120b_${SYSTEM}_${SCENARIO}_TEST09_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/gpt_oss_120b/${SYSTEM}/TRTLLM/${SCENARIO}/gptoss_config_sflow.yaml \
  -f configs/gpt_oss_120b/${SYSTEM}/TRTLLM/${SCENARIO}/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/$TEMPLATE \
  --set WORK_DIR=$PWD \
  --set TEST_MODE=PerformanceOnly \
  --set HARNESS_EXTRA_ARGS="--audit_test=TEST09 --server_target_qps_adj_factor=0.92" \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
```

TEST07 has normal run-to-run variance. If it reports an accuracy score near but below the threshold, rerun it once before debugging the setup.

## Benchmark Passing Criteria

A run is valid only if the required accuracy and performance criteria pass. For
Server and Interactive scenarios, both TTFT and TPOT latency must be within the
threshold; reduce target QPS and rerun if either latency metric fails.


| Scenario    | Accuracy criteria                       | Performance criteria       |
| ----------- | --------------------------------------- | -------------------------- |
| Offline     | `exact_match >= 83.13 * 0.99 = 82.2987` | No TTFT/TPOT constraint    |
| Server      | `exact_match >= 83.13 * 0.99 = 82.2987` | TTFT <= 3 s; TPOT <= 80 ms |
| Interactive | `exact_match >= 83.13 * 0.99 = 82.2987` | TTFT <= 2 s; TPOT <= 15 ms |
