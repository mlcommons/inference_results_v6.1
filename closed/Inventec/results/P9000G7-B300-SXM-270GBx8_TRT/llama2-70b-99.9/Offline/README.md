# Llama2-70B

## Support Matrix

See [Docker support](../../../../configs/DOCKER_SUPPORT.md) and [SLURM support](../../../../configs/SLURM_SUPPORT.md) for per-system, per-scenario support details.

## Getting Started

Change directory to `closed/Inventec`.

```
cd closed/Inventec
```

### Download NVFP4 Quantized Model

(Public, requires agreement) [https://huggingface.co/centml/llama2-70b-chat-hf-torch-fp4_mlperf-inf-v6.0](https://huggingface.co/centml/llama2-70b-chat-hf-torch-fp4_mlperf-inf-v6.0)

Please pull the model checkpoint from huggingface (`hf auth login` with your HF_TOKEN is required).

```
hf download https://huggingface.co/centml/llama2-70b-chat-hf-torch-fp4_mlperf-inf-v6.0 \
  --llm_quantizer_outdir= build/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-torch-fp4
```

Then make a symlink `build/models/Llama2/Llama-2-70b-chat-hf` to `fp4-quantized-modelopt/llama2-70b-chat-hf-torch-fp4`.

```
ln -sf fp4-quantized-modelopt/llama2-70b-chat-hf-torch-fp4 build/models/Llama2/Llama-2-70b-chat-hf
```
When done, you should have the following in your `build/models/Llama2` folder:

```
build/models/Llama2/
├── fp4-quantized-modelopt
│   └── llama2-70b-chat-hf-torch-fp4
│       ├── config.json
│       ├── generation_config.json
│       ├── hf_quant_config.json
│       ├── model-00001-of-00008.safetensors
│       ├── model-00002-of-00008.safetensors
│       ├── model-00003-of-00008.safetensors
│       ├── model-00004-of-00008.safetensors
│       ├── model-00005-of-00008.safetensors
│       ├── model-00006-of-00008.safetensors
│       ├── model-00007-of-00008.safetensors
│       ├── model-00008-of-00008.safetensors
│       ├── model.safetensors.index.json
│       ├── README.md
│       ├── special_tokens_map.json
│       ├── stderr.txt
│       ├── stdout.txt
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── tokenizer.model
└── Llama-2-70b-chat-hf -> fp4-quantized-modelopt/llama2-70b-chat-hf-torch-fp4
```

### Download and Prepare Data

Please download data files by following the MLCommons README with instructions.

**Note:** `/work/build/data` is a symlink that points to `$MLPERF_SCRATCH_PATH/data`

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d build/data/llama2-70b \
  https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.uri

# Unzip files
cd build/data/llama2-70b/
gzip -dk open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
gzip -dk open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz
cd -

# Run pre-process step for llama2 (inside the container)
python3 src/nv_mlpinf/benchmarks/llama2_70b/preprocess_data.py \
    --data_dir build/data/ \
    --preprocessed_data_dir build/preprocessed-data

# Make one more symlink (outside the container)
ln -sf ../data/llama2-70b build/preprocessed_data/open_orca
```

**Verify the following files exist:**

1. Model: `build/models/Llama2/Llama-2-70b-chat-hf/`
2. Preprocessed data at `build/preprocessed_data/llama2-70b/`:
  - `input_lens.npy`
  - `input_ids_padded.npy`
  - `mlperf_llama2_openorca_calibration_1k/data.parquet`

---

## Base Image

Llama2-70B shares the same TensorRT-LLM + nv-mlpinf container image family as DeepSeek-R1. The same image is used for both server and harness/client tasks. Use the x86 image for B200/B300 systems and the aarch64 image for GB200/GB300 systems.

| System | Arch | Server image | Client image |
| ------ | ---- | ------------ | ------------ |
| B200-SXM-180GBx8, B300-SXM-270GBx8 | x86_64 | `nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86` | Same as server image |
| GB200-NVL72, GB300-NVL72 (single node system or full rack system) | aarch64 | `nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64` | Same as server image |

For preparation instructions, see the [DeepSeek-R1 README - Base Image section](../deepseek_r1/README.md#base-image).

## Run the Benchmark in Single Node Using Docker

For interactive benchmarking on a single node (e.g., B200x8, B300x8):

### Step 1: Attach to Docker Container

```bash
export MLPERF_SCRATCH_PATH=/hps/franklin/mlperf_scratch
export MLPERF_IMAGE=nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
cd closed/Inventec
make attach_docker MLPERF_IMAGE=$MLPERF_IMAGE
```

### Step 2: Install nv-mlpinf Package

The following steps are performed inside the container.  SYSTEM_NAME is either `P9000G7-B300-SXM-270GBx8` or `P5800G7-B300-SXM-270GBx8`.

```bash
export SYSTEM_NAME=P9000G7-B300-SXM-270GBx8
pip install -e ".[llm]"
```

### Step 3: Run the Benchmark

**Offline scenario:**

```bash
# Launch server (runs in background)
nv-mlpinf run_llm_server --benchmarks=llama2-70b --scenarios=Offline

# Wait for server to be ready, then run harness
nv-mlpinf run_harness --benchmarks=llama2-70b --scenarios=Offline --test_mode=PerformanceOnly

# For accuracy test (99.0% target)
nv-mlpinf run_harness --benchmarks=llama2-70b --scenarios=Offline --test_mode=AccuracyOnly

# For high accuracy test (99.9% target)
nv-mlpinf run_harness --benchmarks=llama2-70b --scenarios=Offline --accuracy_target=.999 --test_mode=AccuracyOnly
```

**Server scenario:**

```bash
# Launch server
nv-mlpinf run_llm_server --benchmarks=llama2-70b --scenarios=Server

# Run harness
nv-mlpinf run_harness --benchmarks=llama2-70b --scenarios=Server --test_mode=PerformanceOnly

# For accuracy test
nv-mlpinf run_harness --benchmarks=llama2-70b --scenarios=Server --test_mode=AccuracyOnly
```

**To stop the server:** Exit the container and re-enter.

## Run the Benchmark in Multi Node using Slurm + Enroot

This section covers SLURM work through `nv-sflow`: image access, performance, accuracy, and compliance. Check [SLURM support](../../../../configs/SLURM_SUPPORT.md) before choosing a system/scenario.

### Prepare the image for multi node with Slurm + Enroot

Do not build a `.sqsh` manually for the nv-sflow path. Each run pulls the container image from `variables.CONTAINER_IMAGE` in the selected `slurm_env_sflow.yaml`. For example, [configs/llama2_70b/GB200-NVL72_GB200-186GB_aarch64x4/TRTLLM/Offline/slurm_env_sflow.yaml](../../../../configs/llama2_70b/GB200-NVL72_GB200-186GB_aarch64x4/TRTLLM/Offline/slurm_env_sflow.yaml) points to:

```yaml
variables:
  CONTAINER_IMAGE:
    value: "nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64"
```

Confirm your SLURM compute nodes can pull that image through Pyxis/Enroot before launching a full run.

### Common nv-sflow settings

Run from `closed/Inventec`. Replace `ACCT`, `PARTITION`, `SYSTEM`, and `NODES` for your cluster and target system. Choose a `SYSTEM` that contains matching `llama2_config_sflow.yaml` and `slurm_env_sflow.yaml` files for the scenario.  The example below uses the GB200x4 Llama2-70B nv-sflow config.

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
export RUN=llama2_70b_${SYSTEM}_offline_${TEST_MODE}_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/llama2_70b/${SYSTEM}/TRTLLM/Offline/llama2_config_sflow.yaml \
  -f configs/llama2_70b/${SYSTEM}/TRTLLM/Offline/slurm_env_sflow.yaml \
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
export RUN=llama2_70b_${SYSTEM}_server_${TEST_MODE}_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/llama2_70b/${SYSTEM}/TRTLLM/Server/llama2_config_sflow.yaml \
  -f configs/llama2_70b/${SYSTEM}/TRTLLM/Server/slurm_env_sflow.yaml \
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

Interactive uses disaggregated serving with the `trtllm_disagg_loadgen.yaml` disagg loadgen template. When matching Llama2-70B Interactive nv-sflow YAMLs are available for the target system, use the same command shape:

```bash
export TEST_MODE=PerformanceOnly  # use AccuracyOnly for accuracy
export RUN=llama2_70b_${SYSTEM}_interactive_${TEST_MODE}_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/llama2_70b/${SYSTEM}/TRTLLM/Interactive/llama2_config_sflow.yaml \
  -f configs/llama2_70b/${SYSTEM}/TRTLLM/Interactive/slurm_env_sflow.yaml \
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

### Key nv-sflow parameters

| Parameter                                            | Description                                                                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `-f configs/llama2_70b/.../llama2_config_sflow.yaml` | Benchmark, scenario, model, and topology settings                                                                   |
| `-f configs/llama2_70b/.../slurm_env_sflow.yaml`     | Container image, mounts, and SLURM defaults                                                                         |
| `-f scaleout/sflow/templates/...`                    | LoadGen template; use `trtllm_ifb_loadgen.yaml` for Offline/Server and `trtllm_disagg_loadgen.yaml` for Interactive |
| `--set TEST_MODE=...`                                | `PerformanceOnly` or `AccuracyOnly`                                                                                 |
| `--set HARNESS_EXTRA_ARGS=...`                       | Extra harness flags such as audit tests or high-accuracy options                                                    |
| `--nodes`, `--partition`, `--account`                | SLURM submission values                                                                                             |
| `-o`                                                 | Generated sbatch script path                                                                                        |
| `--submit`                                           | Submit the generated sbatch script; omit it to only generate the script                                             |

## Compliance Testing

Llama2-70B uses standard MLPerf compliance tests (TEST06 for applicable scenarios).

### Single Node (Docker)

**Offline scenario:**

```bash
# Launch the server first
nv-mlpinf run_llm_server --benchmarks=llama2-70b --scenarios=Offline

# Run compliance test
make run_audit_test06 RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --test_mode=PerformanceOnly"

# Exit container to close the server before running another test
```

**Server scenario:** Similar process with `--scenarios=Server`

### Multi-Node (SLURM)

Run compliance through `nv-sflow` in `PerformanceOnly` mode by passing the audit test through `HARNESS_EXTRA_ARGS`. Set `SCENARIO=Offline` or `Server`, and set `TEST=TEST06 as applicable.

```bash
SCENARIO=Offline
TEST=TEST06
RUN=llama2_70b_${SYSTEM}_${SCENARIO}_${TEST}_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/llama2_70b/${SYSTEM}/TRTLLM/${SCENARIO}/llama2_config_sflow.yaml \
  -f configs/llama2_70b/${SYSTEM}/TRTLLM/${SCENARIO}/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/trtllm_ifb_loadgen.yaml \
  --set WORK_DIR=$PWD \
  --set TEST_MODE=PerformanceOnly \
  --set HARNESS_EXTRA_ARGS="--audit_test=$TEST" \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
```

---

## Expected Results

You should expect to get the following accuracy results:

**99.0% target:**

```
accuracy: [PASSED] ROUGE1: 44.495 (Threshold=43.836) | [PASSED] ROUGE2: 22.089 (Threshold=21.689) | [PASSED] ROUGEL: 28.694 (Threshold=28.222) | [PASSED] TOKENS_PER_SAMPLE: 293.100 (Threshold=263.970)
```

**99.9% target:**

```
accuracy: [PASSED] ROUGE1: 44.939 (Threshold=44.383) | [PASSED] ROUGE2: 22.326 (Threshold=21.929) | [PASSED] ROUGEL: 28.978 (Threshold=28.507) | [PASSED] TOKENS_PER_SAMPLE: 294.450 (Threshold=265.005)
```

---

## Benchmark Passing Criteria

A run is valid only if the required accuracy and performance criteria pass. For Server and Interactive scenarios, both TTFT and TPOT latency must be within the threshold; reduce target QPS and rerun if either latency metric fails.

The accuracy thresholds below are for the high-accuracy `llama2-70b-99.9` target.

| Scenario | Accuracy criteria | Performance criteria |
| --- | --- | --- |
| Offline | `ROUGE1 >= 44.4312 * 0.999 = 44.3867688`; `ROUGE2 >= 22.0352 * 0.999 = 22.0131648`; `ROUGEL >= 28.6162 * 0.999 = 28.5875838`; `TOKENS_PER_SAMPLE >= 294.45 * 0.9 = 265.005` | No TTFT/TPOT constraint |
| Server | `ROUGE1 >= 44.4312 * 0.999 = 44.3867688`; `ROUGE2 >= 22.0352 * 0.999 = 22.0131648`; `ROUGEL >= 28.6162 * 0.999 = 28.5875838`; `TOKENS_PER_SAMPLE >= 294.45 * 0.9 = 265.005` | TTFT <= 2 s; TPOT <= 200 ms |
| Interactive | `ROUGE1 >= 44.4312 * 0.999 = 44.3867688`; `ROUGE2 >= 22.0352 * 0.999 = 22.0131648`; `ROUGEL >= 28.6162 * 0.999 = 28.5875838`; `TOKENS_PER_SAMPLE >= 294.45 * 0.9 = 265.005` | TTFT <= 450 ms; TPOT <= 40 ms |
