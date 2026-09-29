# MLPerf Inference 6.1

## Setup

### Model and Dataset
To download model and dataset we are using `/data` directory.
Download the dataset for the benchmark by running the below command.

```bash
bash setup/gpt-oss-120b/download_dataset.sh
```

## Download and Quantize Model

### Option 1: (Preferred option for preparing submissions): Download Already Quantized Model

Download already quantized model using HF token provided to you by AMD

```bash
HUGGINGFACE_ACCESS_TOKEN="<your HF token from AMD goes here>"
bash setup/gpt-oss-120b/download_model_fp4.sh --token $HUGGINGFACE_ACCESS_TOKEN --download-prequantized
```

### Option 2: Download and Quantize Model From Scratch

The following script will handle download and quantization:
Note: This script will skip downloading the original model if the unquantized model already exists at `/data/inference/model/gpt-oss-120b/orig/`. If the original model is not present in this directory, the script will first download it and then proceed with quantization.

```bash
bash setup/gpt-oss-120b/download_model_fp4.sh
```

In all cases, quantized model can be found in `/data/inference/model/gpt-oss-120b/fp4_quantized/`

## Running Inference Benchmarks

### Runtime tunables

To boost the machine's performance further, execute the following script before any performance test (should be set once after a reboot):

```bash
bash setup/runtime_tunables.sh
```

### Docker

Build the docker image for the benchmark by running the below command

```bash
bash setup/gpt-oss-120b/build_docker.sh
```

Start the docker container for the benchmark by running the below commands

```bash
export EXTRA_ARGS="--rm --workdir /lab-mlperf-inference/submission"
bash setup/gpt-oss-120b/start_docker.sh
```

## Running Experiments for Submission

### Run Multiple Experiments and Select Best Runs

#### 1. Run Experiments

**Offline Scenario:**

```bash
python3 submission.py --model gpt-oss-120b experiment --scenario Offline --model-conf /lab-mlperf-inference/code/gpt-oss-120b/offline_mi355x.yaml --user-conf /lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf
```

**Server Scenario:**

```bash
python3 submission.py --model gpt-oss-120b experiment --scenario Server --model-conf /lab-mlperf-inference/code/gpt-oss-120b/server_mi355x.yaml --user-conf /lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf
```

#### 2. Check Status

Check the current best model state:

```bash
python3 submission.py --model gpt-oss-120b status
```

#### 3. Update Best Result

Select the current best result for a scenario:

```bash
# Offline
python3 submission.py --model gpt-oss-120b update_best --scenario Offline

# Server
python3 submission.py --model gpt-oss-120b update_best --scenario Server
```

#### 4. Prepare (Accuracy/Compliance)

**Run Accuracy:**

```bash
# Offline
python3 submission.py --model gpt-oss-120b prepare --scenario Offline accuracy

# Server
python3 submission.py --model gpt-oss-120b prepare --scenario Server accuracy
```

**Run Compliance:**

gpt-oss-120b has two compliance tests (`TEST07`, `TEST09`), so `--test-version` is required.

```bash
# Offline
python3 submission.py --model gpt-oss-120b prepare --scenario Offline --test-version TEST07 compliance
python3 submission.py --model gpt-oss-120b prepare --scenario Offline --test-version TEST09 compliance

# Server
python3 submission.py --model gpt-oss-120b prepare --scenario Server --test-version TEST07 compliance
python3 submission.py --model gpt-oss-120b prepare --scenario Server --test-version TEST09 compliance
```

**Force Overwrite (if result already exists):**

```bash
python3 submission.py --model gpt-oss-120b prepare --scenario Offline --force accuracy
python3 submission.py --model gpt-oss-120b prepare --scenario Offline --test-version TEST07 --force compliance
python3 submission.py --model gpt-oss-120b prepare --scenario Offline --test-version TEST09 --force compliance
```

#### 5. Package for Submission

Package the current best results (run `status` first to verify everything is ready):

```bash
# Set required environment variables first
export GPU_COUNT=8
export GPU_NAME="MI355X"
export CPU_COUNT=2
export CPU_NAME="EPYC-9575F" #Set CPU_NAME based on your hardware you use. You can use `lscpu | grep name`.
export COMPANY="AMD" #Set Your company name.

# Then package
python3 submission.py --model gpt-oss-120b package
```

> **Note:** See `.submission_package_env` for environment variable details.
