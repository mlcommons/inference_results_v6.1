# MLPerf Inference 6.1

## Setup

### Model and Dataset
To download model and dataset we are using `/data` directory.
Download the dataset for the benchmark by running the below command.

```bash
bash setup/llama2-70b-99/download_dataset.sh
```

## Download and Quantize Model

If you already have the unquantized model, place it under `/data/inference/model/llama2-70b-chat-hf/orig/`.
If the model already exists, the HF token is not required and the download step will be skipped.

### To Quantize Model

```bash
MODEL_OPTION="_fp4"
bash setup/llama2-70b-99/download_model$MODEL_OPTION.sh
```

If the model does not exist, provide your Hugging Face token (via the `--token` flag) to download the model first and then quantize.

```bash
HUGGINGFACE_ACCESS_TOKEN="<your HF token goes here>"
MODEL_OPTION="_fp4"
bash setup/llama2-70b-99/download_model$MODEL_OPTION.sh --token "$HUGGINGFACE_ACCESS_TOKEN"
```

Output locations:
- Unquantized model: `/data/inference/model/llama2-70b-chat-hf/orig/`
- FP4 quantized model: `/data/inference/model/llama2-70b-chat-hf/fp4_quantized/`

## Running Inference Benchmarks

### Runtime tunables

To boost the machine's performance further, execute the following script before any performance test (should be set once after a reboot):

```bash
bash setup/runtime_tunables.sh
```

### Docker

Build the docker image for the benchmark by running the below command

```bash
bash setup/llama2-70b-99/build_docker.sh
```

Start the docker container for the benchmark by running the below commands

```bash
export EXTRA_ARGS="--rm --workdir /lab-mlperf-inference/submission"
bash setup/llama2-70b-99/start_docker.sh
```

## Running Experiments for Submission

### Run Multiple Experiments and Select Best Runs

#### 1. Run Experiments

**Offline Scenario:**

```bash
python3 submission.py --model llama2-70b-99 experiment --scenario Offline --model-conf /lab-mlperf-inference/code/llama2-70b-99/offline_mi355x.yaml --user-conf /lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf
```

**Server Scenario:**

```bash
python3 submission.py --model llama2-70b-99 experiment --scenario Server --model-conf /lab-mlperf-inference/code/llama2-70b-99/server_mi355x.yaml --user-conf /lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf
```

**Interactive Scenario:**

```bash
python3 submission.py --model llama2-70b-99 experiment --scenario Interactive --model-conf /lab-mlperf-inference/code/llama2-70b-99/interactive_mi355x.yaml --user-conf /lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf
```

#### 2. Check Status

Check the current best model state:

```bash
python3 submission.py --model llama2-70b-99 status
```

#### 3. Update Best Result

Select the current best result for a scenario:

```bash
# Offline
python3 submission.py --model llama2-70b-99 update_best --scenario Offline

# Server
python3 submission.py --model llama2-70b-99 update_best --scenario Server

# Interactive
python3 submission.py --model llama2-70b-99 update_best --scenario Interactive
```

#### 4. Prepare (Accuracy/Compliance)

**Run Accuracy:**

```bash
# Offline
python3 submission.py --model llama2-70b-99 prepare --scenario Offline accuracy

# Server
python3 submission.py --model llama2-70b-99 prepare --scenario Server accuracy

# Interactive
python3 submission.py --model llama2-70b-99 prepare --scenario Interactive accuracy
```

**Run Compliance:**

```bash
# Offline
python3 submission.py --model llama2-70b-99 prepare --scenario Offline compliance

# Server
python3 submission.py --model llama2-70b-99 prepare --scenario Server compliance

# Interactive
python3 submission.py --model llama2-70b-99 prepare --scenario Interactive compliance
```

**Force Overwrite (if result already exists):**

```bash
python3 submission.py --model llama2-70b-99 prepare --scenario Offline --force accuracy
python3 submission.py --model llama2-70b-99 prepare --scenario Offline --force compliance
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
python3 submission.py --model llama2-70b-99 package
```

> **Note:** See `.submission_package_env` for environment variable details.
