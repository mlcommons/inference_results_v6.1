# MLPerf Inference 6.1 - deepseek-r1

## Setup

### Model and Dataset

Download the dataset:

```bash
DOWNLOAD_DIR="/data/inference/data/deepseek-r1"
sudo mkdir -p "${DOWNLOAD_DIR}"
sudo chown -R "$(id -un):$(id -gn)" "${DOWNLOAD_DIR}"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d "${DOWNLOAD_DIR}" https://inference.mlcommons-storage.org/metadata/deepseek-r1-datasets-fp8-eval.uri
```

Download the model:

```bash
MODEL_DIR="/data/inference/model/S3_sq_a05_v2"
sudo mkdir -p "${MODEL_DIR}"
sudo chown -R "$(id -un):$(id -gn)" "${MODEL_DIR}"
hf download amd/Deepseek-S3_sq_a05_v2_mlperf6_1 \
  --local-dir "${MODEL_DIR}"
```

### Quantise the model

The submitted checkpoint (`S3_sq_a05_v2`) is prepared using
`setup/deepseek-r1/dataset_and_model/prepare_model.sh`, which runs inside the
model/dataset prep docker image. The recipe downloads the original FP8
DeepSeek-R1 release from HuggingFace
([`deepseek-ai/DeepSeek-R1`](https://huggingface.co/deepseek-ai/DeepSeek-R1)),
dequantises FP8 to BF16, then applies MXFP4 weight/activation quantization with
SmoothQuant (alpha = 0.5) using AMD Quark 0.10, and exports a HuggingFace
checkpoint.

For calibration, we use the full calibration dataset provided by
[mlcommons/inference](https://mlcommons.org/benchmarks/inference-datacenter/) for
DeepSeek-R1, which is downloaded as part of the dataset step above.

The script downloads the model automatically. It takes three positional
arguments: the HuggingFace token, a `skip-download` flag, and a
`download-prequantized` flag. To reproduce the checkpoint from the original FP8
release:

```bash
bash setup/deepseek-r1/dataset_and_model/prepare_model.sh "${HUGGINGFACE_ACCESS_TOKEN}"
```

### Runtime tunables

```bash
bash setup/runtime_tunables.sh
```

### Docker

```bash
bash setup/deepseek-r1/build_docker.sh
export EXTRA_ARGS="--rm --workdir /lab-mlperf-inference/submission"
bash setup/deepseek-r1/start_docker.sh
```

## Running Experiments for Submission

#### 1. Run Experiments

**Offline Scenario:**

```bash
python3 submission.py --model deepseek-r1 experiment --scenario Offline \
  --model-conf /lab-mlperf-inference/code/deepseek-r1/offline_mi355x_mn.yaml \
  --user-conf /lab-mlperf-inference/code/deepseek-r1/user_mi355x_mn.conf
```

**Server Scenario:**

```bash
python3 submission.py --model deepseek-r1 experiment --scenario Server \
  --model-conf /lab-mlperf-inference/code/deepseek-r1/server_mi355x_mn.yaml \
  --user-conf /lab-mlperf-inference/code/deepseek-r1/user_mi355x_mn.conf
```

#### 2. Check Status

```bash
python3 submission.py --model deepseek-r1 status
```

#### 3. Prepare (Accuracy/Compliance)

```bash
python3 submission.py --model deepseek-r1 prepare --scenario Offline accuracy
python3 submission.py --model deepseek-r1 prepare --scenario Server accuracy
python3 submission.py --model deepseek-r1 prepare --scenario Offline compliance
python3 submission.py --model deepseek-r1 prepare --scenario Server compliance
```

#### 4. Package for Submission

```bash
export GPU_COUNT=72 GPU_NAME="mi355x" CPU_COUNT=2 CPU_NAME="EPYC-9575F" COMPANY="AMD"
python3 submission.py --model deepseek-r1 package
```


Note: To run Multi-node runs refer README.md here - src/harness_llm/backends/vllm/zmq/
