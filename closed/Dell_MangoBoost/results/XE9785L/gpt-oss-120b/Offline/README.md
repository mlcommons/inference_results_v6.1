## Dell_MangoBoost MLPerf Inference v6.1 Submission

**Intra-Node P/D - MI355X GPT-OSS with liquid-cooling**: We submit single node MI355X on XE9785L, which is a liquid cooling MI355X node. To make a performant goodput, we applied ***Intra-node P/D disaggregation*** optimization in Interactive Scenario, with 4-GPUs dedicating working on prefill, plus 4-GPUs dedicating working on decode, connected with XGMI KV connector. The result shows great goodput on interactive Scenario, which achieves **29K** throughput and at the same time maintain P99 TTFT < 2s and P99 TPOT < 15ms.

## 1. Setup

### Model and Dataset

Download the dataset for the benchmark by running the below command

```bash
bash setup/gpt-oss-120b/download_dataset.sh
```

Download the model for the benchmark by running the below command

```bash
HUGGINGFACE_ACCESS_TOKEN="<your HF token goes here>"
bash setup/gpt-oss-120b/download_model_fp4.sh --token $HUGGINGFACE_ACCESS_TOKEN --download-prequantized
```

## Inference

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


## 2. Offline Scenario

Run the following commands inside the docker container

``` bash
## Performance
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/gpt-oss-120b/ \
   --config-name offline_mi355x \
   test_mode=performance \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Offline/performance/run_1

## Accuracy
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/gpt-oss-120b/ \
   --config-name offline_mi355x \
   test_mode=accuracy \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Offline/accuracy \
   vllm_sampling_config.max_tokens=32768

### Evaluate accuracy
bash /lab-mlperf-inference/code/scripts/check_gptoss_accuracy_scores.sh \
   /lab-mlperf-inference/results/gpt-oss-120b/Offline/accuracy/mlperf_log_accuracy.json
```

## 3. Server Scenario

Run the following commands inside the docker container

```bash
## Performance
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/gpt-oss-120b/ \
   --config-name server_mi355x \
   test_mode=performance \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Server/performance/run_1

## Accuracy
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/gpt-oss-120b/ \
   --config-name server_mi355x \
   test_mode=accuracy \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Server/accuracy \
   vllm_sampling_config.max_tokens=32768

### Evaluate accuracy
bash /lab-mlperf-inference/code/scripts/check_gptoss_accuracy_scores.sh \
   /lab-mlperf-inference/results/gpt-oss-120b/Server/accuracy/mlperf_log_accuracy.json
```