# MLPerf Inference 6.1

## Setup

### Model and Dataset

Download the dataset for the benchmark by running the below command

```bash
bash setup/llama2-70b-99/download_dataset.sh
```

Download the model for the benchmark by running the below command

```bash
HUGGINGFACE_ACCESS_TOKEN="<your HF token goes here>"
bash setup/llama2-70b-99/download_model_fp4.sh --token $HUGGINGFACE_ACCESS_TOKEN --download-prequantized
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
bash setup/llama2-70b-99/build_docker.sh
```

Start the docker container for the benchmark by running the below commands

```bash
export EXTRA_ARGS="--rm --workdir /lab-mlperf-inference/submission"
bash setup/llama2-70b-99/start_docker.sh
```


### Running the benchmark

Run the following commands inside the docker container

``` bash
## Performance
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/llama2-70b-99.9/ \
   --config-name server_mi355x \
   test_mode=performance \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99.9/user_mi355x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b-99.9/Server/performance/run_1

## Accuracy
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/llama2-70b-99.9/ \
   --config-name server_mi355x \
   test_mode=accuracy \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99.9/user_mi355x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b-99.9/Server/accuracy 

### Evaluate accuracy
bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
   /lab-mlperf-inference/results/llama2-70b-99.9/Server/accuracy/mlperf_log_accuracy.json
```
