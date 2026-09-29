# Model config

## Omegaconf-based configuration
The model configuration is structured in the following way:

```yml
# benchmark details and backend
benchmark_name: llama3_1-405b
scenario: server
test_mode: performance
backend: sglang
engine_version: async

# Environmental variables for all backends
env_config:
  HIP_FORCE_DEV_KERNARG: 1
  ...

# Environmental variables for only the vllm backend
vllm_env_config:
  VLLM_LOGGING_LEVEL: "ERROR"
  ...

# Environmental variables for only the sglang backend
sglang_env_config:
  SGLANG_AITER_MOE: 1
  ...

# Engine parameters passed directly to sglang during engine init
sglang_engine_config:
  model_path: /model/llama3.1-405b/fp8_quantized
  ...

# Sampling parameters passed to the inference calls with sglang
sglang_sampling_config:
  temperature: 1.0
  ...

# Engine parameters passed directly to vllm during engine init
vllm_engine_config:
  model: /model/llama3.1-405b/fp8_quantized
  ...

# Sampling parameters passed to the inference calls with vllm
vllm_sampling_config:
  temperature: 1
  ...

# Parameters for the harness
harness_config:
  dataset_path: /data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl
  ...
```
There is a YAML file, [config.yaml](harness_llm/common/config.yaml), with default values that all configurations share. The complete configurations for the specific model-scenario-machine trios can be found in the usual code/models directory; there is no change in that.

A new requirement for any benchmark run is to provide the backend parameter for the run, which can be vllm or sglang. This must be provided in the command line or in the script you use; it is not enough to set it in the model config file. Please note that for some use cases, there might not be an sglang config yet.

There is a new [run_harness.sh](run_harness.sh) script for running the benchmark. It will run the power setting script for your scenario before starting the actual benchmark. Running main.py also works as before, with the addition that you must provide a value for the backend harness parameter. Check the [README](../README.md) for example commands.
