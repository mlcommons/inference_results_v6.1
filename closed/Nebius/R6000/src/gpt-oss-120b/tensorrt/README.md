# GPT-OSS-120B

## Getting Started

###
Follow model-agnostic steps described in `closed/Nebius/README.md`.

### Download Model and Dataset
Please refer to the reference implementation README on mlcommons/inference for instructions to download datasets and model ([here](https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b#model-and-dataset-download))

### Build a Docker image
Run from `closed/Nebius`
```
BASE_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20 ENV=release ARGS="--base_requirements --mitten --loadgen --llm_requirements" make build_mlperf_docker
```
Store the name of the newly build Docker image
```
export DOCKER_IMAGE=...
```

## Run

### Start a Docker container
```
docker run --rm -it \
      --gpus all \
      --net host \
      --shm-size=32gb \
      --ulimit memlock=-1 \
      -v $(pwd):/work \
      -v $MLPERF_SCRATCH_PATH:$MLPERF_SCRATCH_PATH \
      -e MLPERF_SCRATCH_PATH \
      -w /work \
	  $DOCKER_IMAGE
```

### Set up once

#### Prepare Data

From `closed/Nebius` run
```
make link_dirs
```

The GPT-OSS-120B benchmark uses separate datasets for accuracy and performance runs:

`/work/build/data` is a symlink that points to `$MLPERF_SCRATCH_PATH/data`

```bash
# Create data directories
mkdir -p /work/build/data/gpt-oss/v4/acc
mkdir -p /work/build/data/gpt-oss/v4/perf
```
Under v4/acc, place:
- input_ids_padded.npy
- input_lens.npy
- acc_eval_ref.parquet (reference data for evaluation)

Under v4/perf, place
- input_ids_padded.npy
- input_lens.npy

Expected data layout:

```
build/data/gpt-oss/v4/
├── acc/
│   ├── input_ids_padded.npy    # Tokenized inputs (padded)
│   ├── input_lens.npy          # Actual input lengths
│   └── acc_eval_ref.parquet    # Ground truth for accuracy evaluation
└── perf/
    ├── input_ids_padded.npy
    └── input_lens.npy
```

#### Compliance Data Preprocessing

For `TEST07` compliance testing, you need to preprocess the GPQA compliance dataset:

```bash
# Preprocess compliance data for TEST07
python code/gpt-oss-120b/tensorrt/preprocess_compliance_data.py \
    --input-file build/data/gpt-oss/v4/acc/acc_eval_compliance_gpqa.parquet \
    --output-dir build/data/gpt-oss/v4/compliance/test07
```

This creates:
```
build/data/gpt-oss/v4/compliance/test07/
├── input_ids_padded.npy    # Tokenized inputs (990 samples)
└── input_lens.npy          # Actual input lengths
```

#### Configuration

The benchmark supports test_mode-aware configuration. The harness automatically selects the appropriate dataset and generation config based on `--test_mode`:

- `--test_mode=AccuracyOnly`: Uses accuracy dataset, max_output_len=32768
- `--test_mode=PerformanceOnly`: Uses performance dataset, max_output_len=10240

See `configs/RTX_PRO_6000_PCIE_96GBx8/Offline/gpt-oss-120b.py` for system-specific configurations.

### Set up every time
```
pip install --upgrade onnx_graphsurgeon huggingface_hub
export SYSTEM_NAME=RTX_PRO_6000_PCIE_96GBx8
export SUBMITTER=Nebius
```

### Benchmark

#### Offline Performance
##### LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
##### Harness
When the server is ready run
```
make run_harness
```

#### Offline Accuracy
##### LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"
make run_llm_server
```
##### Harness
When the server is ready run
```
make run_harness
```

#### Offline TEST07
##### LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
##### Harness
When the server is ready run
```
make run_audit_test07
```

#### Offline TEST09
##### LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
##### Harness
When the server is ready run
```
make run_audit_test09
```

#### Server Performance
#### LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
##### Harness
When the server is ready run
```
make run_harness
```

#### Server Accuracy
##### LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint --test_mode=AccuracyOnly"
make run_llm_server
```
##### Harness
When the server is ready run
```
make run_harness
```

#### Server TEST07
##### LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
##### Harness
When the server is ready run
```
make run_audit_test07
```

#### Server TEST09
##### LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
##### Harness
When the server is ready run
```
make run_audit_test09
```