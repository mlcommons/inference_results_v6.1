# Llama3.1

## Getting started

### Download Model

Please download model files by following the mlcommons README.md with instructions:

```bash
# following steps: https://github.com/mlcommons/inference/tree/master/language/llama3.1-405b#get-model
# If your company has license concern, please download the model from the following link: https://llama3-1.mlcommons.org/
export CHECKPOINT_PATH=build/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct ${CHECKPOINT_PATH}
cd ${CHECKPOINT_PATH} && git checkout be673f326cab4cd22ccfef76109faf68e41aa5f1
```

### Download and Prepare Data

Please download data files by following the mlcommons README.md with instructions.
Please move the downloaded pickle into expected path and follow steps to run the required data pre-processing:

```bash
# follow: https://github.com/mlcommons/inference/tree/master/language/llama3.1-405b#get-dataset
# to download file: mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl, mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl

# make sure you are in mlperf's container
make prebuild ENV=release BENCHMARKS=llama

# move into right directory
mv mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl build/data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl
mv mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl build/data/llama3.1-405b/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl

# run pre-process step for llama3
python3 code/llama3_1-405b/tensorrt/preprocess_data.py --data_dir build/data/ --preprocessed_data_dir build/preprocessed-data
```

Make sure after the 2 steps above, you have:

1. model downloaded at: `build/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct/`
2. preprocessed data at `build/preprocessed_data/llama3.1-405b/`:

- `build/preprocessed_data/llama3.1-405b/input_lens.npy`
- `build/preprocessed_data/llama3.1-405b/input_ids_padded.npy`
- `build/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_dataset_512_processed_fp16_calibration/data.parquet`

## Build and run the benchmarks

Please follow the steps below in MLPerf container.

```bash
# make sure you are in mlperf's container
make prebuild ENV=release BENCHMARKS=llama

# if not, make sure you already built TRTLLM
# No longer needed if it's release container
# make build

make run_llm_server RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=Offline --core_type=trtllm_endpoint"
make run_harness RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
```

## Multi-Node Runs (NVL72)

Llama3.1-405B requires multi-node systems for optimal performance. Use the scaleout scripts:

- **Offline scenario**: Uses In-Flight Batching (IFB) via `run_scaleout.sh`
- **Server/Interactive scenarios**: Uses disaggregated prefill/decode via `run_scaleout_disagg.py`

See `scaleout/REPRODUCE.md` for detailed commands and `run_disagg_405B/README.md` for disaggregated serving documentation.

