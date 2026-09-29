# GPT-OSS-120B

## Getting Started

### Download Model and Dataset
Please refer to the reference implementation README on mlcommons/inference for instructions to download datasets and model ([here](https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b#model-and-dataset-download))

### Download and Prepare Data

The GPT-OSS-120B benchmark uses separate datasets for accuracy and performance runs:

`/work/build/data` is a symlink that points to `$MLPERF_SCRATCH_PATH/data`

```bash
# Create data directories
mkdir -p /work/build/data/gpt-oss/v4/acc
mkdir -p /work/build/data/gpt-oss/v4/perf

# Under v4/acc, place:
# - input_ids_padded.npy
# - input_lens.npy
# - acc_eval_ref.parquet (reference data for evaluation)

# Under v4/perf, place
# - input_ids_padded.npy
# - input_lens.npy
```

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

**Dataset Statistics:**

| Dataset             | Samples | Max ISL | Max OSL | Benchmarks                                             |
| ------------------- | ------- | ------- | ------- | ------------------------------------------------------ |
| Accuracy            | 4,395   | 2,871   | 32,768  | AIME (240), GPQA (990), LiveCodeBench (3,165)          |
| Performance         | 6,396   | 15,330  | 10,240  | pubmed_summarization (synthetic)                       |
| Compliance (TEST07) | 990     | 2,871   | 10,240  | GPQA subset (accuracy verification in perf mode)       |
| Compliance (TEST09) | 6,396   | 15,330  | 10,240  | Same as Performance (output token length verification) |

### Compliance Data Preprocessing

For TEST07 compliance testing, you need to preprocess the GPQA compliance dataset:

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

### Configuration

The benchmark supports test_mode-aware configuration. The harness automatically selects the appropriate dataset and generation config based on `--test_mode`:

- `--test_mode=AccuracyOnly`: Uses accuracy dataset, max_output_len=32768
- `--test_mode=PerformanceOnly`: Uses performance dataset, max_output_len=10240

See `configs/<SYSTEM_NAME>/Offline/gpt-oss-120b.py` for system-specific configurations.

## Run

Please use `scaleout` to run batched jobs:

```bash
sbatch --nodes=1 -p gb200 \
    ./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system GB200-NVL72_GB200-186GB_aarch64x1 \
    --gpus-per-node 4 \
    --dp-multiplicity 4 \
    --container-image docker://nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64 \
    --run-args "--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint"
```

Note that, for x86 machine (HGX systems, for example), you will have to use a different docker tag: `nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86`

### Interactively

```bash
# Enter the MLPerf container
make prebuild ENV=release BENCHMARKS=gptoss
```

```bash
# To use local checkpoint add --llm_quantizer_outdir=<path> to the RUN_ARGS
# To run accuracy mode, change PerformanceOnly to AccuracyOnly
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
export SYSTEM_NAME=B200-SXM-180GBx8
make run_llm_server
make run_harness
```

The best way to clear the context after running the benchmark is to exit the container and re-enter. It takes <1 min.

### Running Compliance Tests

GPT-OSS-120B requires TEST07 and TEST09 compliance tests. **IMPORTANT:** Each test must run with a specific test mode.

**Offline scenario:**
```bash
# TEST07: Run with PerformanceOnly mode
# Launch the server first
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
make run_audit_test07
# Close the server before running another server

# TEST09: Run with AccuracyOnly mode
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"
make run_llm_server
make run_audit_test09
```

It's similar process for server

| Test   | Test Mode       | Purpose                                         | Dataset           | Gen Config                          | Samples |
| ------ | --------------- | ----------------------------------------------- | ----------------- | ----------------------------------- | ------- |
| TEST07 | PerformanceOnly | Accuracy in perf mode (GPQA threshold: 60.698%) | compliance/test07 | performance (reasoning-effort=High) | 990     |
| TEST09 | AccuracyOnly    | Output token length verification                | performance       | performance                         | 6,396   |

### Debugging Accuracy Issues

1. Verify correct dataset is loaded (accuracy vs performance)
2. Check `max_seq_len` supports full output length (3072 + 32768 for accuracy)
3. Ensure generation config matches test mode
