# Whisper Benchmarks

This README provides instructions for building, preprocessing data, and running performance and accuracy tests for the Whisper model.

## Prepare Whisper Model

```bash
bash -x /work/code/whisper/tensorrt/download_models.sh
```
Model dir is set as

`/work/build/models/whisper-large-v3/`

Please check the folder
```bash
$ tree /work/build/models/whisper-large-v3/
/work/build/models/whisper-large-v3/
├── large-v3.pt
├── mel_filters.npz
└── multilingual.tiktoken
```


## Preprocess Dataset

To preprocess the Whisper dataset, use the following command:

```bash
cp 3rdparty/mlc-inference/speech2text/utils/inference_librispeech.csv /work/code/whisper/tensorrt/utils/inference_librispeech.csv
source .llm_x86_64/bin/activate
BENCHMARKS="whisper" make preprocess_data
deactivate
```

This command will:

1.  Download the LibriSpeech `dev-clean` and `dev-other` datasets into `/work/build/data/whisper-large-v3/LibriSpeech/`.
2.  Pre-pack the audio files into `/work/build/preprocessed_data/whisper-large-v3/dev-all-repack/`.
3.  Generate a manifest file at `/work/build/preprocessed_data/whisper-large-v3/dev-all-repack.json`.
# Whisper Benchmarks

This README provides instructions for building, preprocessing data, and running performance and accuracy tests for the Whisper model.
## Generate Checkpoint and Generate Engines

```bash
make generate_engines RUN_ARGS="--benchmarks=whisper --scenarios=Offline"

```

The checkpoint dir `/work/build/models/whisper-large-v3/whisper_large_v3_float16_weight_ckt/` should be:

```
$ tree /work/build/models/whisper-large-v3/whisper_large_v3_float16_weight_ckt/
/work/build/models/whisper-large-v3/whisper_large_v3_float16_weight_ckt/
├── decoder
│   ├── config.json
│   └── rank0.safetensors
├── encoder
│   ├── config.json
│   └── rank0.safetensors
├── stderr.txt
└── stdout.txt

3 directories, 6 files
```

The engine dir `/work/build/engnines/<Your Hardware Plateform>/whisper/*/`

```bash

├── decoder
│   ├── config.json
│   ├── rank0.engine
│   ├── stderr.txt
│   └── stdout.txt
└── encoder
    ├── config.json
    ├── rank0.engine
    ├── stderr.txt
    └── stdout.txt

3 directories, 8 files
```
## Accuracy Test

To run an accuracy test, use the following command:

```bash
make run_harness RUN_ARGS="--benchmarks=whisper --scenarios=Offline --test_mode=AccuracyOnly"
```
### B200x8 AccuracyOnly Test Expected Results:
```
Word Error Rate: 2.1688856743150264%, accuracy=97.83111432568498%

======================== Result summaries: ========================

Offline Scenario:
+----------------------+-------------+-----------+------------------+---------------+------------------+-------------+
| System Name          | Benchmark   | Setting   | All Acc. Pass?   | Metric Name   |   Measured Value | Threshold   |
+======================+=============+===========+==================+===============+==================+=============+
| B200-SXM-180GBx8_TRT | whisper     | cp990     | Yes              | ACCURACY      |            97.83 | >=96.953571 |
+----------------------+-------------+-----------+------------------+---------------+------------------+-------------+

```

## Performance Test

To run a performance test, use the following command:

```bash
make run_harness RUN_ARGS="--benchmarks=whisper --scenarios=Offline --verbose --test_mode=PerformanceOnly"
```

## FAQ

1.
```bash
    import tensorrt_llm
ModuleNotFoundError: No module named 'tensorrt_llm
```
reintall `tensorrt_llm/*whl`
`
pip install ./3rdparty/trtllm/build/tensorrt_llm*.whl
`
