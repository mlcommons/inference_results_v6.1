# Llama 3.1 8B preparation

The canonical configuration is `config/model/llama3.1_8b.yaml`. Set
`MODEL_ROOT` and `DATA_ROOT` in `config/deployment.env`; the paths below are
the only model and data paths consumed by that configuration.

## Data

Follow NVIDIA's [v6.0 Llama 3.1 8B procedure](https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA/src/llama3.1-8b/tensorrt)
to obtain the release-matched CNN/DailyMail evaluation and calibration sources:

```text
${DATA_ROOT}/llama3.1-8b/
├── cnn_eval.json
└── cnn_dailymail_calibration.json
```

`cnn_eval.json` is the accuracy source and supplies the existing `tok_input`
values. `cnn_dailymail_calibration.json` is used only to prepare the
quantization calibration input.

## Data preprocessing

Produce the performance arrays and Quark calibration input with the packaged
helper:

```bash
python3 src/models/llama3_1_8b/preprocess_llama31_8b.py \
  --eval-json "${DATA_ROOT}/llama3.1-8b/cnn_eval.json" \
  --calibration-json "${DATA_ROOT}/llama3.1-8b/cnn_dailymail_calibration.json" \
  --output-dir "${DATA_ROOT}/llama3.1-8b/preprocessed"
```

The required result is:

```text
${DATA_ROOT}/llama3.1-8b/preprocessed/
├── input_ids_padded.npy
├── input_lens.npy
└── cnn_dailymail_calibration.pkl
```

The helper validates and pads supplied tokens. It does not retokenize,
resample, or download CNN/DailyMail data.

## AMD model

Download the licensed Meta Llama 3.1 8B Instruct checkpoint at the revision
specified by AMD's [v6.0 Llama 3.1 8B source](https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/AMD/src/llama3.1-8b).
Place the original checkpoint and tokenizer at `${MODEL_ROOT}/llama3.1-8b/`,
with `config.json` at that root. The MI350X configuration selects the local
MXFP4 artifact
`${MODEL_ROOT}/llama3.1-8b/fp4_quantized_awq_gptq_seq2560_bf16/`.

Create `config/model_prep.env` from its example, build the packaged Quark image,
and generate the artifact:

```bash
cp config/model_prep.env.example config/model_prep.env
./scripts/docker/build_mi350x_quark_prep.sh
./scripts/model_prep/quantize_llama_quark.sh llama3.1-8b fp4
```

The helper uses `preprocessed/cnn_dailymail_calibration.pkl` and the controls
in `config/model_prep.env`.

## NVIDIA model

The H200 configuration selects `${MODEL_ROOT}/llama3.1-8b/fp8_dynamic/`.
Generate that local FP8 artifact from the same checkpoint and prepared
calibration input:

```bash
./scripts/model_prep/quantize_llama_quark.sh llama3.1-8b fp8
```

NVIDIA's [v6.0 Llama 3.1 8B procedure](https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA/src/llama3.1-8b/tensorrt)
is the reference for the checkpoint revision and evaluation data. The artifact
selected by this package is the local `fp8_dynamic/` directory above.

## Configure and run

The canonical configuration is `config/model/llama3.1_8b.yaml`. Configure the
shared roots and selected node addresses in `config/deployment.env` as
described in the package README. Its `standalone` section contains the
per-hardware, per-scenario worker settings, and `profiles.dp` contains the
heterogeneous driver settings.

### Standalone

Run these commands from `${WORK_DIR}` in the configured container. The example
uses H200; replace `h200` with `mi350x` to run the MI350X configuration. Start
only one scenario service at a time and run its driver command after all worker
endpoints are ready.

```bash
# Offline
./start_server.sh --role standalone --hardware h200 --model llama3.1_8b --scenario offline
./run.sh llama3.1_8b offline performance --backend standalone --hardware h200 \
  -- harness_config.target_qps=<target-qps>

# Server
./start_server.sh --role standalone --hardware h200 --model llama3.1_8b --scenario server
./run.sh llama3.1_8b server performance --backend standalone --hardware h200 \
  -- harness_config.target_qps=<target-qps>

# Interactive
./start_server.sh --role standalone --hardware h200 --model llama3.1_8b --scenario interactive
./run.sh llama3.1_8b interactive performance --backend standalone --hardware h200 \
  -- harness_config.target_qps=<target-qps>
```

### Heterogeneous data parallel

Set `H200_HOST` and `MI350X_HOST` in the driver's `config/deployment.env` to
the routable addresses of the two worker nodes. Keep `STANDALONE_HOST=127.0.0.1`
in the local worker environment. The worker commands continue to use the base
`standalone` configuration; `--profile dp` is used only by the driver to select
the 16-endpoint mixed-node list.

```bash
# H200 node
./start_server.sh --role standalone --hardware h200 --model llama3.1_8b --scenario <scenario>

# MI350X node
./start_server.sh --role standalone --hardware mi350x --model llama3.1_8b --scenario <scenario>

# Driver node
./run.sh llama3.1_8b <offline|server|interactive> performance \
  --backend standalone --profile dp -- harness_config.target_qps=<target-qps>
```

`profiles.dp.servers.standalone` gives the endpoint order and
`profiles.dp.harness_config.endpoint_weights` gives the corresponding capacity
weights. Keep the endpoint list, `device_count`, and weight count aligned if
the topology is changed. Do not pass `--profile dp` to `start_server.sh`; it is
not a worker option.

## Heterogeneous systems

A heterogeneous deployment can distribute requests with data parallelism or
separate prefill and decode work between accelerator nodes. Data parallelism is
typically a natural fit for Offline, where aggregate sample rate is the primary
objective. For Server and Interactive, separating latency-sensitive work can
help increase throughput while meeting TTFT and TPOT service-level limits.

Because Llama 3.1 8B has low per-token generation work, this model does not
use prefill/decode disaggregation. Its heterogeneous configuration uses
data-parallel request distribution.
## Results capture and TEST06

After a normal run, LoadGen writes performance results to
`results/llama3.1_8b/<scenario>/performance` and AccuracyOnly results to
`results/llama3.1_8b/<scenario>/accuracy`; the complete driver console output
is also retained as `logs/run_llama3.1_8b_<timestamp>.log`. Keep a matched,
`VALID` performance result and AccuracyOnly result for each candidate. To
preserve a named candidate without overwriting the normal location, set
`MLPERF_OUTPUT_DIR` for both runs:

```bash
capture="${WORK_DIR}/results/captures/<tag>/llama3.1_8b/<scenario>"
MLPERF_OUTPUT_DIR="${capture}/performance" \
  ./run.sh llama3.1_8b <offline|server|interactive> performance \
  --backend standalone --hardware <hardware> -- \
  harness_config.target_qps=<target-qps>
MLPERF_OUTPUT_DIR="${capture}/accuracy" \
  ./run.sh llama3.1_8b <offline|server|interactive> accuracy \
  --backend standalone --hardware <hardware>
```

For the heterogeneous data-parallel case, use the same commands with
`--profile dp` and omit the worker-only hardware selection exactly as in the
driver command above. Score the accuracy result in the benchmark container,
then run the required TEST06 verification against those same captured
directories:

```bash
scripts/eval/run_eval_llama31.sh "${capture}/accuracy/mlperf_log_accuracy.json"
scripts/audit/run_test06.sh \
  --model llama3.1_8b --scenario <offline|server|interactive> \
  --accuracy-dir "${capture}/accuracy" \
  --performance-dir "${capture}/performance" --tag <tag>
```

TEST06 does not start a service. It writes its self-contained audit capture to
`results/audits/TEST06/llama3.1_8b/<scenario>/<tag>/`; use its `evidence/`
directory when populating the matching submission `TEST06` result directory.
The evidence must retain the accuracy log, the performance detail and summary
logs, and `verify_accuracy.txt`. See the package README for the exact copy
layout.
