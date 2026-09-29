# Llama 2 70B preparation

The canonical configuration is `config/model/llama2_70b.yaml`. Set
`MODEL_ROOT` and `DATA_ROOT` in `config/deployment.env`; the paths below are
the only model and data paths consumed by that configuration.

## Data

Follow the upstream [MLPerf Llama 2 dataset procedure](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md#get-dataset),
as used by NVIDIA's [v6.0 Llama 2 procedure](https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA/src/llama2-70b-99.9/tensorrt),
to obtain these release-matched pre-tokenized OpenOrca files:

- `open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz`
- `open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz`

Place their decompressed forms in `${DATA_ROOT}/llama2-70b/`:

```text
${DATA_ROOT}/llama2-70b/
├── open_orca_gpt4_tokenized_llama.sampled_24576.pkl
└── open_orca_gpt4_tokenized_llama.calibration_1000.pkl
```

The sampled file is used for performance and accuracy. The calibration file is
used only to create the local quantized artifacts.

## Data preprocessing

The sampled OpenOrca data is already tokenized. Decompress the two source files
without changing their contents, order, tokenization, or sample selection:

```bash
gzip -dk open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
gzip -dk open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz
```

This package reads the decompressed sampled `pkl` directly. Do not retokenize
it or substitute TensorRT-specific preprocessed arrays.

## AMD model

Download the licensed Llama 2 70B Chat checkpoint using AMD's
[v6.0 Llama 2 source](https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/AMD/src/llama2-70b-99).
Place the base checkpoint at `${MODEL_ROOT}/llama2-70b/`: its `config.json`
and weights are at that root, and the canonical worker uses the tokenizer in
`${MODEL_ROOT}/llama2-70b/orig/`. The MI350X configuration selects the local
MXFP4 artifact `${MODEL_ROOT}/llama2-70b/fp4_quantized_gptq/`.

Create `config/model_prep.env` from its example, build the packaged Quark image,
and generate the artifact:

```bash
cp config/model_prep.env.example config/model_prep.env
./scripts/docker/build_mi350x_quark_prep.sh
./scripts/model_prep/quantize_llama_quark.sh llama2-70b fp4
```

The helper uses the calibration file above and the sequence length and sample
count in `config/model_prep.env`.

## NVIDIA model

The H200 configuration selects `${MODEL_ROOT}/llama2-70b/fp8_dynamic/`.
Generate that local FP8 artifact from the same licensed checkpoint and
calibration data:

```bash
./scripts/model_prep/quantize_llama_quark.sh llama2-70b fp8
```

NVIDIA's [v6.0 Llama 2 procedure](https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA/src/llama2-70b-99.9/tensorrt)
is the reference for the checkpoint and MLPerf data. Its published NVFP4
artifact is for the B300 override and is not selected by the H200+MI350X
configuration in this package.

## Configure and run

The canonical configuration is `config/model/llama2_70b.yaml`. Configure the
shared roots and selected node addresses in `config/deployment.env` as
described in the package README. The scenario-specific worker settings are in
the `standalone`, `prefiller`, and `decoder` sections of that file; the
heterogeneous data-parallel driver settings are in `profiles.dp`.

### Standalone

Run these commands from `${WORK_DIR}` in the configured container. The example
uses H200; replace `h200` with `mi350x` to run the MI350X configuration. Start
only one scenario service at a time and run its driver command after all worker
endpoints are ready.

```bash
# Offline
./start_server.sh --role standalone --hardware h200 --model llama2_70b --scenario offline
./run.sh llama2_70b offline performance --backend standalone --hardware h200 \
  -- harness_config.target_qps=<target-qps>

# Server
./start_server.sh --role standalone --hardware h200 --model llama2_70b --scenario server
./run.sh llama2_70b server performance --backend standalone --hardware h200 \
  -- harness_config.target_qps=<target-qps>

# Interactive
./start_server.sh --role standalone --hardware h200 --model llama2_70b --scenario interactive
./run.sh llama2_70b interactive performance --backend standalone --hardware h200 \
  -- harness_config.target_qps=<target-qps>
```

### Heterogeneous data parallel

Set `H200_HOST` and `MI350X_HOST` in the driver's `config/deployment.env` to
the routable addresses of the two worker nodes. Keep `STANDALONE_HOST=127.0.0.1`
in the local worker environment. The base `standalone` configuration launches
the workers; `--profile dp` is a driver-only selection that supplies the
16-endpoint mixed-node list.

```bash
# H200 node
./start_server.sh --role standalone --hardware h200 --model llama2_70b --scenario <scenario>

# MI350X node
./start_server.sh --role standalone --hardware mi350x --model llama2_70b --scenario <scenario>

# Driver node
./run.sh llama2_70b <offline|server|interactive> performance \
  --backend standalone --profile dp -- harness_config.target_qps=<target-qps>
```

`profiles.dp.<scenario>.servers.standalone` defines the endpoint order, and
`profiles.dp.<scenario>.harness_config.endpoint_weights` contains one capacity
weight for each endpoint. If the endpoint list is changed, keep its order,
`device_count`, and the weight list consistent. Do not pass `--profile dp` to
`start_server.sh`; it is not a worker option.

### Heterogeneous prefill/decode

The base configuration also defines a prefill/decode topology. Set both host
variables to routable addresses on every participating node, start the roles
on their respective nodes, then start the driver:

```bash
# H200 prefill node
./start_server.sh --role prefill --hardware h200 --model llama2_70b --scenario <scenario>

# MI350X decode node
./start_server.sh --role decode --hardware mi350x --remote-hardware h200 \
  --model llama2_70b --scenario <scenario>

# Driver node
./run.sh llama2_70b <offline|server|interactive> performance \
  --backend pd -- harness_config.target_qps=<target-qps>
```

## Heterogeneous systems

A heterogeneous deployment can distribute requests with data parallelism or
separate prefill and decode work between accelerator nodes. Data parallelism is
typically a natural fit for Offline, where aggregate sample rate is the primary
objective. For Server and Interactive, separating latency-sensitive work can
help increase throughput while meeting TTFT and TPOT service-level limits. The
available topology profiles and role settings are defined in this model's
configuration.
## Results capture and TEST06

After a normal run, LoadGen writes performance results to
`results/llama2_70b/<scenario>/performance` and AccuracyOnly results to
`results/llama2_70b/<scenario>/accuracy`; the complete driver console output is
also retained as `logs/run_llama2_70b_<timestamp>.log`. Keep a matched,
`VALID` performance result and AccuracyOnly result for each candidate. To
preserve a named candidate without overwriting the normal location, set
`MLPERF_OUTPUT_DIR` for both runs:

```bash
capture="${WORK_DIR}/results/captures/<tag>/llama2_70b/<scenario>"
MLPERF_OUTPUT_DIR="${capture}/performance" \
  ./run.sh llama2_70b <offline|server|interactive> performance \
  --backend <standalone|pd> --hardware <hardware> -- \
  harness_config.target_qps=<target-qps>
MLPERF_OUTPUT_DIR="${capture}/accuracy" \
  ./run.sh llama2_70b <offline|server|interactive> accuracy \
  --backend <standalone|pd> --hardware <hardware>
```

Score the accuracy result in the benchmark container, then run the required
TEST06 verification against those same captured directories:

```bash
scripts/eval/run_eval_llama2.sh "${capture}/accuracy/mlperf_log_accuracy.json"
scripts/audit/run_test06.sh \
  --model llama2_70b --scenario <offline|server|interactive> \
  --accuracy-dir "${capture}/accuracy" \
  --performance-dir "${capture}/performance" --tag <tag>
```

TEST06 does not start a service. It writes its self-contained audit capture to
`results/audits/TEST06/llama2_70b/<scenario>/<tag>/`; use its `evidence/`
directory when populating the matching submission `TEST06` result directory.
The evidence must retain the accuracy log, the performance detail and summary
logs, and `verify_accuracy.txt`. See the package README for the exact copy
layout.
