# GPT-OSS 120B preparation

The canonical configuration is `config/model/gptoss_120b.yaml`. Set
`MODEL_ROOT` and `DATA_ROOT` in `config/deployment.env`; the paths below are
the only model and data paths consumed by that configuration.

## Data

Follow NVIDIA's [v6.0 GPT-OSS 120B procedure](https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA/src/gpt-oss-120b/tensorrt)
for the release-matched prepared MLPerf data bundle and the required accuracy
and compliance inputs. The harness requires this final layout:

```text
${DATA_ROOT}/gpt-oss-120b/
├── perf/
│   ├── input_ids_padded.npy
│   └── input_lens.npy
└── acc/
    ├── input_ids_padded.npy
    ├── input_lens.npy
    ├── acc_eval_ref.parquet
    └── acc_eval_compliance_gpqa.parquet
```

`perf/` contains the 6,396 performance inputs and `acc/` contains the 4,395
accuracy inputs. Retain `acc_eval_compliance_gpqa.parquet` for TEST07. Do not
substitute raw AIME, GPQA, or LiveCodeBench data, or make an ad hoc sample
selection.

## Data preprocessing

This package has no GPT-OSS data-conversion script. It consumes the prepared
arrays and parquet references from the NVIDIA procedure directly. Preserve
their filenames, input order, tokenization, and sample counts; TEST07
materializes its compliance input from the approved GPQA reference in `acc/`.

## AMD model

Download OpenAI's [GPT-OSS 120B release](https://huggingface.co/openai/gpt-oss-120b)
and place the released native MXFP4 checkpoint and tokenizer at
`${MODEL_ROOT}/gpt-oss-120b/model/`. GPT-OSS is released with MXFP4 MoE
weights. The MI350X configuration selects this directory directly; its
`quantization: quark` setting is the vLLM loader setting for the released
model, not an instruction to create a new artifact. No local GPT-OSS
quantization command is required.

AMD's [v6.0 GPT-OSS source](https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/AMD/src/gpt-oss-120b)
is the vendor reference for the MI350X path.

## NVIDIA model

The H200 configuration selects the same released native MXFP4 model directory,
`${MODEL_ROOT}/gpt-oss-120b/model/`; it does not use a locally converted FP4
model. If a selected profile enables speculative decoding, place its
release-matched EAGLE3 model at `${MODEL_ROOT}/gpt-oss-120b/eagle3/`.

NVIDIA's [v6.0 GPT-OSS procedure](https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA/src/gpt-oss-120b/tensorrt)
is the vendor reference for the H200 path and the required MLPerf data
preparation.

## Configure and run

The canonical configuration is `config/model/gptoss_120b.yaml`. Configure the
shared roots and selected node addresses in `config/deployment.env` as
described in the package README. The `standalone`, `prefiller`, and `decoder`
sections contain the worker controls. Its `pd` profile is selected
automatically by `run.sh` when `--backend pd` is used.

### Standalone

Run these commands from `${WORK_DIR}` in the configured container. The example
uses H200; replace `h200` with `mi350x` to run the MI350X configuration. Start
only one scenario service at a time and run its driver command after all worker
endpoints are ready.

```bash
# Offline
./start_server.sh --role standalone --hardware h200 --model gptoss_120b --scenario offline
./run.sh gptoss_120b offline performance --backend standalone --hardware h200 \
  -- harness_config.target_qps=<target-qps>

# Server
./start_server.sh --role standalone --hardware h200 --model gptoss_120b --scenario server
./run.sh gptoss_120b server performance --backend standalone --hardware h200 \
  -- harness_config.target_qps=<target-qps>

# Interactive
./start_server.sh --role standalone --hardware h200 --model gptoss_120b --scenario interactive
./run.sh gptoss_120b interactive performance --backend standalone --hardware h200 \
  -- harness_config.target_qps=<target-qps>
```

GPT-OSS Interactive is configured for local use but is not part of the
retained v6.1 submission.

### Heterogeneous prefill/decode

Set `H200_HOST` and `MI350X_HOST` in `config/deployment.env` to routable node
addresses before starting the services. Start the worker roles on their
respective nodes, then run the driver from a node that can reach both address
sets:

```bash
# H200 prefill node
./start_server.sh --role prefill --hardware h200 --model gptoss_120b --scenario <scenario>

# MI350X decode node
./start_server.sh --role decode --hardware mi350x --remote-hardware h200 \
  --model gptoss_120b --scenario <scenario>

# Driver node
./run.sh gptoss_120b <offline|server|interactive> performance \
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
## Results capture, scoring, TEST07, and TEST09

After a normal run, LoadGen writes performance results to
`results/gptoss_120b/<scenario>/performance` and AccuracyOnly results to
`results/gptoss_120b/<scenario>/accuracy`; the complete driver console output
is also retained as `logs/run_gptoss_120b_<timestamp>.log`. Keep a matched,
`VALID` performance result and AccuracyOnly result for each candidate. To
preserve a named candidate without overwriting the normal location, set
`MLPERF_OUTPUT_DIR` for both runs:

```bash
capture="${WORK_DIR}/results/captures/<tag>/gptoss_120b/<scenario>"
MLPERF_OUTPUT_DIR="${capture}/performance" \
  ./run.sh gptoss_120b <offline|server|interactive> performance \
  --backend <standalone|pd> --hardware <hardware> -- \
  harness_config.target_qps=<target-qps>
MLPERF_OUTPUT_DIR="${capture}/accuracy" \
  ./run.sh gptoss_120b <offline|server|interactive> accuracy \
  --backend <standalone|pd> --hardware <hardware>
scripts/eval/run_eval.sh "${capture}/accuracy/mlperf_log_accuracy.json"
```

Run TEST07 and TEST09 against the same ready service and valid-QPS recipe as
the submitted performance point. Pass the same backend, hardware, and optional
profile used by that point; TEST07 applies only the official GPQA input and
TEST09 applies no data or token-limit override:

```bash
scripts/audit/run_test07.sh \
  --model gptoss_120b --scenario <offline|server|interactive> \
  --backend <standalone|pd> --hardware <hardware> [--profile <profile>] \
  --target-qps <target-qps> --tag <tag>
scripts/audit/run_test09.sh \
  --model gptoss_120b --scenario <offline|server|interactive> \
  --backend <standalone|pd> --hardware <hardware> [--profile <profile>] \
  --target-qps <target-qps> --tag <tag>
```

These wrappers launch the audited performance runs and write complete captures
to `results/audits/TEST07/gptoss_120b/<scenario>/<tag>/` and
`results/audits/TEST09/gptoss_120b/<scenario>/<tag>/`. Copy each `evidence/`
directory into its matching submission test directory only after the wrapper
reports `PASS`. Each one must retain the audit accuracy log, performance detail
and summary logs, and its verifier output (`verify_accuracy.txt` for TEST07 or
`verify_output_len.txt` for TEST09). See the package README for the exact copy
layout.
