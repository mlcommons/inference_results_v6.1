# Cisco vLLM heterogeneous — MLPerf Inference v6.1

Cisco's vLLM heterogeneous implementation aims to connect GPU systems from any
vendor over Ethernet-based fabrics and use each accelerator for the workload
roles where its hardware is most effective. This v6.1 source package
implements that approach with NVIDIA H200 and AMD MI350X nodes. It uses
portable deployment configuration rather than embedding site-specific paths,
addresses, or fabric names.

This directory contains the harness, model configurations, launch scripts,
container recipes, preparation helpers, and accuracy helpers needed to
reproduce the retained v6.1 results.

## Model guides

The model guides identify where to obtain each model and MLPerf data artifact,
how to prepare the data, and how to produce or select the AMD and NVIDIA model
artifacts used by this package. They also name the corresponding configuration
under `config/model/` and give the scenario-specific service-launch commands.

- [Llama 2 70B](src/models/llama2_70b/README.md)
- [Llama 3.1 8B](src/models/llama3_1_8b/README.md)
- [GPT-OSS 120B](src/models/gptoss_120b/README.md)

## Configure the deployment

Create local deployment and container configuration files, then fill in the
paths, image tags, mounts, and any node or fabric settings required by the
system:

```bash
cp config/deployment.env.example config/deployment.env
cp config/container.env.example config/container.env
```

`config/deployment.env` is read by both `start_server.sh` and `run.sh`. Set
`WORK_DIR` to this package's path inside the container, then set `MODEL_ROOT`
and `DATA_ROOT` to the common in-container roots used by the model guides. For
a local standalone run, set `STANDALONE_HOST=127.0.0.1`. For a heterogeneous
run, set `H200_HOST` and `MI350X_HOST` to routable addresses; the driver must
be able to reach both. Use the same `WORK_DIR`, `MODEL_ROOT`, and `DATA_ROOT`
on every participating node.

`config/container.env` supplies the selected runtime images and the host-side
mounts that make those three roots available in the container. Its
`HOST_MODEL_ROOT` and `HOST_DATA_ROOT` correspond to `MODEL_ROOT` and
`DATA_ROOT` inside the container. Set fabric variables only when the selected
prefill/decode configuration needs them. The local environment files are
intentionally not versioned.

The three canonical model configurations are `config/model/llama2_70b.yaml`,
`config/model/llama3.1_8b.yaml`, and `config/model/gptoss_120b.yaml`. Do not
copy or rename them to make a deployment. Set the environment values above,
then use the model guide's standalone or heterogeneous command block. Those
guides identify the available topology profiles, roles, and scenario commands.

Build the required runtime image or enter an existing configured image on each
node:

```bash
./scripts/docker/build_h200_vllm024_overlay.sh
./scripts/docker/build_mi350x_vllm024_overlay.sh
./start_container.sh --hardware <h200|mi350x>
```

From `${WORK_DIR}` in the container, start exactly one scenario service per
node and wait for its endpoints to become ready. Then execute the matching
`run.sh` command in the model guide. `run.sh` writes the full LoadGen output to
`logs/`; select a target QPS appropriate for the configured system and scenario.

## Results capture and audit evidence

A normal benchmark run writes its LoadGen result directory to
`results/<config>/<scenario>/<performance|accuracy>` and its complete console
log to `logs/run_<config>_<timestamp>.log`. To preserve more than one candidate
without overwriting the default location, choose a capture directory explicitly:

```bash
capture="${WORK_DIR}/results/captures/<tag>/<config>/<scenario>"
MLPERF_OUTPUT_DIR="${capture}/performance" \
  ./run.sh <config> <offline|server|interactive> performance \
  --backend <standalone|pd> --hardware <hardware> -- \
  harness_config.target_qps=<qps>
MLPERF_OUTPUT_DIR="${capture}/accuracy" \
  ./run.sh <config> <offline|server|interactive> accuracy \
  --backend <standalone|pd> --hardware <hardware>
```

Keep the complete capture directory until the result and its audit evidence are
accepted. The `mlperf_log_summary.txt` in a performance run must say `Result is
: VALID`. Score each AccuracyOnly capture in the benchmark container; the
wrappers write `accuracy.txt` beside the supplied accuracy log:

```bash
scripts/eval/run_eval_llama2.sh "${capture}/accuracy/mlperf_log_accuracy.json"
scripts/eval/run_eval_llama31.sh "${capture}/accuracy/mlperf_log_accuracy.json"
scripts/eval/run_eval.sh "${capture}/accuracy/mlperf_log_accuracy.json"
```

Use the first command for Llama 2 70B, the second for Llama 3.1 8B, and the
third for GPT-OSS 120B. Do not use an arbitrary performance run as audit
evidence: retain the run from the same system, configuration, scenario, and
measurement recipe as the result being submitted.

Set `MLPERF_INFERENCE_DIR` in `config/deployment.env` to the release-matched
MLPerf Inference checkout mounted in the container. The audit wrappers use its
official `compliance/` files, stage `audit.config` in a temporary directory,
and leave the ordinary run directory unchanged. They retain these files even
when a package checker does not require all of them:

- `accuracy/mlperf_log_accuracy.json`
- `performance/run_1/mlperf_log_detail.txt`
- `performance/run_1/mlperf_log_summary.txt`
- the corresponding official verifier output

The package currently uses TEST06 for Llama 2 70B and Llama 3.1 8B, and TEST07
and TEST09 for GPT-OSS 120B. No other audit type is used by the retained
results.

### TEST06

TEST06 verifies an existing matching AccuracyOnly and performance capture; it
does not start a benchmark service. Run it in the benchmark container after
both source runs are complete:

```bash
scripts/audit/run_test06.sh \
  --model <llama2_70b|llama3.1_8b> \
  --scenario <offline|server|interactive> \
  --accuracy-dir "${capture}/accuracy" \
  --performance-dir "${capture}/performance" \
  --tag <tag>
```

The wrapper runs MLPerf's official TEST06 verifier and reports `TEST06_RESULT:
PASS` only if verification succeeds. Its self-contained capture is written to
`results/audits/TEST06/<config>/<scenario>/<tag>/`.

### TEST07 and TEST09

TEST07 and TEST09 are GPT-OSS performance audits and must be run against the
same ready service and valid-QPS recipe as the submitted performance result.
Use the same backend, hardware selection, and optional profile used for that
result. TEST07 supplies only the official 990-sample GPQA dataset override;
TEST09 supplies no dataset or token-limit override.

```bash
scripts/audit/run_test07.sh \
  --model gptoss_120b --scenario <offline|server|interactive> \
  --backend <standalone|pd> --hardware <hardware> [--profile <profile>] \
  --target-qps <qps> --tag <tag>

scripts/audit/run_test09.sh \
  --model gptoss_120b --scenario <offline|server|interactive> \
  --backend <standalone|pd> --hardware <hardware> [--profile <profile>] \
  --target-qps <qps> --tag <tag>
```

The wrappers reject GPT-OSS configurations that do not retain performance
`min_tokens=1`, `max_tokens=10000`, and
`strip_output_special_tokens=false`. TEST07 also checks that the official GPQA
input exists. Each wrapper invokes the official verifier and reports `PASS`
only when both LoadGen and the verifier pass. Captures are written to
`results/audits/TEST07/...` and `results/audits/TEST09/...`; the verifier
outputs are respectively `evidence/verify_accuracy.txt` and
`evidence/verify_output_len.txt`.

### Copying accepted evidence into the submission tree

After an audit wrapper returns `PASS`, copy its `evidence/` contents into the
matching canonical submission test directory. Preserve the exact directory
names and case used by the submission layout:

```bash
capture="${WORK_DIR}/results/audits/TEST0X/<config>/<scenario>/<tag>"
destination="<submission-root>/closed/cisco/results/<system>/<MLPerf-model>/<Offline|Server|Interactive>/TEST0X"
mkdir -p "${destination}"
cp -a "${capture}/evidence/." "${destination}/"
```

The destination must contain the three required MLPerf logs and the verifier
output above. Keep the adjacent `raw/`, `verification/`, `driver.log`,
`audit.config`, configuration snapshot, and manifest in the capture directory
as the reproducibility record; only the required evidence belongs beneath the
submission result's `TEST0X` directory.
