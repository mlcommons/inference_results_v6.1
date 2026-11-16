# DeepSeek-R1

The Vera Rubin NVL72 submission (`VR-NVL72_VR-288GB_aarch64x72`) serves DeepSeek-R1 with TensorRT-LLM disaggregated
prefill/decode workers behind NVIDIA Dynamo frontends, drives it with the MLPerf Endpoints client and orchestrates it
with nv-sflow on Slurm + Enroot.

## Support Matrix

See [Docker support](../../../../configs/DOCKER_SUPPORT.md) and [SLURM support](../../../../configs/SLURM_SUPPORT.md) for per-system, per-scenario support details.

## Getting Started

Choose a directory on a filesystem that the Slurm login and compute nodes share. The model and the dataset go under
it; every container mounts it at the same path. Clone this repository on a shared filesystem too: the workflow mounts
`closed/NVIDIA` (`WORK_DIR`) into every container and runs its tools from there.

```bash
export SCRATCH_DIR=<shared directory>
```

### Download Model

Download the quantized FP4 checkpoint:

```bash
export CHECKPOINT_PATH=${SCRATCH_DIR}/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4-v2
git lfs install

git clone https://huggingface.co/centml/DeepSeek-R1-NVFP4-v2-mlpinf ${CHECKPOINT_PATH}
git -C ${CHECKPOINT_PATH} checkout 93947a0d7bd04f73ff98636f6b18ff5839e7aaf9
```

#### Checkpoint provenance & reproduction

The FP4 checkpoint is `deepseek-ai/DeepSeek-R1` quantized to NVFP4 with
[NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
using its official `examples/deepseek` recipe: reshard to model-parallel-8,
a single calibration sweep recording per-tensor activation `amax`
(`NVFP4_DEFAULT_CFG`), then a one-shot FP8 → NVFP4 weight conversion. Scheme:
MLP/MoE expert linears + `attn.wo` in NVFP4 (group-16 block scaling), MLA
projections in bf16. The Vera Rubin configs store the MLA KV cache in 4 bits
(`kv_cache_config.dtype: nvfp4` with the MXFP4-2D layout, `TRTLLM_FP4_MLA_MXFP4_2D=1`).

This checkpoint was built with the **MLPerf deepseek-r1 calibration dataset**
(500 samples). Exact Model-Optimizer/DeepSeek-V3 commit pins, the calibration
override, and a runnable end-to-end script (`quantize_dsr1_mlperf_calib.py`) are
documented in
[`scripts/dsr1_nvfp4_ptq/`](../../../../scripts/dsr1_nvfp4_ptq/README.md).

### Download and Prepare Data

Download the evaluation dataset following the MLCommons README and convert it to the parquet file that the MLPerf
Endpoints client reads (the same file serves the performance and the accuracy runs). Run it in the directory that holds
the `.pkl`, with Python 3, `pandas` and `pyarrow` (`pip install pandas pyarrow`):

```bash
# Download from: https://github.com/mlcommons/inference/tree/master/language/deepseek-r1/README.md
# - mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl
mkdir -p ${SCRATCH_DIR}/data/deepseek-r1-endpoints
python3 - ${SCRATCH_DIR}/data/deepseek-r1-endpoints/deepseek_r1_eval.parquet <<'EOF'
import sys
import pandas as pd
df = pd.read_pickle("mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl")
pd.DataFrame({"input_tokens": [list(map(int, t)) for t in df["tok_input"]], "ground_truth": df["ground_truth"],
              "dataset": df["dataset"], "question": df["question"]}).to_parquet(sys.argv[1], index=False)
EOF
```

**Verify the following files exist:**

1. Model: `${SCRATCH_DIR}/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4-v2/`
2. Data: `${SCRATCH_DIR}/data/deepseek-r1-endpoints/deepseek_r1_eval.parquet` (4388 rows; the file used for the
   results has sha256 `36facfae18f42587b7e4863e16a0e96446470761017d207791e609537d0f6f02`, which depends on the
   pandas/pyarrow writer, so compare contents if it differs)

## Base Image

| Task | Arch | Image |
| ---- | ---- | ----- |
| Server (TensorRT-LLM + NVIDIA Dynamo) | aarch64 | `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/v6.1-rubin-available-submission:DSR1-VR-NVL72-dsr1_26.08` |
| MLPerf Endpoints client | aarch64 | `ghcr.io/mlcommons/endpoints:598dbfde` |
| LiveCodeBench grader | aarch64 | `ghcr.io/mlcommons/endpoints:9d749906-livecodebench` |

## Run the Benchmark in Multi Node using Slurm + Enroot

This section covers the multi-node SLURM work through `nv-sflow`. Check [SLURM support](../../../../configs/SLURM_SUPPORT.md) before choosing a system/scenario.

Each scenario runs on one NVL72 rack: 18 nodes × 4 GPUs in one NVLink domain, allocated with `--segment=18`
(set in `slurm_env_sflow.yaml`). Four 18-GPU prefill/decode pairs serve the model; all 72 GPUs are the system under
test. The workflow pins the HBM clock to 4752 MHz on every node before the servers start (`MEM_CLOCK_MHZ`, needs
passwordless `sudo nvidia-smi -lmc` on the compute nodes; add `--set MEM_CLOCK_MHZ=0` if the clocks are pinned
otherwise), then runs the performance phase followed by the accuracy phase (`MODE=both`). The clocks are released
after the client finishes; after a failed or cancelled job, run `sudo nvidia-smi -rmc` on its nodes.

### Prepare the images for multi node with Slurm + Enroot

Import the prebuilt partner server, client, and grader images once on an aarch64 node. Configure Enroot registry
authentication for `registry.gitlab.com` first; access to the partner project and `read_registry` permission are required.
See the [partner release notes](../../../../../../release_notes/v6.1-rubin-available-submission.md) for setup and launch instructions.
Source rebuilds are documented in [docker/vr-dsr1-trtllm-dynamo](../../../../docker/vr-dsr1-trtllm-dynamo/README.md).

```bash
mkdir -p ${SCRATCH_DIR}/images
enroot import -o ${SCRATCH_DIR}/images/dsr1_rubin_trtllm_dynamo.sqsh \
  'docker://registry.gitlab.com#nvidia/mlperf-inference-partner/nv-mlpinf-partner/v6.1-rubin-available-submission:DSR1-VR-NVL72-dsr1_26.08'
enroot import -o ${SCRATCH_DIR}/images/endpoints_598dbfde.sqsh 'docker://ghcr.io#mlcommons/endpoints:598dbfde'
enroot import -o ${SCRATCH_DIR}/images/endpoints_lcb.sqsh 'docker://ghcr.io#mlcommons/endpoints:9d749906-livecodebench'
```

### Install nv-sflow

The commands below use nv-sflow `v0.3.1` on the login node; each job installs the same version on its compute nodes
(`--sflow-version`), so the compute nodes need access to github.com.

```bash
python3 -m venv ${HOME}/sflow-venv
source ${HOME}/sflow-venv/bin/activate
pip install "sflow @ git+https://github.com/NVIDIA/nv-sflow.git@v0.3.1"
```

### Common nv-sflow settings

Run from `closed/NVIDIA`. Replace `ACCT`, `PARTITION` and the image paths for your cluster.

```bash
ACCT=<your-slurm-account>
PARTITION=<your-partition>
SYSTEM=VR-NVL72_VR-288GB_aarch64x72
NODES=18
SERVER_IMAGE=${SCRATCH_DIR}/images/dsr1_rubin_trtllm_dynamo.sqsh
CLIENT_IMAGE=${SCRATCH_DIR}/images/endpoints_598dbfde.sqsh
LCB_IMAGE=${SCRATCH_DIR}/images/endpoints_lcb.sqsh
TEMPLATE=dynamo_disagg_rack18_endpoints_dsr1_perf_acc.yaml
SFLOW_DUMP_DIR=build/sbatch_scripts_sflow
mkdir -p "$SFLOW_DUMP_DIR"
```

### Offline, Server and Interactive performance and accuracy

Set `SCENARIO=Offline`, `Server` or `Interactive`:

```bash
SCENARIO=Offline
RUN=deepseek_r1_${SYSTEM}_${SCENARIO}_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f configs/deepseek_r1/${SYSTEM}/${SCENARIO}/deepseek_config_sflow.yaml \
  -f configs/deepseek_r1/${SYSTEM}/${SCENARIO}/slurm_env_sflow.yaml \
  -f scaleout/sflow/templates/${TEMPLATE} \
  --set WORK_DIR=$PWD \
  --set SCRATCH_DIR=$SCRATCH_DIR \
  --set CONTAINER_IMAGE=$SERVER_IMAGE \
  --set ENDPOINT_CONTAINER_IMAGE=$CLIENT_IMAGE \
  --set LCB_CONTAINER_IMAGE=$LCB_IMAGE \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  --time=02:30:00 \
  --sflow-version v0.3.1 \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
```

An Offline job takes about 1.5 hours, Server and Interactive about 1 hour. Add `--set MODE=perf` or `--set MODE=acc`
to run one phase only. The client writes its report to `sflow_output/<job>/endpoint_benchmark/`:

| File | Content |
| --- | --- |
| `config.yaml` | client configuration as run |
| `performance/result_summary.json` | `tps`, TTFT / TPOT percentiles (`early_stopping_percentiles`), sample counts, duration |
| `accuracy/accuracy_results.json` | `average_accuracy` (exact_match), output lengths |
| `report.txt` | human-readable report |

The throughput metric is `tps` (Server and Interactive: legacy LoadGen definition); tokens/s/GPU = `tps / 72`.

## Benchmark Passing Criteria

A run is valid only if the required accuracy and performance criteria pass. For
Server and Interactive scenarios, both TTFT and TPOT latency must be within the
threshold; reduce target QPS and rerun if either latency metric fails.

| Scenario | Accuracy criteria | Performance criteria |
| --- | --- | --- |
| Offline | `exact_match >= 0.99 * 81.3582 = 80.544618`; `TOKENS_PER_SAMPLE >= 0.9 * 3886.2274 = 3497.60466` | No TTFT/TPOT constraint |
| Server | `exact_match >= 0.99 * 81.3582 = 80.544618`; `TOKENS_PER_SAMPLE >= 0.9 * 3886.2274 = 3497.60466` | TTFT <= 2 s; TPOT <= 80 ms |
| Interactive | `exact_match >= 0.99 * 81.3582 = 80.544618`; `TOKENS_PER_SAMPLE >= 0.9 * 3886.2274 = 3497.60466` | TTFT <= 1.5 s; TPOT <= 15 ms |

Every issued sample must complete (`n_samples_completed == n_samples_issued`, `n_samples_failed == 0` in
`result_summary.json`).
