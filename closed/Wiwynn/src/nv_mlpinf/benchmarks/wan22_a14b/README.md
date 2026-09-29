# Wan 2.2 A14B - Text-to-Video (sflow + inference-endpoints)

End-to-end MLPerf video-generation pipeline for **Wan 2.2 A14B** on Blackwell
systems (B200x8, B300x8, GB200-NVL72, GB300-NVL72). The server and the load
generator (SUT) are decoupled, following the same sflow convention as the other
endpoint-based benchmarks:

- **Server:** TRT-LLM VisualGen (`trtllm-serve … --visual_gen_args`) exposing
  `POST /v1/videos/generations`.
- **Client:** the `inference-endpoint` load generator (`api_type: videogen`).
- **Orchestration:** sflow templates in
  `configs/wan22_a14b/_shared/templates/`; shared runtime tools in
  `scaleout/sflow/tools/`; image build recipes + Dockerfiles in
  `src/nv_mlpinf/benchmarks/wan22_a14b/{scripts,docker}/`.

## Quick start

All commands run from `closed/Wiwynn`. Set these variables once and reuse
them throughout:

```bash
cd closed/Wiwynn
WORK_DIR=$PWD
SCRATCH_DIR=/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone  # shared storage visible to all nodes
SFLOW_VENV=/lustre/fsw/coreai_mlperf_inference/$USER/.venv-sflow

# x86_64 only (B300-SXM-270GBx8, B200-SXM-180GBx8):
# Path to a static Linux x86_64 ffmpeg binary. The x86 server image encodes
# videos as H.264 mp4 only when ffmpeg is present; without it the AVI fallback
# breaks VBench scoring. Download a static build once and point FFMPEG_BIN at it.
FFMPEG_BIN=/path/to/static/ffmpeg
```

### Set up sflow (one-time, on the SLURM login node)

Every launch command below runs `sflow batch` from a Python venv with `sflow`
installed. Create it once and reuse it. New to nv-sflow? See
[scaleout/sflow/README.md](../../../../scaleout/sflow/README.md).

```bash
# From closed/Wiwynn (uv: https://docs.astral.sh/uv/getting-started/installation/)
uv venv
uv pip install "sflow @ git+https://github.com/NVIDIA/nv-sflow.git@main"
source .venv/bin/activate
```

The command blocks below re-run `source .venv/bin/activate` so each is
copy-paste-runnable on its own.

### Step 1 - Stage data

```bash
# Download the FP8 checkpoint
huggingface-cli download centml/wan2.2-t2v-a14b-diffusers-mlpinf-fp8 \
  --local-dir "$SCRATCH_DIR/models/wan22-a14b/Wan2.2-T2V-A14B-Diffusers-FP8"

# Copy the fixed initial latent (ships in-repo, not in the HF checkpoint)
mkdir -p "$SCRATCH_DIR/preprocessed_data/wan22-a14b"
cp 3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/fixed_latent.pt \
   "$SCRATCH_DIR/preprocessed_data/wan22-a14b/fixed_latent.pt"
```

#### Pre-stage the VBench cache (accuracy runs)

Accuracy runs only. The `-vbench` client image ships the VBench venv but **none**
of the 6 dimension weights, so pre-stage them into `build/vbench_cache`. Run once
on a login node (needs curl/wget + unzip + git + network); ~5 GB. Perf-only and
TEST04 audit runs do not need this.

```bash
src/nv_mlpinf/benchmarks/wan22_a14b/scripts/prepare_vbench_cache.sh   # -> build/vbench_cache
```

### Step 2 - Pull the images

Pre-built images are wired into `configs/wan22_a14b/_shared/slurm_env.yaml` as
the defaults. The x86_64 server image must be set via `--set CONTAINER_IMAGE=...`
(see the examples below).

| Image | Tag |
|-------|-----|
| Server (aarch64) | `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/wan22-trtllm-server:v6.1-jun26-wan22-trtllm-aarch64` |
| Server (x86_64) | `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/wan22-trtllm-server:v6.1-jul10-wan22-trtllm-x86_64` |
| Client (all arches) | `ghcr.io/mlcommons/endpoints:598dbfde68ebfa42256762ebb60fd4da227b813c-vbench` |

Pyxis/enroot pulls `docker://` registry URLs directly. For the private GitLab
registry, add credentials to `~/.config/enroot/.credentials`:

```
machine registry.gitlab.com login <your-gitlab-user> password <your-token>
```

### Step 3 - Submission Run

A full submission is two commands per scenario: first perf+accuracy, then the
TEST04 compliance audit. The TEST04 audit is self-contained (it runs its own
reference and fixed-sample phases in one job), so it does not consume the perf
run's output; running it second is a submission-packaging convention, not a data
dependency.

#### Command 1 - Performance + accuracy

<details>
<summary>GB300-NVL72 x72 (18 nodes, Offline) - aarch64</summary>

```bash
source $SFLOW_VENV/bin/activate
sflow batch \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/GB300-NVL72_GB300-288GB_aarch64x72/visual_gen/Offline/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_submission.yaml \
  --set WORK_DIR=$WORK_DIR \
  --set SCRATCH_DIR=$SCRATCH_DIR \
  --set NUM_SERVERS=72 \
  --nodes=18 --partition=gb300 --account=coreai_mlperf_inference --time=05:00:00 \
  --job-name=wan22-gb300x72-offline-acc \
  -o build/sbatch_scripts_sflow/wan22_gb300x72_offline_acc.sh
deactivate
sbatch build/sbatch_scripts_sflow/wan22_gb300x72_offline_acc.sh
```
</details>

<details>
<summary>GB300-NVL72 x72 (18 nodes, SingleStream) - aarch64</summary>

```bash
source $SFLOW_VENV/bin/activate
sflow batch \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/GB300-NVL72_GB300-288GB_aarch64x72/visual_gen/SingleStream/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_submission.yaml \
  --set WORK_DIR=$WORK_DIR \
  --set SCRATCH_DIR=$SCRATCH_DIR \
  --set NUM_SERVERS=1 \
  --nodes=18 --partition=gb300 --account=coreai_mlperf_inference --time=05:00:00 \
  --job-name=wan22-gb300x72-ss-acc \
  -o build/sbatch_scripts_sflow/wan22_gb300x72_ss_acc.sh
deactivate
sbatch build/sbatch_scripts_sflow/wan22_gb300x72_ss_acc.sh
```
</details>

<details>
<summary><code>B300-SXM-270GBx8</code> (1 node, Offline) - x86_64</summary>

```bash
source $SFLOW_VENV/bin/activate
sflow batch \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/B300-SXM-270GBx8/visual_gen/Offline/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_submission.yaml \
  --set WORK_DIR=$WORK_DIR \
  --set SCRATCH_DIR=$SCRATCH_DIR \
  --set CONTAINER_IMAGE="docker://registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/wan22-trtllm-server:v6.1-jul10-wan22-trtllm-x86_64" \
  "--set CONTAINER_MOUNTS=$WORK_DIR:/work,$SCRATCH_DIR:/home/mlperf_inference_storage,$WORK_DIR/ci/scripts/smoke_test/patches/videogen_adapter.py:/opt/venv/lib/python3.12/site-packages/inference_endpoint/videogen/adapter.py,$WORK_DIR/ci/scripts/smoke_test/patches/visual_gen_executor.py:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/visual_gen/executor.py,$FFMPEG_BIN:/usr/local/bin/ffmpeg" \
  --nodes=1 --partition=batch --account=coreai_mlperf_inference --time=05:00:00 \
  --job-name=wan22-b300x8-offline-acc \
  -o build/sbatch_scripts_sflow/wan22_b300x8_offline_acc.sh
deactivate
sbatch build/sbatch_scripts_sflow/wan22_b300x8_offline_acc.sh
```
</details>

<details>
<summary><code>B300-SXM-270GBx8</code> (1 node, SingleStream) - x86_64</summary>

```bash
source $SFLOW_VENV/bin/activate
sflow batch \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/B300-SXM-270GBx8/visual_gen/SingleStream/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_submission.yaml \
  --set WORK_DIR=$WORK_DIR \
  --set SCRATCH_DIR=$SCRATCH_DIR \
  --set CONTAINER_IMAGE="docker://registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/wan22-trtllm-server:v6.1-jul10-wan22-trtllm-x86_64" \
  "--set CONTAINER_MOUNTS=$WORK_DIR:/work,$SCRATCH_DIR:/home/mlperf_inference_storage,$WORK_DIR/ci/scripts/smoke_test/patches/videogen_adapter.py:/opt/venv/lib/python3.12/site-packages/inference_endpoint/videogen/adapter.py,$WORK_DIR/ci/scripts/smoke_test/patches/visual_gen_executor.py:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/visual_gen/executor.py,$FFMPEG_BIN:/usr/local/bin/ffmpeg" \
  --nodes=1 --partition=batch --account=coreai_mlperf_inference --time=05:00:00 \
  --job-name=wan22-b300x8-ss-acc \
  -o build/sbatch_scripts_sflow/wan22_b300x8_ss_acc.sh
deactivate
sbatch build/sbatch_scripts_sflow/wan22_b300x8_ss_acc.sh
```
</details>

> **GPU count:** an x72 run uses exactly **72 GPUs (18 nodes)**, all servers
> (Offline: 72 single-GPU replicas; SS: 1 replica across 72). The VBench scorer is
> **not** a 73rd GPU: the endpoint task requests no `resources.gpus` and
> time-shares a server GPU freed after generation. A static `ENDPOINT_GPUS=1`
> would need 72+1 > 72, which sflow rejects. Audit uses `no_vbench` (no scorer GPU).
>
> **Offline VBench weights:** the `598dbfde` image ships the VBench venv but
> **none** of the 6 dim weights, so `prepare_vbench_cache.sh` (Step 1) stages them
> into `build/vbench_cache`. That lives under `WORK_DIR` (mounted at `/work`), so
> it is reachable at `/work/build/vbench_cache` with no extra mount. The template
> points `VBENCH_CACHE_DIR`/`TORCH_HOME`/`HOME` there.

#### Command 2 - Compliance (MLPerf TEST04)

TEST04 (output-caching detection) runs standalone after Command 1. Point
`ENDPOINT_CONFIG` at the audit config. The audit uses the `no_vbench` template and
does no VBench scoring, so it needs no scorer GPU or weight cache.

The endpoint client runs TEST04 natively: a reference phase (distinct prompts)
followed by a fixed-sample phase (one prompt repeated), then writes
`audit/audit_result.json` and `verify_OUTPUT_CACHING_TEST.txt`.
Valid iff `audit_qps < ref_qps × (1 + threshold)`.

<details>
<summary>GB300-NVL72 x72 (18 nodes, Offline) - aarch64</summary>

```bash
source $SFLOW_VENV/bin/activate
sflow batch \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/GB300-NVL72_GB300-288GB_aarch64x72/visual_gen/Offline/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_no_vbench.yaml \
  --set WORK_DIR=$WORK_DIR \
  --set SCRATCH_DIR=$SCRATCH_DIR \
  --set NUM_SERVERS=72 \
  --set ENDPOINT_CONFIG=configs/wan22_a14b/_shared/endpoint_audit_test04_offline_x72.yaml \
  --nodes=18 --partition=gb300 --account=coreai_mlperf_inference --time=05:00:00 \
  --job-name=wan22-gb300x72-offline-test04 \
  -o build/sbatch_scripts_sflow/wan22_gb300x72_offline_test04.sh
deactivate
sbatch build/sbatch_scripts_sflow/wan22_gb300x72_offline_test04.sh
```
</details>

<details>
<summary>GB300-NVL72 x72 (18 nodes, SingleStream) - aarch64</summary>

```bash
source $SFLOW_VENV/bin/activate
sflow batch \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/GB300-NVL72_GB300-288GB_aarch64x72/visual_gen/SingleStream/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_no_vbench.yaml \
  --set WORK_DIR=$WORK_DIR \
  --set SCRATCH_DIR=$SCRATCH_DIR \
  --set NUM_SERVERS=1 \
  --set ENDPOINT_CONFIG=configs/wan22_a14b/_shared/endpoint_audit_test04_singlestream.yaml \
  --nodes=18 --partition=gb300 --account=coreai_mlperf_inference --time=05:00:00 \
  --job-name=wan22-gb300x72-ss-test04 \
  -o build/sbatch_scripts_sflow/wan22_gb300x72_ss_test04.sh
deactivate
sbatch build/sbatch_scripts_sflow/wan22_gb300x72_ss_test04.sh
```
</details>

<details>
<summary><code>B300-SXM-270GBx8</code> (1 node, Offline) - x86_64</summary>

```bash
source $SFLOW_VENV/bin/activate
sflow batch \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/B300-SXM-270GBx8/visual_gen/Offline/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_no_vbench.yaml \
  --set WORK_DIR=$WORK_DIR \
  --set SCRATCH_DIR=$SCRATCH_DIR \
  --set CONTAINER_IMAGE="docker://registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/wan22-trtllm-server:v6.1-jul10-wan22-trtllm-x86_64" \
  --set ENDPOINT_CONFIG=configs/wan22_a14b/_shared/endpoint_audit_test04_offline_x8.yaml \
  "--set CONTAINER_MOUNTS=$WORK_DIR:/work,$SCRATCH_DIR:/home/mlperf_inference_storage,$WORK_DIR/ci/scripts/smoke_test/patches/videogen_adapter.py:/opt/venv/lib/python3.12/site-packages/inference_endpoint/videogen/adapter.py,$WORK_DIR/ci/scripts/smoke_test/patches/visual_gen_executor.py:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/visual_gen/executor.py,$FFMPEG_BIN:/usr/local/bin/ffmpeg" \
  --nodes=1 --partition=batch --account=coreai_mlperf_inference --time=05:00:00 \
  --job-name=wan22-b300x8-offline-test04 \
  -o build/sbatch_scripts_sflow/wan22_b300x8_offline_test04.sh
deactivate
sbatch build/sbatch_scripts_sflow/wan22_b300x8_offline_test04.sh
```
</details>

<details>
<summary><code>B300-SXM-270GBx8</code> (1 node, SingleStream) - x86_64</summary>

```bash
source $SFLOW_VENV/bin/activate
sflow batch \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/B300-SXM-270GBx8/visual_gen/SingleStream/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_no_vbench.yaml \
  --set WORK_DIR=$WORK_DIR \
  --set SCRATCH_DIR=$SCRATCH_DIR \
  --set CONTAINER_IMAGE="docker://registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/wan22-trtllm-server:v6.1-jul10-wan22-trtllm-x86_64" \
  --set ENDPOINT_CONFIG=configs/wan22_a14b/_shared/endpoint_audit_test04_singlestream.yaml \
  "--set CONTAINER_MOUNTS=$WORK_DIR:/work,$SCRATCH_DIR:/home/mlperf_inference_storage,$WORK_DIR/ci/scripts/smoke_test/patches/videogen_adapter.py:/opt/venv/lib/python3.12/site-packages/inference_endpoint/videogen/adapter.py,$WORK_DIR/ci/scripts/smoke_test/patches/visual_gen_executor.py:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/visual_gen/executor.py,$FFMPEG_BIN:/usr/local/bin/ffmpeg" \
  --nodes=1 --partition=batch --account=coreai_mlperf_inference --time=05:00:00 \
  --job-name=wan22-b300x8-ss-test04 \
  -o build/sbatch_scripts_sflow/wan22_b300x8_ss_test04.sh
deactivate
sbatch build/sbatch_scripts_sflow/wan22_b300x8_ss_test04.sh
```
</details>

Validated TEST04 configs and pass results:

| System | Scenario | Samples | Threshold | Result |
|--------|----------|---------|-----------|--------|
| GB300-NVL72 x72 | Offline | 144/144 | 0.10 | PASS |
| GB300-NVL72 x72 | SingleStream | 50/50 | 0.20 | PASS |
| `B300-SXM-270GBx8` | Offline | 144/144 | 0.10 | PASS |
| `B300-SXM-270GBx8` | SingleStream | 50/50 | 0.20 | PASS |

#### Performance only run (no VBench) (optional)

Use the `no_vbench.yaml` template with `endpoint_perf.yaml` (perf dataset only, no
accuracy block). Example for GB300-NVL72 x4:

```bash
source $SFLOW_VENV/bin/activate
sflow batch \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/GB300-NVL72_GB300-288GB_aarch64x4/visual_gen/Offline/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_no_vbench.yaml \
  --set WORK_DIR=$WORK_DIR \
  --set SCRATCH_DIR=$SCRATCH_DIR \
  --set NUM_SERVERS=4 \
  --set ENDPOINT_CONFIG=configs/wan22_a14b/GB300-NVL72_GB300-288GB_aarch64x4/visual_gen/Offline/endpoint_perf.yaml \
  --nodes=1 --partition=gb300 --account=coreai_mlperf_inference --time=03:00:00 \
  --job-name=wan22-gb300x4-offline-perf \
  -o build/sbatch_scripts_sflow/wan22_gb300x4_offline_perf.sh
deactivate
sbatch build/sbatch_scripts_sflow/wan22_gb300x4_offline_perf.sh
```

### Step 4 - Submission packaging

wan22 is **endpoint-based**, so the `submission_checker` reads the endpoint
result files (`result_summary.json`, `results.json`, `config.yaml`) through its
`endpoints_parser`, **not** the LoadGen `mlperf_log_*` files. `wan-2.2-t2v-a14b`
is in the checker's `ENDPOINTS_ALLOWED_MODELS`; it validates endpoint
performance and accuracy (endpoint compliance is currently stubbed upstream, but
the TEST04 files are still required by the rules).

For each system and scenario (`<scenario>` = `Offline` or `SingleStream`), build
this tree, staging each run's files from its `endpoints_report/` output:

```
results/<system>/wan-2.2-t2v-a14b/<scenario>/
  performance/run_1/
      result_summary.json          # perf metric (QPS / per-query latency)
      results.json
      config.yaml
  accuracy/
      result_summary.json
      results.json                 # holds accuracy_scores.vbench_score (threshold 69.7752)
      config.yaml
      videos/                      # required: the specific sample video IDs the checker lists
  compliance/TEST04/
      audit_result.json            # from Command 2
      verify_OUTPUT_CACHING_TEST.txt   # "Performance check pass: True"
measurements/<system>/wan-2.2-t2v-a14b/<scenario>/
      measurements.json            # required
      README.md                    # required
      mlperf.conf                  # recommended (seed reference)
                                   # user.conf is NOT required for endpoint submissions
```

> Notes:
> - wan22 requires **TEST04** (output-caching) only, not TEST06/TEST07/TEST09.
> - There is no `truncate_accuracy_log.py` step: that truncates the LoadGen
>   `mlperf_log_accuracy.json`, which endpoint runs do not produce.

## Support Matrix

See [configs/SLURM_SUPPORT.md](../../../../configs/SLURM_SUPPORT.md) for
per-system, per-scenario SLURM support details.

## Parallelism

Video generation is **parallel within each replica**: a replica is one full Wan
2.2 server spanning `GPUS_PER_SERVER` GPUs (no ctx/gen/frontend split).

- **Offline** runs many **single-GPU replicas** (aggregated data-parallel): the
  endpoint client load-balances `NUM_SERVERS` independent servers, each
  generating one video at a time. `NUM_SERVERS` scales from 4 (1 node) to 72
  (18 nodes); the per-replica config is identical at every scale.
- **SingleStream** runs **one** replica that shards a single video across
  multiple GPUs (CFG × Ulysses sequence parallel, plus 2D context parallel on
  the full-rack x72) for lowest per-video latency.

| Scenario | Parallelism | GPUs/replica | Replicas | Attention backend |
|----------|-------------|--------------|----------|-------------------|
| Offline (all systems) | `cfg=1, ulysses=1` | 1 | `NUM_SERVERS` (4 … 72) | TE |
| SingleStream - GB200/GB300 x4 | `cfg=2, ulysses=2, vae=4` | 4 / 1 node | 1 | TE |
| SingleStream - B200/B300 x8 | `cfg=2, ulysses=4, vae=8` | 8 / 1 node | 1 | TE |
| SingleStream - GB200/GB300 x72 (full rack) | `cfg=2, ulysses=4, attn2d=[3,3], vae=4` (CP degree 9) | 72 / 18 nodes | 1 | FA4 |

> **Why x72 SingleStream uses `backend: FA4`, not `TE`:** the full-rack replica
> spans 72 GPUs / 18 nodes via 2D context parallel (`attn2d=[3,3]`). Context
> Parallel needs an LSE-capable attention backend (FA4 or CUTEDSL); TE does not
> support CP. The single-node scenarios use the `TE` FP8 attention backend.
> Note the backend difference in the submission notes.

## Generation parameters & determinism

- **Generation params** (VideoPathRequest defaults, identical across all
  configs): 81 frames, 720×1280, 20 steps, guidance 4.0 / 3.0, boundary 0.875,
  seed 42.
- **Determinism:** a fixed initial latent is enabled server-side via
  `TLLM_VISUAL_GEN_FIXED_LATENT_PATH` (set in `_shared/worker_env.yaml`), which
  VisualGen resolves into `fixed_latent_path`.
- **Quantization:** all configs load FP8 linear weights from the pre-quantized
  ModelOpt-FP8 checkpoint (static per-tensor). `visual_gen.yaml` intentionally
  omits `quant_config` so the checkpoint's embedded quantization is used as-is.

## Model & data

**Prompts (bundled, no download).** The 248 VBench prompts ship in-repo at
`configs/wan22_a14b/_shared/data/wan22_prompts.jsonl`. The endpoint configs load
them from `/work/configs/wan22_a14b/_shared/data/wan22_prompts.jsonl` (i.e.
`closed/Wiwynn` mounted at `/work`); no manual preprocessing is needed. Offline
perf configs issue `n_samples_to_issue: 144`; accuracy scores the full set.

**Checkpoint + fixed latent (staged under `SCRATCH_DIR`).** `SCRATCH_DIR` is
mounted at `/home/mlperf_inference_storage` in the containers and must be
reachable from every node in the job. Stage:

| Artifact | Source | Expected path (under `SCRATCH_DIR`) |
|----------|--------|--------------------------------------|
| ModelOpt FP8 diffusers checkpoint | HF [`centml/wan2.2-t2v-a14b-diffusers-mlpinf-fp8`](https://huggingface.co/centml/wan2.2-t2v-a14b-diffusers-mlpinf-fp8) | `models/wan22-a14b/Wan2.2-T2V-A14B-Diffusers-FP8` |
| Fixed initial latent | bundled `3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/fixed_latent.pt` | `preprocessed_data/wan22-a14b/fixed_latent.pt` |

The checkpoint is a ModelOpt FP8 (static per-tensor) quantization of the
Wan 2.2 A14B diffusers model - download it directly; no local quantization
step is needed:

```bash
huggingface-cli download centml/wan2.2-t2v-a14b-diffusers-mlpinf-fp8 \
  --local-dir "$SCRATCH_DIR/models/wan22-a14b/Wan2.2-T2V-A14B-Diffusers-FP8"
```

The fixed initial latent ships in-repo via the MLCommons inference submodule
(it is *not* part of the HF checkpoint above) - copy it into place:

```bash
mkdir -p "$SCRATCH_DIR/preprocessed_data/wan22-a14b"
cp 3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/fixed_latent.pt \
   "$SCRATCH_DIR/preprocessed_data/wan22-a14b/fixed_latent.pt"
```

Or run the helper that does both: `src/nv_mlpinf/benchmarks/wan22_a14b/scripts/prepare_data.sh` (honors
`MLPERF_SCRATCH_PATH`).

### Generating the quantization checkpoint

The download above is the recommended path. To instead quantize your own checkpoint
from the BF16 diffusers model (`Wan2.2-T2V-A14B-Diffusers`) - e.g. to re-calibrate
FP8 with different prompts - use ModelOpt's data-parallel diffusers quantizer,
`examples/diffusers/quantization/quantize_dp.py`, from a
[TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) checkout
(`$MODELOPT_REPO` below). Each rank loads the pipeline on one GPU, calibrates on a
disjoint shard of the prompts, and MAX-merges per-tensor `amax` across ranks. Output is
`transformer.pt` + `transformer_2.pt` plus a ready-to-serve `hf/` diffusers export.

```bash
cd "$MODELOPT_REPO" && source .venv/bin/activate
OUT=$SCRATCH_DIR/models/wan22-a14b/Wan2.2-T2V-A14B-Diffusers-FP8-custom
# Calibration prompts used to calibrate FP8 quantization ship in the mlc-inference submodule:
PROMPTS=3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/calibration_prompts.txt
# Single-rank example (1 GPU); calib-size can be any value (50 = v6.1 min-queries):
CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29511 \
python examples/diffusers/quantization/quantize_dp.py \
  --model wan2.2-t2v-14b --backbone transformer transformer_2 \
  --override-model-path "$BF16_MODEL" --model-dtype BFloat16 \
  --format fp8 --quantize-mha \
  --prompts-file "$PROMPTS" --batch-size 1 --calib-size 50 --n-steps 20 \
  --quantized-torch-ckpt-save-path "$OUT" --hf-ckpt-dir "$OUT/hf"

# To use multiple GPUs (faster), calib-size must be divisible by WORLD_SIZE - use 52 for 4 ranks:
# for r in 0 1 2 3; do
#   CUDA_VISIBLE_DEVICES=$r RANK=$r WORLD_SIZE=4 MASTER_ADDR=localhost MASTER_PORT=29511 \
#   python examples/diffusers/quantization/quantize_dp.py \
#     --model wan2.2-t2v-14b --backbone transformer transformer_2 \
#     --override-model-path "$BF16_MODEL" --model-dtype BFloat16 \
#     --format fp8 --quantize-mha \
#     --prompts-file "$PROMPTS" --batch-size 1 --calib-size 52 --n-steps 20 \
#     --quantized-torch-ckpt-save-path "$OUT" --hf-ckpt-dir "$OUT/hf" &
# done; wait
```

Gotchas (learned the hard way on GB200/GB300):

- **Do not pass `--collect-method` for FP8.** Omit it entirely; `--format fp8 --quantize-mha` is the correct FP8 invocation.
- **`--calib-size` must be divisible by `WORLD_SIZE`.** Otherwise ranks get uneven batch
  counts and the straggler makes the amax all-reduce exceed NCCL's default 600 s timeout
  (a single Wan 2.2 calibration batch is ~10 min). Use a divisible `calib-size`, or raise
  the process-group timeout in `quantize_dp.py`.
- **Runtime is ~4.5 h on 4× GB200.** Use a **non-preemptible** partition - a preempted
  run loses everything (checkpoints are only written between the two backbones).

Then point the harness at the generated `hf/` directory with `--set MODEL_PATH=...`
(symlink it under `SCRATCH_DIR` so every node can reach it).

> **Accuracy target (VBench):** mean VBench score of the BF16 reference is
> 70.48; the 99% pass threshold is 69.7752.

## Build the images from source

Both build via sflow on the enroot/pyxis cluster (no docker host required).
Build on a host of the **same architecture** as the target system (aarch64 for
GB200/GB300, x86_64 for B200/B300).

> **TRT-LLM source (server image only).** Build the server wheel from the wan22
> VisualGen branch of TensorRT-LLM:
> ```bash
> git clone https://github.com/NVIDIA/TensorRT-LLM.git
> git -C TensorRT-LLM checkout feat/1.3-mlpinf-vg
> ```
> Point `--set TRTLLM_REPO=<path>` at that checkout. The runtime path needs none
> of this - the published server image already bundles the VisualGen build.

**Server** - TRT-LLM VisualGen build (sm100+sm103 for GB200/GB300; sm100 for
B200/B300). Saves a `.sqsh`:

```bash
sflow run -f src/nv_mlpinf/benchmarks/wan22_a14b/scripts/generate_server_sqsh.yaml \
  --set DUMP_DIR=/lustre/fsw/coreai_mlperf_inference/$USER/sqsh \
  --set TRTLLM_REPO=<path to the feat/1.3-mlpinf-vg checkout> \
  --set BUILD_IMAGE=<TRT-LLM devel .sqsh> \
  --tui
```

`BUILD_IMAGE` is the devel base with the build toolchain (the matching
`LLM_*_DOCKER_IMAGE` tag from
`3rdparty/trtllm/jenkins/current_image_tags.properties`, or a local `.sqsh` of
it). Output: `<DUMP_DIR>/wan22-trtllm-server-sm100_sm103.sqsh`.

**Client** - `inference-endpoint` + `libgl1`/`libglib2.0-0` (VBench scorer's
opencv). Build via
`src/nv_mlpinf/benchmarks/wan22_a14b/scripts/generate_client_sqsh.yaml`. For the
accuracy/VBench variant add `--set BUILD_VBENCH=1 --set SQSH_NAME=endpoint_client_vbench`;
this bakes the locked vbench uv project + `.venv` into the image at
`/opt/vbench_accuracy`, so accuracy runs reuse it via `endpoint_env_vbench.yaml`'s
`UV_NO_SYNC=1` (no scratch staging). On aarch64, tokenizers 0.13.3 has no wheel,
so the bake compiles it from source with Rust - build on the matching partition
(e.g. `--set SLURM_PARTITION=gb300`).

## Prerequisites / bring-up notes

- **Server image** must contain the VisualGen attention backends the configs use
  - `TE` (FP8) for the single-node scenarios and `FA4` for the x72 full-rack
  SingleStream. The pre-built server image ships both; a stock TRT-LLM image
  without them will reject `backend: TE` / `FA4`.
- **Client image** must include `libgl1` + `libglib2.0-0` (VBench scorer's
  opencv); the accuracy/VBench variant additionally bakes the locked vbench uv
  project + `.venv` at `/opt/vbench_accuracy` (the configs' `vbench_project_path`
  points there).
- **Staged data** under `SCRATCH_DIR` (→ `/home/mlperf_inference_storage`): the
  FP8 checkpoint and `preprocessed_data/wan22-a14b/fixed_latent.pt` (see
  [Model & data](#model--data)). The vbench env is not staged here - it ships
  inside the VBench client image.
- Operators use sflow's native `container_image` field (no
  `container_name`/`extra_args`) so each replica gets its own per-task container
  (no enroot name collision when packing replicas per node).
