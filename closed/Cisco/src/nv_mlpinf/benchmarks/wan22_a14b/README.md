# Wan 2.2 A14B — Text-to-Video (sflow + inference-endpoints)

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

On aarch64 (GB200 / GB300) you can run with the **pre-built images** — no build
step. First stage the FP8 checkpoint + fixed latent under `SCRATCH_DIR` (see
[Model & data](#model--data)), then run.

### Pre-built images (recommended — aarch64 / GB200, GB300)

Pre-built images are published for the aarch64 systems and are the defaults in
`configs/wan22_a14b/_shared/slurm_env.yaml`, so a normal aarch64 run needs no
image flags:

- **server:** `docker://registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/wan22-trtllm-server:v6.1-jun26-wan22-trtllm-aarch64`
- **client:** `docker://registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/wan22-endpoint-client:v6.1-jun26-wan22-endpoints-aarch64`

To pull a private registry URL on the cluster, add `~/.config/enroot/.credentials`:
`machine registry.gitlab.com login <user> password <token>`. Pyxis pulls
`docker://` registry URLs directly; override either default with
`--set CONTAINER_IMAGE=...` / `--set ENDPOINT_CONTAINER_IMAGE=...` (a local
`.sqsh` path or a different tag).

| System | Arch | Server image | Client image |
|--------|------|--------------|--------------|
| GB200-NVL72, GB300-NVL72 | aarch64 | pre-built (above) | pre-built (above) |
| B200x8, B300x8 | x86_64 | pending | pending |

> **x86 (B200x8 / B300x8):** no pre-built amd64 image is published yet — build
> both images from source on an x86 host/partition (see
> [Build the images from source](#build-the-images-from-source)).

### Run

All commands run from `closed/NVIDIA` (so `$PWD` → `/work` inside the container).
Pick a `<SYSTEM>` dir (`GB300-NVL72_GB300-288GB_aarch64x4`,
`GB300-NVL72_GB300-288GB_aarch64x72`, `GB200-NVL72_GB200-186GB_aarch64x4`,
`GB200-NVL72_GB200-186GB_aarch64x72`, `B200-SXM-180GBx8`, `B300-SXM-270GBx8`) and
a `<SCENARIO>` (`Offline` or `SingleStream`). `--nodes = ceil(TOTAL_GPUS / 4)`:
Offline x4→1, x72→18; SingleStream x4/x8→1, **x72 full-rack→18**.

A run uses three `-f` files: `_shared/slurm_env.yaml`, the per-system
`<SYSTEM>/visual_gen/<SCENARIO>/wan22_config.yaml`, and one of the templates in
`_shared/templates/`.

A run has a **type** (which template, plus one dataset edit) and a **launch
mechanism** (batch or interactive). The two are independent — either mechanism
works for either type.

**Run type:**
- **Submission** (performance + accuracy) — template
  `wan22_videogen_endpoints_submission.yaml`. Runs both the `wan22_perf` and
  `wan22_vbench` datasets; needs the vbench client image and
  `--set ENDPOINT_GPUS=1` (the scorer reuses an idle server GPU *after*
  generation, not an extra one). See [Submission run](#submission-run).
- **Performance-only** — template `wan22_videogen_endpoints_test_run.yaml` (no
  VBench scorer). **First comment out the `wan22_vbench` accuracy dataset** in the
  scenario's `endpoint_submission.yaml`; otherwise the accuracy pass runs with no
  scorer GPU and fails. See [Perf-only test session](#perf-only-test-session).

Both templates read the scenario's `endpoint_submission.yaml` via the
`ENDPOINT_CONFIG` default in `wan22_config.yaml`.

**Launch mechanism** — batch (below) or interactive; the examples below show a
performance-only run, with a one-line note for switching each to a submission run.

#### Batch (recommended)

`sflow batch` bootstraps a per-node sflow venv **inside** the job, so it must run
with a clean environment — generate the script with the sflow venv active, but
**`sbatch` it from a shell with no venv activated** (otherwise the login venv
leaks in via `--export=ALL` and the aarch64 bootstrap fails silently).

This example is a **performance-only** run — comment out the `wan22_vbench`
dataset in the scenario's `endpoint_submission.yaml` first (see
[Perf-only test session](#perf-only-test-session)). For a **submission** run via
batch, swap the template to `wan22_videogen_endpoints_submission.yaml` and add
`--set NUM_SERVERS=<GPUs/node> --set ENDPOINT_GPUS=1`.

```bash
# 1) generate the sbatch script (sflow venv active)
source <your-sflow-venv>/bin/activate
sflow batch \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/GB300-NVL72_GB300-288GB_aarch64x4/visual_gen/Offline/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_test_run.yaml \
  --set WORK_DIR=$PWD \
  --set SCRATCH_DIR=<shared storage path> \
  --nodes=1 --partition=gb300 --account=coreai_mlperf_inference --time=03:00:00 \
  --job-name=wan22-gb300x4-offline \
  -o build/sbatch_scripts_sflow/wan22_gb300x4_offline.sh          # NOTE: no --submit

# 2) submit from a CLEAN shell (no venv active)
deactivate 2>/dev/null
sbatch build/sbatch_scripts_sflow/wan22_gb300x4_offline.sh

# For the full-rack low-latency single video, swap in the x72 SingleStream config:
#   -f configs/wan22_a14b/GB300-NVL72_GB300-288GB_aarch64x72/visual_gen/SingleStream/wan22_config.yaml
#   --nodes=18 --job-name=wan22-gb300x72-singlestream
```

On aarch64 the server/client images default to the pre-built tags, so no image
`--set` is needed. On x86, add `--set CONTAINER_IMAGE=<server .sqsh>` and
`--set ENDPOINT_CONTAINER_IMAGE=<client .sqsh>` (built from source).

#### Interactive (quick smoke test)

Same `-f` files with `sflow run … --tui` (reads nodes/partition/time from
`slurm_env.yaml`); blocks on the login node. Like the batch example this is
**performance-only** — comment out the `wan22_vbench` dataset first (see
[Perf-only test session](#perf-only-test-session)); for a full submission use the
[Submission run](#submission-run) below.

```bash
sflow run \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/GB300-NVL72_GB300-288GB_aarch64x4/visual_gen/Offline/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_test_run.yaml \
  --set WORK_DIR=$PWD \
  --set SCRATCH_DIR=<shared storage path> \
  --tui
```

#### Submission run

Performance + accuracy / VBench, back to back. Use the submission template + the
**vbench** client image (the aarch64 default; on x86 pass it explicitly — see
below). The scenario's `endpoint_submission.yaml` is already the
`ENDPOINT_CONFIG` default (in `wan22_config.yaml`), so no `--set ENDPOINT_CONFIG`
is needed. VBench scoring runs *after* video generation, so the scorer reuses a
server GPU (the servers are idle by then) instead of reserving an extra one —
every GPU on the node serves (e.g. GB300x4: `NUM_SERVERS=4, ENDPOINT_GPUS=1`).
Shown interactively below; to run it as a batch job, use the `sflow batch`
two-step from [Batch](#batch-recommended) with this template and the same
`--set` flags.

```bash
sflow run \
  -f configs/wan22_a14b/_shared/slurm_env.yaml \
  -f configs/wan22_a14b/GB300-NVL72_GB300-288GB_aarch64x4/visual_gen/Offline/wan22_config.yaml \
  -f configs/wan22_a14b/_shared/templates/wan22_videogen_endpoints_submission.yaml \
  --set WORK_DIR=$PWD --set NUM_SERVERS=4 --set ENDPOINT_GPUS=1 \
  --set SCRATCH_DIR=<shared storage path> \
  --tui
```

On x86, also `--set CONTAINER_IMAGE=<server .sqsh>` and
`--set ENDPOINT_CONTAINER_IMAGE=<vbench client .sqsh>`.

#### Perf-only test session

The Batch and Interactive examples above use the `test_run` template, which runs
whichever datasets the scenario's `endpoint_submission.yaml` defines. To measure
**performance only** — skipping the slow VBench accuracy pass, so no scorer GPU
(`ENDPOINT_GPUS`) and no vbench client image are needed — comment out the
`wan22_vbench` accuracy dataset in that file:

```yaml
datasets:
  - name: wan22_perf
    type: performance
    path: /work/configs/wan22_a14b/_shared/data/wan22_prompts.jsonl
    samples: 144
  # - name: wan22_vbench          # <-- comment out for a perf-only run
  #   type: accuracy
  #   path: /work/configs/wan22_a14b/_shared/data/wan22_prompts.jsonl
  #   samples: 248
  #   accuracy_config:
  #     eval_method: vbench
  #     ...
```

Then launch exactly as in the Batch / Interactive examples above (the `test_run`
template, default images, no `ENDPOINT_GPUS`).

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
| SingleStream — GB200/GB300 x4 | `cfg=2, ulysses=2, vae=4` | 4 / 1 node | 1 | TE |
| SingleStream — B200/B300 x8 | `cfg=2, ulysses=4, vae=8` | 8 / 1 node | 1 | TE |
| SingleStream — GB200/GB300 x72 (full rack) | `cfg=2, ulysses=4, attn2d=[3,3], vae=4` (CP degree 9) | 72 / 18 nodes | 1 | FA4 |

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
`closed/NVIDIA` mounted at `/work`); no manual preprocessing is needed. Offline
perf configs issue `n_samples_to_issue: 144`; accuracy scores the full set.

**Checkpoint + fixed latent (staged under `SCRATCH_DIR`).** `SCRATCH_DIR` is
mounted at `/home/mlperf_inference_storage` in the containers and must be
reachable from every node in the job. Stage:

| Artifact | Source | Expected path (under `SCRATCH_DIR`) |
|----------|--------|--------------------------------------|
| ModelOpt FP8 diffusers checkpoint | HF [`nvidia/Wan2.2-T2V-A14B-Diffusers-FP8`](https://huggingface.co/nvidia/Wan2.2-T2V-A14B-Diffusers-FP8) | `models/wan22-a14b/Wan2.2-T2V-A14B-Diffusers-FP8` |
| Fixed initial latent | bundled `3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/fixed_latent.pt` | `preprocessed_data/wan22-a14b/fixed_latent.pt` |

The checkpoint is NVIDIA's pre-published ModelOpt FP8 (static per-tensor)
quantization of the Wan 2.2 A14B diffusers model — download it directly; no
local quantization step is needed:

```bash
huggingface-cli download nvidia/Wan2.2-T2V-A14B-Diffusers-FP8 \
  --local-dir "$SCRATCH_DIR/models/wan22-a14b/Wan2.2-T2V-A14B-Diffusers-FP8"
```

The fixed initial latent ships in-repo via the MLCommons inference submodule
(it is *not* part of the HF checkpoint above) — copy it into place:

```bash
mkdir -p "$SCRATCH_DIR/preprocessed_data/wan22-a14b"
cp closed/NVIDIA/3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/fixed_latent.pt \
   "$SCRATCH_DIR/preprocessed_data/wan22-a14b/fixed_latent.pt"
```

Or run the helper that does both: `scripts/prepare_data.sh` (honors
`MLPERF_SCRATCH_PATH`).

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
> of this — the published server image already bundles the VisualGen build.

**Server** — TRT-LLM VisualGen build (sm100+sm103 for GB200/GB300; sm100 for
B200/B300). Saves a `.sqsh`:

```bash
sflow run -f src/nv_mlpinf/benchmarks/wan22_a14b/scripts/generate_server_sqsh.yaml \
  --set DUMP_DIR=/lustre/fsw/coreai_mlperf_inference/$USER/sqsh \
  --set TRTLLM_REPO=<path to the feat/1.3-mlpinf-vg checkout> \
  --set BUILD_IMAGE=<TRT-LLM devel .sqsh or urm.nvidia.com tag> \
  --tui
```

`BUILD_IMAGE` is the devel base with the build toolchain (the matching
`LLM_*_DOCKER_IMAGE` tag from
`3rdparty/trtllm/jenkins/current_image_tags.properties`, or a local `.sqsh` of
it). Output: `<DUMP_DIR>/wan22-trtllm-server-sm100_sm103.sqsh`.

**Client** — `inference-endpoint` + `libgl1`/`libglib2.0-0` (VBench scorer's
opencv). Build via
`src/nv_mlpinf/benchmarks/wan22_a14b/scripts/generate_client_sqsh.yaml`. For the
accuracy/VBench variant add `--set BUILD_VBENCH=1 --set SQSH_NAME=endpoint_client_vbench`;
this bakes the locked vbench uv project + `.venv` into the image at
`/opt/vbench_accuracy`, so accuracy runs reuse it via `endpoint_env_vbench.yaml`'s
`UV_NO_SYNC=1` (no scratch staging). On aarch64, tokenizers 0.13.3 has no wheel,
so the bake compiles it from source with Rust — build on the matching partition
(e.g. `--set SLURM_PARTITION=gb300`).

## Prerequisites / bring-up notes

- **Server image** must contain the VisualGen attention backends the configs use
  — `TE` (FP8) for the single-node scenarios and `FA4` for the x72 full-rack
  SingleStream. The pre-built server image ships both; a stock TRT-LLM image
  without them will reject `backend: TE` / `FA4`.
- **Client image** must include `libgl1` + `libglib2.0-0` (VBench scorer's
  opencv); the accuracy/VBench variant additionally bakes the locked vbench uv
  project + `.venv` at `/opt/vbench_accuracy` (the configs' `vbench_project_path`
  points there).
- **Staged data** under `SCRATCH_DIR` (→ `/home/mlperf_inference_storage`): the
  FP8 checkpoint and `preprocessed_data/wan22-a14b/fixed_latent.pt` (see
  [Model & data](#model--data)). The vbench env is not staged here — it ships
  inside the VBench client image.
- Operators use sflow's native `container_image` field (no
  `container_name`/`extra_args`) so each replica gets its own per-task container
  (no enroot name collision when packing replicas per node).
