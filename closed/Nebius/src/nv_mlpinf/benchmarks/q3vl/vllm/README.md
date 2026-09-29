# Qwen3-VL-235B-A22B

## Support Matrix

See [configs/SLURM_SUPPORT.md](../../../../../configs/SLURM_SUPPORT.md) for per-system, per-scenario SLURM support details.

 
## Additional setup before running any benchmarks
 
Ensure you have set this up before running any benchmark - for proper performance.
 
### 1. Copy files to home and pull images-
```bash
 # set up hf_cache in home - for speed
mkdir -p ~/hf_cache

# pull and move image to home - mentioned below
# image at: registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/v6.1-jul20-q3vl-amd64:latest

# pull + squash the endpoints harness image (this is what ENDPOINT_CONTAINER_IMAGE points to)
enroot import -o ~/mlcommons_endpoints__cc75b38e.sqsh docker://ghcr.io/mlcommons/endpoints@sha256:cc75b38e9d72358126e350602bdf0eef4ea53101061ba384fdfc20d7b69aa7a4

cd closed/Nebius
```
 
If the images above aren't already staged on your cluster, see images below for where to pull them from, and download model for the model weights. Home directory is used primarily for speed.
 
### 2. Pre-flight checks
 
Already fixed in the shared configs — just confirm before running. Full checks at the end of this file for reference if you run into any of the following issues -
 
1. `_shared/slurm_env.yaml` — worker tasks use a dedicated `worker_container` operator, **not** `test_container`. *[Check 1]*
2. `_shared/templates/vllm_dynamo_serve_endpoints.yaml` — worker task's `operator.name` is `worker_container`. *[Check 1]*
3. Worker script exports `CUDA_VISIBLE_DEVICES=$SFLOW_REPLICA_INDEX` (else workers see every GPU on the node). *[Check 2]*
4. Worker readiness probe `match_pattern` is `"dynamo.backend.generate"`, **not** `"Engine 000:"`.  *[Check 3]*
5. `qwen3vl_config.yaml` — `CPU_BIND_CORES: "0"`, **not** `"1"` (`"1"` pins to a single core instead of the NUMA node). *[Check 4]*

 
### 3. Set up 3rd-party deps
 
```bash
cd closed/NVIDIA
git submodule update --init
```
 
### 4. Set up sflow - v0.2.1
 
```bash
# Download uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# venv lives in closed/NVIDIA
uv venv

# updated version of sflow has a bug - do not use that
# uv pip install "sflow @ git+https://github.com/NVIDIA/nv-sflow.git@main"

# use sflow v0.2.1
uv pip install "sflow @ git+https://github.com/NVIDIA/nv-sflow.git@v0.2.1"
 
source .venv/bin/activate
```

## Getting Started

This directory contains the source code for NVIDIA's submission towards the
[vision-language model (VLM) benchmark](https://github.com/mlcommons/inference/tree/master/multimodal/qwen3-vl)
in the MLPerf Inference Benchmark Suite, starting from the v6.1 round.

### Download Model (Optional)

**Option 1 (Recommended): Rely on the sflow template for automatic download.**

The sflow workflow includes a `prefetch_model` task that runs before any vLLM
server starts. On the first run, pass a Hugging Face token so the template can
download the model into the shared host cache:

```bash
sflow batch ... \
  --set HF_CACHE_HOST_DIR=/path/to/hf_cache \
  --set HF_TOKEN=$HF_TOKEN
```

For later runs, keep passing the same `HF_CACHE_HOST_DIR`; `hf download` reuses
files already present in that cache.

**Option 2: Download the model yourself.**

```bash
hf download nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.1-FP8-KV --revision main --cache-dir /path/to/hf_cache/hub --token <your hf token: hf_xxx>
```

After the manual download, pass the cache root, not the `hub` subdirectory, to
sflow:

```bash
sflow batch ... \
  --set HF_CACHE_HOST_DIR=/path/to/hf_cache
```

### Download and Prepare Data (Automatically downloaded)

The Qwen3-VL-235B-A22B benchmark uses the same dataset for accuracy and performance runs. No manual transformations is needed before a benchmark run.
The recommended way is to follow this guide to allow [Endpoints]((git@github.com:mlcommons/endpoints.git)) client to download the dataset and do all processing automatically.

**Dataset Statistics:**


| Dataset                      | Samples | Accuracy Target | Scenarios Applied |
| ---------------------------- | ------- | --------------- | ----------------- |
| Shopify-product-catalogue    | 48289   | 0.7824          | Offline, Server   |
| Shopify-product-catalogue-8k | 8000    | 0.7799          | Interactive       |


## Base Image

Due to the server and SUT are now decoupled in v6.1 round via endpoints, two separate images are required to run the benchmark. Use the amd64 images for B200/B300 systems and the aarch64 images for GB200/GB300 systems.

### Use pre-built docker image (Recommended)

Internal users should use the images from NVIDIA GitLab:

| System | Arch | Server image                                                                                                                                                               | Client image |
| ------ | ---- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ------------ |
| B200-SXM-180GBx8, B300-SXM-270GBx8 | amd64 | `gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv:amd64_cuda13.0.1_CentML_dynamo-mlperf-inf-mm-q3vl-v6.1_CentML_vllm-mlperf-inf-mm-q3vl-v6.1`   | `ghcr.io/mlcommons/endpoints:598dbfde68ebfa42256762ebb60fd4da227b813c@sha256:cc75b38e9d72358126e350602bdf0eef4ea53101061ba384fdfc20d7b69aa7a4` |
| GB200-NVL72, GB300-NVL72 (single node system or full rack system) | aarch64 | `gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv:arm64_cuda13.0.1_CentML_dynamo-mlperf-inf-mm-q3vl-v6.1_CentML_vllm-mlperf-inf-mm-q3vl-v6.1` | `ghcr.io/mlcommons/endpoints:598dbfde68ebfa42256762ebb60fd4da227b813c@sha256:de854e3f913cbb0fd6f6b8e75c8fd817c08cc4971303302514cbaa132aa5b9c2` |

External users should use the images from the partner registry:

| System | Arch | Server image                                                                                           | Client image |
| ------ | ---- |--------------------------------------------------------------------------------------------------------| ------------ |
| B200-SXM-180GBx8, B300-SXM-270GBx8 | amd64 | `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/v6.1-jul20-q3vl-amd64:latest`   | `ghcr.io/mlcommons/endpoints:598dbfde68ebfa42256762ebb60fd4da227b813c@sha256:cc75b38e9d72358126e350602bdf0eef4ea53101061ba384fdfc20d7b69aa7a4` |
| GB200-NVL72, GB300-NVL72 (single node system or full rack system) | aarch64 | `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/v6.1-jul20-q3vl-aarch64:latest` | `ghcr.io/mlcommons/endpoints:598dbfde68ebfa42256762ebb60fd4da227b813c@sha256:de854e3f913cbb0fd6f6b8e75c8fd817c08cc4971303302514cbaa132aa5b9c2` |

## Zero-copy shared-memory tensor arena

The current vLLM build uses a zero-copy shared-memory tensor arena to avoid
pickling large CPU tensors, such as multimodal `pixel_values`, when the engine
broadcasts work to node-local tensor-parallel workers. This removes repeated
per-rank host copies from the engine's scheduling path. See the
[vLLM design document](https://github.com/CentML/vllm/blob/c1db3973aab91bf502ee87bc2daf9ec74ddd63a9/docs/design/shm_tensor_arena.md)
for the implementation, validation, and limitations.

The arena is **enabled automatically**: `VLLM_SHM_TENSOR_ARENA` defaults to
`1`, so the checked-in q3vl recipes do not need to set it. It is used only when
all queue readers are node-local. Ineligible tensors, an exhausted arena, an
oversized tensor, or queues with remote readers safely fall back to the normal
pickle/socket path.

Tune the feature through the scenario's `worker_env.yaml`, which is loaded
before each vLLM worker starts:

| Environment variable | Default | Tuning guidance |
| -------------------- | ------- | --------------- |
| `VLLM_SHM_TENSOR_ARENA` | `1` | Set to `0` to disable the feature and restore the normal transport path. |
| `VLLM_SHM_TENSOR_ARENA_SLOTS` | `8` | Increase when logs show slot-exhaustion fallbacks during concurrent large-image bursts. More slots increase shared-memory and pinned-host-memory requirements. |
| `VLLM_SHM_TENSOR_ARENA_SLOT_MB` | `256` | Set above the largest CPU tensor that should use the arena. Larger tensors fall back to pickle. Each arena reserves `SLOTS × SLOT_MB` of shared address space; the default is 2 GiB. |
| `VLLM_SHM_TENSOR_ARENA_MIN_MB` | `8` | Lower to divert smaller tensors; raise it to reserve arena slots for larger tensors. Tensors below the threshold use normal pickling. |

For example, to use 12 slots of 384 MiB for tensors of at least 16 MiB, add
the following to the applicable config under
`configs/qwen3_vl_235b_a22b/<system>/VLLM/<scenario>/worker_env.yaml`:

```yaml
VLLM_SHM_TENSOR_ARENA: 1
VLLM_SHM_TENSOR_ARENA_SLOTS: 12
VLLM_SHM_TENSOR_ARENA_SLOT_MB: 384
VLLM_SHM_TENSOR_ARENA_MIN_MB: 16
```

Size the arena within the node's `/dev/shm` and pinned-host-memory budget, then
compare tail TTFT and arena fallback logs under the target image-size and QPS
distribution. To manually disable the feature for an A/B test or rollback,
only this entry is required:

```yaml
VLLM_SHM_TENSOR_ARENA: 0
```

## Run the Benchmark in Multi Node using Slurm + Enroot

> ⚠️ You are recommended to use more detailed/corrected instructions, if available in  `closed/Nebius/configs/qwen3_vl_235b_a22b/<TARGET-SYSTEM>/VLLM/<BENCHMARK-TYPE>`. Only use the below instructions if those do not exist.

This section covers Qwen3-VL SLURM runs through `nv-sflow`: backend container
startup, endpoint client execution, model prefetch, performance, and accuracy.
Run the examples from a SLURM cluster login node under `closed/NVIDIA`. If you
are new to `nv-sflow`, first read [scaleout/sflow/README.md](../../../../../scaleout/sflow/README.md)
for installation and the `sflow batch` / `sflow run` concepts.

### Common nv-sflow settings

Replace `ACCT`, `PARTITION`, `SYSTEM`, and `HF_CACHE_HOST_DIR` for your cluster
and target system. The examples below use the full GB300x72 config.

```bash
ACCT=<your-slurm-account>
PARTITION=<your-slurm-partition>
SYSTEM=GB300-NVL72_GB300-288GB_aarch64x72
NODES=18
TIME=04:00:00
SROOT=configs/qwen3_vl_235b_a22b
SFLOW_DUMP_DIR=build/sbatch_scripts_sflow
HF_CACHE_HOST_DIR=/path/to/hf_cache
HF_TOKEN=hf_xxx  # recommended for the first model download
mkdir -p "$SFLOW_DUMP_DIR"
```

### Offline Scenario performance and accuracy back to back run

```bash
RUN=q3vl_${SYSTEM}_offline_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/${SYSTEM}/VLLM/Offline/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HF_CACHE_HOST_DIR \
  --set HF_TOKEN=$HF_TOKEN \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --set SLURM_TIME=$TIME \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  --time=$TIME \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
```

### Server Scenario performance and accuracy back to back run

```bash
RUN=q3vl_${SYSTEM}_server_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/${SYSTEM}/VLLM/Server/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HF_CACHE_HOST_DIR \
  --set HF_TOKEN=$HF_TOKEN \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --set SLURM_TIME=$TIME \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  --time=$TIME \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
```

### Interactive Scenario performance and accuracy back to back run

Interactive uses the 8k Shopify dataset and runs with P/D disaggregated setup, and the
only tested and supported system for this disaggregated setup is GB300-NVL72_GB300-288GB_aarch64x72.

```bash
RUN=q3vl_${SYSTEM}_interactive_$(date +%Y%m%d-%H%M%S)

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_pd_disagg_serve_endpoints.yaml \
  -f $SROOT/GB300-NVL72_GB300-288GB_aarch64x72/VLLM/Interactive/qwen3vl_pd_disagg_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HF_CACHE_HOST_DIR \
  --set HF_TOKEN=$HF_TOKEN \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --set SLURM_TIME=$TIME \
  --nodes=$NODES \
  --partition=$PARTITION \
  --account=$ACCT \
  --time=$TIME \
  -o ${SFLOW_DUMP_DIR}/$RUN.sh \
  --submit
```

### Debug mode: use nv-sflow Terminal UI mode (`sflow run`)

Use `sflow run` for short debug runs when you want the live TUI instead of an
asynchronous sbatch script. This example uses the single-node x4 Server config;
use `sflow batch` for full-rack submissions.

```bash
DEBUG_SYSTEM=GB300-NVL72_GB300-288GB_aarch64x4
RUN=q3vl_${DEBUG_SYSTEM}_server_debug_$(date +%Y%m%d-%H%M%S)
OUT=scaleout/sflow_output/$RUN

sflow run \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_serve_endpoints.yaml \
  -f $SROOT/${DEBUG_SYSTEM}/VLLM/Server/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HF_CACHE_HOST_DIR \
  --set HF_TOKEN=$HF_TOKEN \
  --set SLURM_ACCOUNT=$ACCT \
  --set SLURM_PARTITION=$PARTITION \
  --set SLURM_TIME=$TIME \
  --output-dir $OUT \
  --tui
```

### Tip: scaled test runs before full rack

For the full-rack x72 configs, `TOTAL_GPUS: 72` in `qwen3vl_config.yaml` means
18 nodes with 4 GPUs per node. For a smaller test, copy or edit the scenario's
`qwen3vl_config.yaml` and reduce `TOTAL_GPUS` to a smaller multiple of 4, then
set `NODES=TOTAL_GPUS/4` in the `sflow batch` command.

When scaling down, also reduce the load in the matching `endpoint.yaml`. For
Server and Interactive, lower `settings.load_pattern.target_qps` roughly in
proportion to the GPU count. For Offline, lower
`settings.runtime.n_samples_to_issue: 869202` so the test does not issue the
full-rack sample count. Once the smaller-scale run is stable, switch back to the
full x72 config for a full-rack result.

### Results

- Per-task logs: `$OUT/{output_folder}/{sflow.log,<task>/...}`
- Final QPS / Latency: `results/job_folder/results.json`
- Loadgen percentiles (TTFT, TPOT, end-to-end latency):
`results/qwen3_vl_235b_a22b_shopify_benchmark_offline_nvl4/report.txt`


## Benchmark Passing Criteria

A run is valid only if the required accuracy and performance criteria pass.
Qwen3-VL uses the inference endpoint client. That client records latency in the
endpoint report, but it does not natively enforce the benchmark latency
constraint. Check the reported p99 end-to-end latency before treating Server or
Interactive results as valid. If latency is over the threshold, lower target QPS
and rerun.


| Scenario    | Dataset                      | Accuracy criteria                               | Performance criteria      |
| ----------- | ---------------------------- | ----------------------------------------------- | ------------------------- |
| Offline     | Shopify Product Catalogue    | `F1_HIERARCHICAL >= 0.7903 * 0.99 = 0.782397`   | No p99 latency constraint |
| Server      | Shopify Product Catalogue    | `F1_HIERARCHICAL >= 0.7903 * 0.99 = 0.782397`   | End-to-end p99 <= 12 s    |
| Interactive | Shopify Product Catalogue 8k | `F1_HIERARCHICAL >= 0.78777 * 0.99 = 0.7798923` | End-to-end p99 <= 1.5 s   |


Following section is for **developers only** who are interested in build the docker images or generate their own quantized checkpoints.

## Build the container image

#### Via Docker

You can leverage [scripts/build_image.sh](scripts/build_image.sh) to build a container
image end-to-end for running this benchmark. At the
[closed/NVIDIA/src/nv_mlpinf/benchmarks/q3vl/vllm](closed/NVIDIA/src/nv_mlpinf/benchmarks/q3vl/vllm)
directory (i.e., where this `README.md` is), run the following command:

```bash
bash scripts/build_image.sh
```

#### Via enroot (no Docker daemon required)

[scripts/build_image_enroot.sh](scripts/build_image_enroot.sh) builds the same container
image using enroot directly on SLURM compute nodes. All sources (vllm, dynamo, mlperf
packages) are cloned from git inside the container.

**Basic usage** (from the project directory):

```bash
# Run directly on a compute node
bash scripts/build_image_enroot.sh \
    --dynamo-revision 04a6e12 \
    --vllm-revision a65093c

# Or submit via SLURM
sbatch --account=<account> --partition=<partition> -N1 --time=04:00:00 --mem=0 \
    --output=./output/slurm_%j/stdout --error=./output/slurm_%j/stderr \
    scripts/build_image_enroot.sh \
        --dynamo-revision 04a6e12 \
        --vllm-revision a65093c
```

**Caching the vllm build** (vllm compilation takes ~1.5 hours):

```bash
# First build: compile vllm and cache the intermediate image
bash scripts/build_image_enroot.sh \
        --dynamo-revision 04a6e12 \
        --vllm-revision a65093c \
        --cache-vllm-base

# Subsequent builds: reuse the cached vllm image (~5 min)
bash scripts/build_image_enroot.sh \
    --dynamo-revision 04a6e12 \
    --vllm-revision a65093c \
    --vllm-base-sqsh build/cache/vllm-CentML_vllm-mlperf-inf-mm-q3vl-v6.0-cuda13.0.1-arm64.sqsh
```

**Output**: a `.sqsh` file in `build/` (override with `--sqsh-output-dir`).
The CUDA base image is cached in `build/cache/` and reused across builds.

Run `bash scripts/build_image_enroot.sh --help` for all available options.

If you would like to run the benchmark on a `amd64` (i.e., x86) system, you would need
to build the image on an `amd64` (i.e., x86) system. Conversely, you would need to build
the image on an `arm64` (i.e., `aarch64`) system for running the benchmark on a `arm64`
(i.e., `aarch64`) system.

Building the vLLM base image can be intensive on CPU and host memory resources. We
recommend to build the image on a machine with at least 72 CPU threads and 574 GB of
host memory.






### NVFP4 + FP8-KV Quantization with TensorRT-Model-Optimizer

> [!NOTE]
> We quantized the model to NVFP4 (W4A4) with a calibrated per-tensor FP8 KV cache and uploaded it to
> [nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.1-FP8-KV](https://huggingface.co/nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.1-FP8-KV)
> (served from the repo `main`).
> Please use this provided checkpoint if you don't have a specific need to calibrate one yourself.
> All benchmarking configs use it by default (`MODEL_REPO_ID` in each
> `configs/qwen3_vl_235b_a22b/.../qwen3vl_config.yaml`).

The checkpoint is produced with [NVIDIA TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer):
NVFP4 on every linear layer (static MSE weight scales plus dynamic NVFP4 inputs) with a calibrated
per-tensor FP8 KV cache. Adding the FP8 KV cache and recovering accuracy with MSE weight calibration
matches the FP16-KV NVFP4-only baseline within noise at higher throughput.

We calibrate on the **Shopify** dataset (the benchmark's own data) to comply with the MLPerf calibration
rules.

To reproduce the checkpoint, see [scripts/quantization/README.md](scripts/quantization/README.md): one
command (`scripts/quantization/quantize_qwen3vl_nvfp4_fp8kv.py`) that builds the Shopify calibration, runs
the ModelOpt PTQ, and writes a vLLM-loadable export. The quantization runs in a dedicated ModelOpt
environment and does not require the vLLM base or submission container.

## Checks (Nebius specific)
 
### Check 1 — worker container isolation
 
`test_container` sets a fixed `--container-name`; pyxis attaches to an existing container of that name instead of creating a fresh one, so worker replicas shared a container and lost GPU isolation. `_shared/slurm_env.yaml` adds a dedicated `worker_container` operator (no `--container-name`, so each replica gets its own container):
 
```diff
   - name: test_container
     ...
     extra_args:
       - --container-image=${{ variables.CONTAINER_IMAGE }}
       - --container-name=${{ variables.BACKEND_CONTAINER_NAME }}
+  - name: worker_container
+    type: srun
+    container_writable: false
+    container_mount_home: false
+    container_remap_root: true
+    mpi: pmix
+    container_mounts:
+      - ${{ variables.CONTAINER_MOUNTS }}
+    container_workdir: /work
+    export: "ALL,ENROOT_TRANSFER_RETRIES=3,ENROOT_CONNECT_TIMEOUT=60"
+    extra_args:
+      - --container-image=${{ variables.CONTAINER_IMAGE }}
+
   - name: frontend_container
```
 
`_shared/templates/vllm_dynamo_serve_endpoints.yaml`'s worker task switches to that operator:
 
```diff
       operator:
-        name: test_container
+        name: worker_container
```
 
Note: `frontend_container` still has this same underlying problem, unfixed.
 
### Check 2 — GPU visibility per worker
 
`_shared/templates/vllm_dynamo_serve_endpoints.yaml`:
 
```diff
         - eval "$(python3 /work/scaleout/sflow/tools/run_with_env.py ${{ variables.WORKER_ENV_YAML }} | tee /dev/stderr)"
+        - export CUDA_VISIBLE_DEVICES=$SFLOW_REPLICA_INDEX
```
 
### Check 3 — readiness probe pattern
 
`_shared/templates/vllm_dynamo_serve_endpoints.yaml`. The old pattern (`"Engine 000:"`) never appears in actual startup logs, which could hang the workflow waiting on a match that never comes:
 
```diff
         readiness:
           log_watch:
-            match_pattern: "Engine 000:"
+            match_pattern: "dynamo.backend.generate"
             match_count: 1
```
 
### Check 4 — CPU binding granularity
 
`.../VLLM/Server/qwen3vl_config.yaml` and `.../VLLM/Offline/qwen3vl_config.yaml`. `CPU_BIND_CORES: "1"` pinned each vLLM worker to a single CPU core instead of the intended NUMA-node-level binding, crippling the worker. **Applied for B200; still needed for B300 — see Known Issues.**
 
```diff
   CPU_BIND_DIVISOR:
     value: "4"
   CPU_BIND_CORES:
-    value: "1"
+    value: "0"
```
