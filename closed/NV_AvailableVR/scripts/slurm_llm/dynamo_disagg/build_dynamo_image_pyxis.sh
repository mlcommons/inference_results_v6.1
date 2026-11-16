#!/bin/bash
# Build the Dynamo disagg image by executing Dockerfile.dynamo's RUN steps
# verbatim inside the SAME image main's TRTLLM Interactive flow uses, then
# snapshotting it with pyxis --container-save. No docker required — works on
# any pyxis/enroot SLURM cluster (lyris, ptyche, ...), on an aarch64 node.
#
# Usage (login node):
#   Inside an existing allocation:
#     bash scripts/slurm_llm/dynamo_disagg/build_dynamo_image_pyxis.sh
#   Standalone (script sruns its own 1-node allocation):
#     PARTITION=gb200 bash scripts/slurm_llm/dynamo_disagg/build_dynamo_image_pyxis.sh
#
# Env overrides:
#   BASE_IMAGE   base image (local .sqsh path or NGC URL). Default: the sqsh
#                mirror of main's pinned Interactive image. The NGC URL
#                (nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64)
#                works on clusters without the mirror, given enroot NGC creds.
#   OUTPUT_SQSH  output image path (default: under the caller's images dir)
#   PARTITION / ACCOUNT / TIME   allocation params for standalone mode
#
# Build takes ~30-60 min (the maturin/Rust build of dynamo's python bindings
# dominates). The compute node needs outbound network access (github.com,
# astral.sh, rustup.rs) — the same access the CI's sflow venv bootstrap uses.

set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-/lustre/fsw/coreai_mlperf_inference/mlperf_inference_images/mlpinf+mlperf-inference+tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64+latest.sqsh}"
OUTPUT_SQSH="${OUTPUT_SQSH:-/lustre/fsw/coreai_mlperf_inference/${USER}/mlperf_inference_images/mlpinf-interactive-b5ddff4-jan28+dynamo-v0.8.0-pyxis.sqsh}"

mkdir -p "$(dirname "$OUTPUT_SQSH")"

ALLOC_FLAGS=()
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    # Partition names are per-cluster: lyris = gb200 / gb300, ptyche = 36x2-a01r.
    : "${PARTITION:?standalone mode: set PARTITION (lyris: gb200|gb300, ptyche: 36x2-a01r) or run inside salloc}"
    ALLOC_FLAGS+=(--partition "$PARTITION" --account "${ACCOUNT:-coreai_mlperf_inference}" --time "${TIME:-02:00:00}")
fi
# Some clusters enforce the job-name convention coreai_mlperf_inference-<subproject>.<details>.
ALLOC_FLAGS+=(--job-name "${JOB_NAME:-coreai_mlperf_inference-dynamo_image_build.${USER}}")

echo "Base image:   $BASE_IMAGE"
echo "Output image: $OUTPUT_SQSH"

# The bash -c body below is Dockerfile.dynamo's ENV + RUN steps, in order,
# unmodified except for docker-layer syntax -> shell.
srun "${ALLOC_FLAGS[@]}" --nodes=1 --ntasks=1 \
    --container-image="$BASE_IMAGE" \
    --container-remap-root \
    --container-writable \
    --container-save="$OUTPUT_SQSH" \
    bash -exc '
export CARGO_HOME=/root/.cargo
export RUSTUP_HOME=/root/.rustup
export PATH="${CARGO_HOME}/bin:/usr/local/bin/etcd:${PATH}"
export NATS_VERSION="v2.10.28"
export ETCD_VERSION="v3.5.21"
ARCH=arm64
DYNAMO_REPO=https://github.com/ai-dynamo/dynamo.git
DYNAMO_BRANCH=mlperf-v6.0-dynamo-v0.8.0

# --- System Dependencies ---
apt-get update && apt-get install -y --no-install-recommends \
    build-essential libhwloc-dev libudev-dev pkg-config libclang-dev \
    protobuf-compiler python3-dev cmake curl git wget \
  && rm -rf /var/lib/apt/lists/*

# --- NATS Server ---
wget --tries=3 --waitretry=5 \
    https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-${ARCH}.deb
dpkg -i nats-server-${NATS_VERSION}-${ARCH}.deb
rm nats-server-${NATS_VERSION}-${ARCH}.deb
nats-server --version

# --- etcd ---
rm -rf /usr/local/bin/etcd /usr/local/bin/etcdctl 2>/dev/null || true
wget --tries=3 --waitretry=5 \
    https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-${ARCH}.tar.gz \
    -O /tmp/etcd.tar.gz
mkdir -p /usr/local/bin/etcd
tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1
rm /tmp/etcd.tar.gz

# --- Rust + uv ---
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "${CARGO_HOME}/env"
rustup default stable
curl -LsSf https://astral.sh/uv/install.sh | sh

# --- Clone Dynamo ---
cd /opt
git clone --depth 1 --branch ${DYNAMO_BRANCH} ${DYNAMO_REPO} dynamo

# --- Build Dynamo wheel (slowest step) ---
cd /opt/dynamo
/root/.local/bin/uv venv /tmp/build_venv
. /tmp/build_venv/bin/activate
/root/.local/bin/uv pip install pip maturin
cd lib/bindings/python
maturin build --release --out /opt/dynamo/dist
deactivate
rm -rf /tmp/build_venv

# --- Install Dynamo and dependencies ---
pip install /opt/dynamo/dist/*.whl
pip install -e /opt/dynamo
pip install "nixl[cu13]<=0.8.0"
pip install cupy-cuda13x

# --- Cleanup ---
rm -rf /opt/dynamo/target "${CARGO_HOME}/registry" "${CARGO_HOME}/git"

# --- Verify installation ---
python -c "import dynamo; print(\"Dynamo installed successfully\")"
python -c "import cupy; print(f\"CuPy version: {cupy.__version__}\")"
python -c "import nixl; print(\"NIXL installed successfully\")"
'

echo "Done: $OUTPUT_SQSH"
