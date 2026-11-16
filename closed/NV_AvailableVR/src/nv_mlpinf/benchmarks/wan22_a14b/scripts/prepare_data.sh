#!/usr/bin/env bash
# Stage Wan2.2-T2V-A14B data under MLPERF_SCRATCH_PATH:
#   1. Copy the MLPerf-deterministic fixed initial latent from the bundled
#      mlc-inference submodule into preprocessed_data/wan22-a14b/.
#   2. Download the Wan2.2 T2V-A14B diffusers model from HuggingFace into
#      models/wan22-a14b/.
#
# The configs expect (see README "Model & data"):
#   preprocessed_data/wan22-a14b/fixed_latent.pt            (TLLM_VISUAL_GEN_FIXED_LATENT_PATH)
#   models/wan22-a14b/Wan2.2-T2V-A14B-Diffusers-FP8         (MODEL_PATH)
#
# The model is NVIDIA's pre-published ModelOpt FP8 (static per-tensor) checkpoint
# on HuggingFace (nvidia/Wan2.2-T2V-A14B-Diffusers-FP8) — downloaded directly, no
# local quantization step. The fixed latent ships in the mlc-inference submodule.
set -euo pipefail

SCRATCH="${MLPERF_SCRATCH_PATH:-/home/mlperf_inference_storage}"
MODEL_NAME="${MODEL_NAME:-nvidia/Wan2.2-T2V-A14B-Diffusers-FP8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts -> wan22_a14b -> benchmarks -> nv_mlpinf -> src -> NVIDIA
NVIDIA_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

MLC_WAN_DIR="$NVIDIA_DIR/3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b"
FIXED_LATENT_SRC="$MLC_WAN_DIR/data/fixed_latent.pt"
DOWNLOAD_MODEL_PY="$MLC_WAN_DIR/download_model.py"

LATENT_DST="$SCRATCH/preprocessed_data/wan22-a14b/fixed_latent.pt"
MODEL_DST="$SCRATCH/models/wan22-a14b"

echo "============================================================"
echo "Wan2.2-T2V-A14B data prep"
echo "  MLPERF_SCRATCH_PATH : $SCRATCH"
echo "  Model               : $MODEL_NAME"
echo "============================================================"

# 1. Fixed initial latent (deterministic) ----------------------------------
if [[ ! -f "$FIXED_LATENT_SRC" ]]; then
  echo "ERROR: fixed_latent.pt not found at $FIXED_LATENT_SRC" >&2
  echo "       Did you init the mlc-inference submodule?" >&2
  echo "       git submodule update --init 3rdparty/mlc-inference" >&2
  exit 1
fi
echo "[1/2] Staging fixed latent -> $LATENT_DST"
mkdir -p "$(dirname "$LATENT_DST")"
cp -v "$FIXED_LATENT_SRC" "$LATENT_DST"

# 2. Model checkpoint from HuggingFace -------------------------------------
echo "[2/2] Downloading $MODEL_NAME -> $MODEL_DST"
mkdir -p "$MODEL_DST"
python3 "$DOWNLOAD_MODEL_PY" --download-path "$MODEL_DST" --model-name "$MODEL_NAME"

echo "============================================================"
echo "Done."
echo "  Fixed latent : $LATENT_DST"
echo "  Model        : $MODEL_DST/${MODEL_NAME##*/}"
echo "============================================================"
