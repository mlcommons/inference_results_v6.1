#!/usr/bin/env bash
# Populate a local cache for openai_harmony's tiktoken vocab blob, so that
# benchmark/inference jobs running on compute nodes without egress to
# openaipublic.blob.core.windows.net can load HARMONY_GPT_OSS offline.
#
# Run this once on a host that CAN reach openaipublic.blob.core.windows.net
# (e.g. the cluster login node). After it succeeds, export the env var
# it prints at the bottom in whatever launches your benchmark.

set -euo pipefail

CACHE_DIR="${CACHE_DIR:-$HOME/.harmony_cache_gptoss}"
VENV_DIR="${VENV_DIR:-/tmp/harmony_venv}"

echo ">>> cache dir : $CACHE_DIR"
echo ">>> venv dir  : $VENV_DIR"

mkdir -p "$CACHE_DIR"

if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
    echo ">>> creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet openai-harmony

export TIKTOKEN_RS_CACHE_DIR="$CACHE_DIR"

python3 - <<'PY'
import os, openai_harmony as h
print("TIKTOKEN_RS_CACHE_DIR =", os.environ["TIKTOKEN_RS_CACHE_DIR"])
enc = h.load_harmony_encoding(h.HarmonyEncodingName.HARMONY_GPT_OSS)
print("loaded encoding:", enc.name)
PY

echo
echo ">>> cache contents:"
ls -la "$CACHE_DIR"

# Sanity: if nothing landed in CACHE_DIR, the lib fell back to /tmp/data-gym-cache.
if [[ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]]; then
    echo "!!! $CACHE_DIR is empty — checking /tmp/data-gym-cache fallback"
    if [[ -d /tmp/data-gym-cache ]]; then
        echo "    found blobs in /tmp/data-gym-cache, copying over"
        cp /tmp/data-gym-cache/* "$CACHE_DIR/"
        ls -la "$CACHE_DIR"
    else
        echo "    no fallback dir either — something went wrong"
        exit 1
    fi
fi

cat <<EOF

================================================================
Harmony cache ready at: $CACHE_DIR
Export this in any job/container that needs offline harmony:

  export TIKTOKEN_RS_CACHE_DIR=$CACHE_DIR

For pyxis/enroot, pass it through with:
  --container-env=TIKTOKEN_RS_CACHE_DIR
and make sure $CACHE_DIR is on a path the container can read
(if \$HOME isn't mounted, copy the dir to a shared cluster FS).
================================================================
EOF
