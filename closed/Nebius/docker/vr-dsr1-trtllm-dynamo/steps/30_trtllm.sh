#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0
#
# Step 30: build the TensorRT-LLM wheel at TRTLLM_COMMIT for TRTLLM_CUDA_ARCHS and install it.
# A persistent BUILD_DIR lets an interrupted build resume (ninja is incremental).
# Usage: RECIPE_DIR=... TRTLLM_SRC=<checkout> BUILD_DIR=<dir> WHEEL_OUT=<dir> [BUILD_JOBS=N] bash 30_trtllm.sh
set -euo pipefail
: "${RECIPE_DIR:?}" "${TRTLLM_SRC:?}" "${BUILD_DIR:?}" "${WHEEL_OUT:?}"
source "$RECIPE_DIR/versions.env"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1
# build_wheel.py --no-venv re-installs requirements-dev.txt (and conan): constrain only the protected base packages.
export PIP_CONSTRAINT=/opt/dsr1/protect.txt
export PIP_INDEX_URL="$PUBLIC_PYPI" PIP_EXTRA_INDEX_URL=
export LD_LIBRARY_PATH=/opt/hpcx/ompi5/lib:${LD_LIBRARY_PATH:-}
cd "$TRTLLM_SRC"
git config --global --add safe.directory '*' || true
SHA=$(git rev-parse HEAD)
echo "[30] source $TRTLLM_SRC @ $SHA (expected $TRTLLM_COMMIT)"
[ "$SHA" = "$TRTLLM_COMMIT" ] || { echo "[30] WRONG COMMIT"; exit 2; }
PTR=$(git lfs ls-files 2>/dev/null | grep ' - ' | grep -vcE 'x86_64-linux-gnu|^.{13}(docs|triton_backend)/' || true)
echo "[30] git-lfs files still pointers (outside excluded paths): $PTR"
[ "$PTR" = 0 ] || { echo "[30] LFS NOT MATERIALIZED"; exit 2; }
L=${BUILD_LOG:-/opt/dsr1/logs/build_wheel.log}; mkdir -p "$BUILD_DIR" "$WHEEL_OUT" "$(dirname "$L")"; rm -f "$WHEEL_OUT"/tensorrt_llm-*.whl
python3 -c 'import torch; print("[30] torch", torch.__version__, "cuda", torch.version.cuda)'
nvcc --version | tail -1 | sed 's/^/[30] /'
T0=$(date +%s)
# shellcheck disable=SC2086
python3 scripts/build_wheel.py -G Ninja -a "$TRTLLM_CUDA_ARCHS" -j "${BUILD_JOBS:-$(nproc)}" $TRTLLM_BUILD_EXTRA \
  --build_dir "$BUILD_DIR" --dist_dir "$WHEEL_OUT" --no-venv > "$L" 2>&1 \
  || { echo "[30] build_wheel FAILED after $(( $(date +%s) - T0 )) s"; grep -nE 'error|Error|FAILED' "$L" | head -40; tail -60 "$L"; exit 3; }
echo "[30] build_wheel OK in $(( $(date +%s) - T0 )) s"
cp "$L" /opt/dsr1/logs/build_wheel.log 2>/dev/null || true
grep -E 'GPU architectures|CMAKE_CUDA_ARCHITECTURES|Excluding SM|NIXL|NVRTC' "$L" | sort -u | head -20 | sed 's/^/[30] /'
W=$(ls "$WHEEL_OUT"/tensorrt_llm-*.whl | head -1)
sha256sum "$W" | sed 's/^/[30] wheel /'
pip install --no-deps --force-reinstall "$W" 2>&1 | tail -1
python3 -c 'import importlib.metadata as m; print("[30] tensorrt_llm", m.version("tensorrt_llm"))'
mkdir -p /opt/dsr1; echo "$SHA" > /opt/dsr1/TRTLLM_COMMIT; sha256sum "$W" > /opt/dsr1/TRTLLM_WHEEL.sha256

# Freeze after this step; step 40 installs under these constraints.
pip list --format=freeze > /opt/dsr1/logs/pip_freeze.30_trtllm.txt
grep -vE '^(tensorrt[-_]llm|ai-dynamo|ai-dynamo-runtime)==' /opt/dsr1/logs/pip_freeze.30_trtllm.txt > /opt/dsr1/constraints.txt
echo "[30] pip diff step 10 -> 30:"; diff /opt/dsr1/logs/pip_freeze.10_pydeps.txt /opt/dsr1/logs/pip_freeze.30_trtllm.txt | grep '^[<>]' | sed 's/^/[30]   /' || true
SP=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
ls "$SP/tensorrt_llm/libs/" | sed 's/^/[30] libs: /'
cuobjdump --list-elf "$SP/tensorrt_llm/libs/libtensorrt_llm.so" 2>/dev/null | grep -o 'sm_[0-9]*[af]*' | sort | uniq -c | sed 's/^/[30] libtensorrt_llm cubins: /' || true
# Import check (needs libcuda: a GPU build node or the NVIDIA container runtime).
cd /tmp && python3 -c 'import tensorrt_llm, torch; print("[30] import OK", tensorrt_llm.__version__, tensorrt_llm.__file__, torch.cuda.get_device_capability())'
echo "[30] DONE"
