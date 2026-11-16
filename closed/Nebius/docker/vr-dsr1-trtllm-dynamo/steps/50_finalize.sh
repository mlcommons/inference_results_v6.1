#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0
#
# Step 50: provenance in /opt/dsr1 (BUILD_INFO, pip_freeze.final.txt, recipe copy) and the final checks: protected base
# packages unchanged, every nvidia-cutlass-dsl* distribution at the public pin, Dynamo, nats-server and etcd present.
set -euo pipefail
: "${RECIPE_DIR:?}"
source "$RECIPE_DIR/versions.env"
export CUTLASS_DSL_PUBLIC_VERSION CUTLASS_DSL_PACKAGES
SP=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
FAIL=0
mkdir -p /opt/dsr1/recipe; cp -r "$RECIPE_DIR"/. /opt/dsr1/recipe/ 2>/dev/null || true
pip list --format=freeze > /opt/dsr1/pip_freeze.final.txt

python3 - <<'PY' || FAIL=1
import importlib.metadata as m, cutlass
v = {}
for d in ["tensorrt_llm", "torch", "triton", "flashinfer-python", "nvidia-cutlass-dsl", "ai-dynamo", "ai-dynamo-runtime",
          "transformers", "nixl-cu13", "numpy", "aiohttp", "tensorrt"]:
    try:
        v[d] = m.version(d)
    except m.PackageNotFoundError:
        v[d] = None
print("[50] versions", v)
print("[50] cutlass", cutlass.__version__, cutlass.__file__)
import dynamo.trtllm  # noqa: F401
PY
python3 "$RECIPE_DIR/steps/check_pkgs.py" /opt/dsr1/protect.txt | sed 's/^/[50] /' || FAIL=1
for b in nats-server etcd nsys; do command -v $b > /dev/null || { echo "[50] missing $b"; FAIL=1; }; done
echo "[50] pip diff base -> final:"; diff /opt/dsr1/logs/pip_freeze.00_base.txt /opt/dsr1/pip_freeze.final.txt | grep '^[<>]' | grep -iE 'torch|triton|flashinfer|cutlass|tensorrt|cuda|numpy|aiohttp|dynamo|nixl|nccl' | sed 's/^/[50]   /' || true

{
  echo "DeepSeek-R1 Vera Rubin (sm_107) TensorRT-LLM + Dynamo image, built $(date -u +%FT%TZ) on $(uname -m)"
  echo "base: ${BASE_IMAGE_REPO}:${BASE_IMAGE_TAG}@${BASE_IMAGE_ARM64_DIGEST} (arm64)"
  echo "tensorrt_llm: $(cat /opt/dsr1/TRTLLM_COMMIT 2>/dev/null) (${TRTLLM_REF} from ${TRTLLM_REPO}), archs ${TRTLLM_CUDA_ARCHS}, extra: ${TRTLLM_BUILD_EXTRA}"
  echo "wheel: $(cat /opt/dsr1/TRTLLM_WHEEL.sha256 2>/dev/null)"
  echo "ucx: base UCX $(/usr/local/ucx/bin/ucx_info -v | sed -n 's/# Library version: //p')"
  echo "cutlass-dsl: $(pip list --format=freeze | grep -E '^nvidia-cutlass-dsl==') (public PyPI)"
  echo "flashinfer: $(pip list --format=freeze | grep -E '^flashinfer-python==') fmhaRunner SM107->SM100 patch: $(grep -c 'getSMVersion() == 107 ? kSM_100' "$SP/$FLASHINFER_FMHA_RUNNER_REL" || true)"
  echo "dynamo: ai-dynamo $(pip list --format=freeze | sed -n 's/^ai-dynamo==//p') + ai-dynamo-runtime $(pip list --format=freeze | sed -n 's/^ai-dynamo-runtime==//p') (PyPI, unpatched)"
  echo "nats $(nats-server --version 2>&1), $(etcd --version | head -1), nsys $(nsys --version 2>&1 | head -1)"
} > /opt/dsr1/BUILD_INFO
sed 's/^/[50] /' /opt/dsr1/BUILD_INFO
echo "[50] FAIL=$FAIL"
exit $FAIL
