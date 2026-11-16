#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0
#
# Step 10: Python dependencies of TensorRT-LLM from public PyPI (plus the PyTorch cu130 index named in its
# requirements), with the FlashInfer and CuTe DSL pins and the protected base packages locked to their versions.
set -euo pipefail
: "${RECIPE_DIR:?}" "${TRTLLM_SRC:?}"
source "$RECIPE_DIR/versions.env"
export CUTLASS_DSL_PUBLIC_VERSION CUTLASS_DSL_PACKAGES
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1
# Public PyPI only, whatever the base image's pip config says.
export PIP_INDEX_URL="$PUBLIC_PYPI" PIP_EXTRA_INDEX_URL=
mkdir -p /opt/dsr1/logs
SP=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
pip config list 2>/dev/null | sed 's/^/[10] pip config: /' || true
pip list --format=freeze > /opt/dsr1/logs/pip_freeze.00_base.txt

# Remove duplicate dist-info directories, keeping the highest version of each distribution.
python3 - "$SP" <<'PY'
import collections, os, re, shutil, sys
from packaging.version import Version
sp = sys.argv[1]; d = collections.defaultdict(list)
for n in os.listdir(sp):
    m = re.match(r"(.+?)-([^-]+)\.dist-info$", n)
    if m: d[m.group(1).lower().replace("_", "-")].append((Version(m.group(2)), n))
for k, v in d.items():
    if len(v) > 1:
        v.sort()
        for _, n in v[:-1]:
            print(f"[10] removing stale {n} (keeping {v[-1][1]})"); shutil.rmtree(os.path.join(sp, n))
PY

# FlashInfer pin, then record the protected base packages at their exact versions.
PIP_CONSTRAINT='' pip install --no-deps --force-reinstall "flashinfer-python==${FLASHINFER_VERSION}" 2>&1 | tail -1
python3 - "$PROTECTED_PKGS" > /opt/dsr1/protect.txt <<'PY'
import sys, importlib.metadata as m
for n in sys.argv[1].split():
    try:
        print(f"{n}=={m.version(n)}")
    except m.PackageNotFoundError:
        pass
PY
echo "[10] protected:"; sed 's/^/[10]   /' /opt/dsr1/protect.txt

# Replace every installed nvidia-cutlass-dsl* distribution with the public pin, and protect the pin.
mapfile -t OLD < <(pip list --format=freeze | sed -nE 's/^(nvidia[-_.]cutlass[-_.]dsl[A-Za-z0-9._-]*)(==| @ ).*/\1/Ip')
if [ "${#OLD[@]}" -gt 0 ]; then pip uninstall -y "${OLD[@]}" 2>&1 | grep -E 'Successfully' || true; fi
rm -rf "$SP"/nvidia_cutlass_dsl*
PINS=()
for p in $CUTLASS_DSL_PACKAGES; do PINS+=("$p==$CUTLASS_DSL_PUBLIC_VERSION"); done
PIP_CONSTRAINT='' pip install --no-deps "${PINS[@]}" 2>&1 | tail -1
pip list --format=freeze | grep -E '^nvidia-cutlass-dsl(-libs-[a-z0-9]+)?==' | tee -a /opt/dsr1/protect.txt | sed 's/^/[10] public cutlass-dsl pin: /'

# TensorRT-LLM requirements-dev.txt under the protected-package constraints.
export PIP_CONSTRAINT=/opt/dsr1/protect.txt
cd "$TRTLLM_SRC"
export LD_LIBRARY_PATH=/opt/hpcx/ompi5/lib:${LD_LIBRARY_PATH:-}   # base MPI for mpi4py
echo "[10] requirements-dev.txt (pypi.org + download.pytorch.org/whl/cu130 only)"
pip install -r requirements-dev.txt 2>&1 | grep -vE '^(Requirement already satisfied|  Downloading|  Using cached|Collecting|  Obtaining|Downloading|  Attempting|  Found existing|    Uninstalling|      Successfully uninstalled)' | tail -30
python3 -c 'import cutlass; print("[10] cutlass", cutlass.__version__, cutlass.__file__)'

# Record the result and check it.
pip list --format=freeze > /opt/dsr1/logs/pip_freeze.10_pydeps.txt
echo "[10] pip diff base -> after step 10:"
diff /opt/dsr1/logs/pip_freeze.00_base.txt /opt/dsr1/logs/pip_freeze.10_pydeps.txt | grep '^[<>]' | sed 's/^/[10]   /' || true
python3 "$RECIPE_DIR/steps/check_pkgs.py" /opt/dsr1/protect.txt | sed 's/^/[10] /'
LOCAL=$(grep -E '^[A-Za-z0-9_.-]+==[^ ]*\+[0-9]{8,}' /opt/dsr1/logs/pip_freeze.10_pydeps.txt | grep -viE '^(torch|torchvision|torchaudio|triton|tensorrt|transformer.engine|apex|torch.tensorrt|torchao)==' || true)
[ -z "$LOCAL" ] || echo "[10] local-version packages not from the base (check that they are public): $LOCAL"
pip check 2>&1 | sed 's/^/[10] pip check: /' | head -40 || true
echo "[10] DONE"
