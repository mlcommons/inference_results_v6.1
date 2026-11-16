#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0
#
# Step 45: patch the FlashInfer trtllm-gen FMHA runner JIT header to map SM107 to the SM100 kernel family.
set -euo pipefail
: "${RECIPE_DIR:?}"
source "$RECIPE_DIR/versions.env"
SP=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
F="$SP/$FLASHINFER_FMHA_RUNNER_REL"
if [ ! -f "$F" ]; then
  echo "[45] $F not found (flashinfer layout differs)"; exit 2
elif grep -Fq 'mSM(getSMVersion() == 107 ? kSM_100 : getSMVersion())' "$F"; then
  echo "[45] flashinfer fmhaRunner already patched"
elif grep -Fq 'mSM(getSMVersion())' "$F"; then
  cp "$F" /opt/dsr1/logs/fmhaRunner.cuh.orig
  sed -i 's/mSM(getSMVersion())/mSM(getSMVersion() == 107 ? kSM_100 : getSMVersion())/' "$F"
  grep -Fq 'mSM(getSMVersion() == 107 ? kSM_100 : getSMVersion())' "$F"
  echo "[45] flashinfer fmhaRunner patched: SM107 -> kSM_100"; diff /opt/dsr1/logs/fmhaRunner.cuh.orig "$F" | sed 's/^/[45]   /' || true
else
  echo "[45] fmhaRunner has no 'mSM(getSMVersion())' anchor; current mSM lines:"; grep -n 'mSM(' "$F" | head -5 | sed 's/^/[45]   /'
fi
echo "[45] DONE"
