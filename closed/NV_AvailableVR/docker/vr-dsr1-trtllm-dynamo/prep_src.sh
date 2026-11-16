#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0
#
# Check out TRTLLM_COMMIT from TRTLLM_REPO (versions.env) with its git-lfs content into $RWD/src/trtllm-<sha8>.
# Usage: RWD=<work dir> bash prep_src.sh
set -euo pipefail
RWD=${RWD:?set RWD=<work dir on a shared filesystem>}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/versions.env"
D=$RWD/src/trtllm-${TRTLLM_COMMIT:0:8}
[ -e "$D" ] && { echo "exists: $D (HEAD $(git -C "$D" rev-parse HEAD))"; exit 0; }
export GIT_LFS_SKIP_SMUDGE=1
git init -q "$D"; cd "$D"; git remote add origin "$TRTLLM_REPO"
git fetch -q --depth 1 origin "$TRTLLM_COMMIT" || git fetch -q origin "$TRTLLM_REF"
git checkout -q --detach "$TRTLLM_COMMIT"
git lfs install --local > /dev/null
git -c lfs.concurrenttransfers=16 lfs pull --exclude="**/x86_64-linux-gnu/**,docs/**,triton_backend/**"
[ "$(git rev-parse HEAD)" = "$TRTLLM_COMMIT" ] || { echo "WRONG HEAD"; exit 3; }
echo "lfs pointers outside excluded paths: $(git lfs ls-files | grep ' - ' | grep -vcE 'x86_64-linux-gnu|^.{13}(docs|triton_backend)/' || true)"
echo "READY $D @ $(git rev-parse HEAD)"
