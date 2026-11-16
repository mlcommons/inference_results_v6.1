#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0
#
# Step 40: NVIDIA Dynamo from PyPI (ai-dynamo + ai-dynamo-runtime, unpatched) with the runtime dependencies used by
# dynamo.frontend and dynamo.trtllm, plus the nats-server and etcd release binaries (sha256-checked).
set -euo pipefail
: "${RECIPE_DIR:?}"
source "$RECIPE_DIR/versions.env"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_INDEX_URL="$PUBLIC_PYPI" PIP_EXTRA_INDEX_URL=
export PIP_CONSTRAINT=/opt/dsr1/constraints.txt
W=${BUILD_TMP:-/tmp}/dsr1-dynamo; rm -rf "$W"; mkdir -p "$W"; cd "$W"
ARCH=$(uname -m); GOARCH=arm64
[ "$ARCH" = x86_64 ] && GOARCH=amd64
PIP_CONSTRAINT='' pip install --no-deps "ai-dynamo==${DYNAMO_VERSION}" "ai-dynamo-runtime==${DYNAMO_VERSION}" 2>&1 | tail -1

# ai-dynamo base requirements (no extras, minus unused frameworks), installed under the step-30 freeze except aiohttp.
REQ=$(python3 - <<'PY'
import importlib.metadata as m
from packaging.requirements import Requirement
skip = {"tensorrt-llm", "ai-dynamo-runtime", "kubernetes", "kubernetes-asyncio", "aisimulate", "jax", "jaxlib", "vllm", "sglang"}
out = []
for r in m.requires("ai-dynamo") or []:
    q = Requirement(r)
    if q.marker and not q.marker.evaluate({"extra": ""}):
        continue
    if q.name.lower() in skip:
        continue
    out.append(str(q).split(";")[0])
print("\n".join(out))
PY
)
echo "[40] ai-dynamo base requirements:"; echo "$REQ" | sed 's/^/[40]   /'
grep -v '^aiohttp==' /opt/dsr1/constraints.txt > "$W/constraints.dynamo.txt"
echo "$REQ" > "$W/req.txt"
PIP_CONSTRAINT="$W/constraints.dynamo.txt" pip install -r "$W/req.txt" 2>&1 | grep -E 'Successfully|ERROR' || true
pip check 2>&1 | grep -i dynamo | sed 's/^/[40] pip check: /' || true

# nats-server and etcd.
curl -sSfL -o nats.tgz "https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-linux-${GOARCH}.tar.gz"
[ "$GOARCH" = arm64 ] && echo "${NATS_ARM64_SHA256}  nats.tgz" | sha256sum -c -
tar xzf nats.tgz && install -m 0755 "nats-server-${NATS_VERSION}-linux-${GOARCH}/nats-server" /usr/local/bin/nats-server
curl -sSfL -o etcd.tgz "https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-${GOARCH}.tar.gz"
[ "$GOARCH" = arm64 ] && echo "${ETCD_ARM64_SHA256}  etcd.tgz" | sha256sum -c -
tar xzf etcd.tgz && install -m 0755 "etcd-${ETCD_VERSION}-linux-${GOARCH}/etcd" "etcd-${ETCD_VERSION}-linux-${GOARCH}/etcdctl" /usr/local/bin/
nats-server --version; etcd --version | head -1

python3 -c 'import importlib.metadata as m; print("[40] ai-dynamo", m.version("ai-dynamo"), "runtime", m.version("ai-dynamo-runtime"))'
python3 -m dynamo.frontend --help > /dev/null && echo "[40] dynamo.frontend --help OK"
python3 -c 'import dynamo.trtllm' && echo "[40] import dynamo.trtllm OK"
pip list --format=freeze > /opt/dsr1/logs/pip_freeze.40_dynamo.txt
echo "[40] pip diff step 30 -> 40:"; diff /opt/dsr1/logs/pip_freeze.30_trtllm.txt /opt/dsr1/logs/pip_freeze.40_dynamo.txt | grep '^[<>]' | sed 's/^/[40]   /' || true
cd / && rm -rf "$W"
echo "[40] DONE"
