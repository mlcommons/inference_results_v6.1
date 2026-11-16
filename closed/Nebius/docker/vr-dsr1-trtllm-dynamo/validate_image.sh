#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0
#
# GPU smoke test of the image; run inside the image on a Vera Rubin GPU node. Exits non-zero on a failed check.
# Usage: bash validate_image.sh <out_dir> <tensorrt_llm_source_at_the_image_commit>
set -uo pipefail
O=${1:?}; SRC=${2:?}; mkdir -p "$O"; cd /tmp || exit 2
FAIL=0
step() { echo "===== $* $(date -u +%T)"; }
step "nvidia-smi"; nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.mem,clocks.max.mem --format=csv,noheader
step "BUILD_INFO"; cat /opt/dsr1/BUILD_INFO
step "import tensorrt_llm + capability"
python3 - <<'PY' || FAIL=1
import torch, tensorrt_llm, importlib.metadata as m
print("tensorrt_llm", tensorrt_llm.__version__, tensorrt_llm.__file__)
print("torch", torch.__version__, "cap", torch.cuda.get_device_capability(), "ngpu", torch.cuda.device_count(),
      "sms", torch.cuda.get_device_properties(0).multi_processor_count)
assert torch.cuda.get_device_capability() == (10, 7)
for d in ["flashinfer-python", "nvidia-cutlass-dsl", "triton", "ai-dynamo", "ai-dynamo-runtime", "nixl-cu13"]:
    try:
        print(d, m.version(d))
    except m.PackageNotFoundError:
        print(d, None)
PY
step "cubins in libtensorrt_llm.so"
cuobjdump --list-elf /usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libtensorrt_llm.so 2>/dev/null | grep -o 'sm_[0-9]*[af]*' | sort | uniq -c
step "ucx / nixl"; ucx_info -v | head -2; ls /usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/ | tr '\n' ' '; echo
python3 -c 'import tensorrt_llm.bindings.internal.batch_manager as bm; print("bindings OK")' || FAIL=1
step "dynamo"
python3 -c 'import dynamo.trtllm' && echo "import dynamo.trtllm OK" || FAIL=1
python3 -m dynamo.trtllm --help > /dev/null 2>&1 && echo "dynamo.trtllm --help OK" || { echo "dynamo.trtllm --help FAILED"; FAIL=1; }
python3 -m dynamo.frontend --help > /dev/null 2>&1 && echo "dynamo.frontend --help OK" || FAIL=1
step "cli / binaries"
trtllm-serve --help > /dev/null && echo "trtllm-serve --help OK" || FAIL=1
nats-server --version; etcd --version | head -1; nsys --version || FAIL=1
step "GPU ops: flashinfer rmsnorm, trtllm fp4_quantize"
python3 - <<'PY' || FAIL=1
import torch, tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
import flashinfer
x = torch.randn(64, 7168, dtype=torch.bfloat16, device="cuda"); w = torch.randn(7168, dtype=torch.bfloat16, device="cuda")
y = flashinfer.norm.rmsnorm(x, w, 1e-6)
ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float())
err = (y.float() - ref).abs().max().item(); print("flashinfer rmsnorm max abs err", err); assert err < 0.25
gs = (448 * 6 / x.float().abs().max()).reshape(1)
q, sf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False)
print("trtllm fp4_quantize", tuple(q.shape), q.dtype, tuple(sf.shape)); torch.cuda.synchronize()
PY
step "unit tests (DS-R1 path on sm_107)"
cd "$SRC/tests/unittest" || exit 2
PYTEST=(python3 -m pytest -q -p no:cacheprovider --timeout=900 -o addopts= -W ignore::DeprecationWarning)
run_t() { local name=$1; shift; echo "--- $name"; timeout 1500 "${PYTEST[@]}" "$@" > "$O/pytest_$name.log" 2>&1; local rc=$?
  tail -3 "$O/pytest_$name.log"; echo "rc=$rc"; }
# Informational unit-test lanes (not gating), one process each.
[ -f _torch/misc/test_rubin_kernel_arch.py ] && run_t rubin_kernel_arch _torch/misc/test_rubin_kernel_arch.py
T=_torch/thop/parallel/test_fp4_gemm_quantize.py
[ -f "$T" ] && run_t test_fp4_gemm_quantize "$T" -x -k "not large" --maxfail=3
echo "VALIDATE_FAIL=$FAIL"
exit $FAIL
