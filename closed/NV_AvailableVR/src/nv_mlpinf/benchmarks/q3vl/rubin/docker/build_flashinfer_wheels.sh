#!/usr/bin/env bash
# Entry point of the flashinfer-wheels.Dockerfile builder (no GPU needed). Builds the paired
# FlashInfer wheels of the checkout mounted at /src/flashinfer into /out:
#   flashinfer-python   pure packaging (no CUDA compile: BUILD_NVEP=0, FLASHINFER_BUILD_NO_PIP=1)
#   flashinfer-cubin    downloads every cubin the checkout's flashinfer/artifacts.py pins from
#                       FlashInfer's public cubin repository (its default). For the cute-dsl FMHA
#                       pin it fetches only the checksums.txt manifests, never the .so exports;
#                       mpi-dynamo-vllm-rubin.Dockerfile prefetches those.
# Both carry the same version, FLASHINFER_WHEEL_VERSION (flashinfer requires an exact match).
set -euo pipefail
: "${FLASHINFER_COMMIT:?}" "${FLASHINFER_LOCAL_VERSION:?}" "${FLASHINFER_WHEEL_VERSION:?}"
src=/src/flashinfer
out=/out
test "$(git -C "$src" rev-parse HEAD)" = "$FLASHINFER_COMMIT"
test -z "$(git -C "$src" status --porcelain --untracked-files=no)"
unset FLASHINFER_CUBINS_REPOSITORY FLASHINFER_CUBIN_DIR
export BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 CUDA_MAJOR=13
export FLASHINFER_LOCAL_VERSION
rm -f "$out"/*.whl
cd /tmp
python3 -m pip wheel --no-deps --no-build-isolation "$src" -w "$out"
FLASHINFER_CUBIN_DIR="$src/flashinfer-cubin/flashinfer_cubin/cubins" \
    python3 -m pip wheel --no-deps --no-build-isolation "$src/flashinfer-cubin" -w "$out"
cat "$src/flashinfer/_build_meta.py" "$src/flashinfer-cubin/flashinfer_cubin/_build_meta.py"
grep -q "__git_commit__ = \"$FLASHINFER_COMMIT\"" "$src/flashinfer/_build_meta.py"
grep -q "__version__ = \"$FLASHINFER_WHEEL_VERSION\"" "$src/flashinfer/_build_meta.py"
grep -q "__git_version__ = \"$FLASHINFER_COMMIT\"" "$src/flashinfer-cubin/flashinfer_cubin/_build_meta.py"
grep -q "__version__ = \"$FLASHINFER_WHEEL_VERSION\"" "$src/flashinfer-cubin/flashinfer_cubin/_build_meta.py"
for package in flashinfer_python flashinfer_cubin; do
    test -s "$out/${package}-${FLASHINFER_WHEEL_VERSION}-py3-none-any.whl"
done
python3 - "$out/flashinfer_cubin-${FLASHINFER_WHEEL_VERSION}-py3-none-any.whl" <<'PY'
import collections, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
c = collections.Counter(n.rsplit(".", 1)[-1] for n in z.namelist() if "/cubins/" in n)
dsl = [n for n in z.namelist() if "/fmha/cute-dsl/" in n]
print("cubin wheel entries by extension:", dict(c))
print("cute-dsl entries:", len(dsl), "(.so:", sum(n.endswith(".so") for n in dsl), ")")
assert not any(n.endswith(".so") for n in dsl), "unexpected cute-dsl .so in the cubin wheel"
PY
sha256sum "$out"/*.whl
echo FLASHINFER_WHEELS_READY
