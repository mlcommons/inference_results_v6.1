#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0
#
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -J dsr1-vr-image
#
# Build the image on one Vera Rubin node with Slurm + pyxis/enroot (no docker):
#   0. import the base image by its arm64 digest (or reuse BASE_SQSH)
#   1. steps 10, 30, 45 on the base -> OUT_NODYN
#   2. steps 40, 50 on OUT_NODYN -> OUT
#   3. validate_image.sh; image sha256, validation summary and pip list -> ${OUT%.sqsh}.txt
# Pinned inputs come from versions.env; the TensorRT-LLM source from prep_src.sh. Only $RWD is mounted into the build
# containers (at /mnt, an existing directory of the base), so every input and output lives under $RWD.
# Usage: sbatch -A <account> -p <partition> -o <log> --export=ALL,RWD=<work dir>,RECIPE_DIR=<this dir>[,TAG=<tag>] \
#          build_dsr1_rubin_recipe.sbatch
set -uo pipefail
RWD=${RWD:?set RWD=<work dir on a shared filesystem>}
RECIPE_DIR=${RECIPE_DIR:?set RECIPE_DIR=<this recipe directory>}
case "$RECIPE_DIR/" in
  "$RWD"/*) ;;
  *) rm -rf "$RWD/recipe"; mkdir -p "$RWD/recipe"; cp -r "$RECIPE_DIR"/. "$RWD/recipe/"
     echo "staged $RECIPE_DIR -> $RWD/recipe"; RECIPE_DIR=$RWD/recipe;;
esac
source "$RECIPE_DIR/versions.env"
S8=${TRTLLM_COMMIT:0:8}; TAG=${TAG:-}
BASE_SQSH=${BASE_SQSH:-$RWD/containers/base/trtllm-devel-${BASE_IMAGE_TAG}-arm64.sqsh}
OUT=${OUT:-$RWD/containers/dsr1-rubin-trtllm-${S8}-dyn${DYNAMO_VERSION}${TAG:+-$TAG}.sqsh}
OUT_NODYN=${OUT_NODYN:-${OUT%-dyn*}-nodyn${TAG:+-$TAG}.sqsh}
SRC=${SRC:-$RWD/src/trtllm-$S8}
BUILD_DIR=${BUILD_DIR:-$RWD/build/trtllm-$S8${TAG:+-$TAG}}
WHEEL_OUT=${WHEEL_OUT:-$RWD/wheels/trtllm-$S8${TAG:+-$TAG}}
LOGD=$RWD/logs/image/recipe_${SLURM_JOB_ID:-local}; mkdir -p "$LOGD" "$(dirname "$BASE_SQSH")"
CT=/mnt; MOUNTS="$RWD:$CT"; c() { echo "$CT${1#"$RWD"}"; }
for p in "$SRC" "$BUILD_DIR" "$WHEEL_OUT"; do
  case "$p" in "$RWD"/*) ;; *) echo "PATH_OUTSIDE_RWD: $p (only $RWD is mounted into the build containers)"; exit 2;; esac
done
echo "BASE=${BASE_IMAGE_REPO}:${BASE_IMAGE_TAG}@${BASE_IMAGE_ARM64_DIGEST} SRC=$SRC OUT=$OUT"; date; hostname
[ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" = "$TRTLLM_COMMIT" ] || { echo "SOURCE_NOT_AT_COMMIT ($SRC): run prep_src.sh"; exit 4; }
T0=$(date +%s)

# 0. Base image, imported anonymously (no registry credentials).
if [ ! -s "$BASE_SQSH" ]; then
  T=$(mktemp -d /tmp/enroot-dsr1.XXXX); mkdir -p "$T/conf"
  TMPDIR=$T ENROOT_CONFIG_PATH=$T/conf ENROOT_CACHE_PATH=$T/cache ENROOT_DATA_PATH=$T/data ENROOT_MAX_PROCESSORS=64 \
    enroot import -o "$BASE_SQSH" "docker://nvcr.io#${BASE_IMAGE_REPO#nvcr.io/}@${BASE_IMAGE_ARM64_DIGEST}" > "$LOGD/enroot_import.log" 2>&1 \
    || { echo BASE_IMPORT_FAILED; tail -20 "$LOGD/enroot_import.log"; rm -rf "$T"; exit 3; }
  rm -rf "$T"; echo "base imported: $(ls -la "$BASE_SQSH")"
fi

# Run steps in a writable container started from <in-image> and save the result as <save-as>.
session() {  # session <in-image> <save-as> <steps...>
  local IN=$1 SAVE=$2; shift 2
  srun --ntasks=1 --container-image="$IN" --container-writable --container-remap-root --no-container-mount-home \
    --container-mounts="$MOUNTS" --container-save="$SAVE" \
    env RECIPE_DIR="$(c "$RECIPE_DIR")" TRTLLM_SRC="$(c "$SRC")" BUILD_DIR="$(c "$BUILD_DIR")" WHEEL_OUT="$(c "$WHEEL_OUT")" \
        TRTLLM_COMMIT="$TRTLLM_COMMIT" TRTLLM_REF="$TRTLLM_REF" TRTLLM_REPO="$TRTLLM_REPO" \
        BUILD_TMP=/tmp BUILD_JOBS="${BUILD_JOBS:-320}" BUILD_LOG="$(c "$LOGD")/build_wheel.log" LOGD="$(c "$LOGD")" STEPS="$*" \
    bash -c 'set -euo pipefail; export HOME=/tmp/dsr1-home; mkdir -p $HOME $LOGD/opt_dsr1_logs
      trap "cp -r /opt/dsr1/logs/. $LOGD/opt_dsr1_logs/ 2>/dev/null || true" EXIT
      for s in $STEPS; do echo "===== $s $(date -u +%FT%TZ)"; bash "$RECIPE_DIR/steps/$s.sh"; done
      rm -rf /tmp/dsr1-* /root/.cache/pip 2>/dev/null || true' 2>&1 | tee -a "$LOGD/build.log"
  local RC=${PIPESTATUS[0]}
  [ "$RC" -eq 0 ] && [ ! -s "$SAVE" ] && { echo "EXPORT_FAILED: $SAVE missing or empty after srun rc=0"; RC=9; }
  echo "SESSION_RC=$RC steps=[$*] -> $SAVE elapsed $(( $(date +%s) - T0 )) s"; date
  [ "$RC" -ne 0 ] && { echo "SESSION_FAILED: removing $SAVE"; rm -f "$SAVE"; }
  return "$RC"
}

# 1-2. Build.
if [ ! -s "$OUT" ]; then
  if [ ! -s "$OUT_NODYN" ]; then
    session "$BASE_SQSH" "$OUT_NODYN" 10_pydeps 30_trtllm 45_rubin_extras || exit 5
    echo "NODYN_IMAGE_READY $OUT_NODYN"
  fi
  session "$OUT_NODYN" "$OUT" 40_dynamo 50_finalize || exit 6
fi

# 3. Validate and record.
sha256sum "$OUT" > "$OUT.sha256" & SHAPID=$!
srun --overlap --ntasks=1 --container-image="$OUT" --no-container-mount-home --container-mounts="$MOUNTS" \
  bash "$(c "$RECIPE_DIR")/validate_image.sh" "$(c "$LOGD")/validate" "$(c "$SRC")" > "$LOGD/validate.log" 2>&1
VRC=$?
srun --overlap --ntasks=1 --container-image="$OUT" --no-container-mount-home cat /opt/dsr1/pip_freeze.final.txt > "$LOGD/pip_freeze.final.txt" 2>/dev/null
wait $SHAPID
INFO=${OUT%.sqsh}.txt
{
  echo "image: $OUT"; echo "sha256: $(cut -d' ' -f1 "$OUT.sha256")"; echo "size_bytes: $(stat -c %s "$OUT")"
  echo "built: $(date -u +%FT%TZ), logs: $LOGD, total $(( $(date +%s) - T0 )) s"
  echo "validate_image.sh rc=$VRC"; grep -E '^(--- |rc=|=====)|passed|failed|OK|cap' "$LOGD/validate.log" | tail -40
  echo "--- pip list --format=freeze"; cat "$LOGD/pip_freeze.final.txt"
} > "$INFO"
echo "VALIDATE_RC=$VRC INFO=$INFO"; date
