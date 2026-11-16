#!/bin/bash

set -eux
set -o pipefail

# Rubin (VR-NVL72) variant of ../../vllm/scripts/build_image.sh, from public sources only (the full list
# is in ../README.md). Three builds, in order:
#   1. flashinfer-python + flashinfer-cubin wheels of the FlashInfer revision
#      (docker/flashinfer-wheels.Dockerfile builder, run on the checkout);
#   2. the vLLM image: vLLM's own docker/Dockerfile Rubin path (target vllm-openai) with those
#      wheels and docker/vllm-dockerfile-rubin.patch;
#   3. the benchmark image: docker/mpi-dynamo-vllm-rubin.Dockerfile on top of it (Dynamo, etcd,
#      NATS, the tuned cute-dsl FMHA exports prefetched into FlashInfer's cubin dir).
# Steps 1-2 are skipped when the local image store already has the vLLM image, unless
# --vllm-force-rebuild. Unlike ../../vllm/scripts/build_image.sh, the vLLM image is never pulled from a
# registry: it is built here whenever the local image store does not have it.

PROJECT_ROOT=$(dirname "${BASH_SOURCE[0]}")/../
PROJECT_ROOT=$(realpath "${PROJECT_ROOT}")

DEFAULT_VLLM_REPO=https://github.com/CentML/vllm.git
vllm_repo=${DEFAULT_VLLM_REPO}

# Pinned vLLM and FlashInfer commits (the q3vl-release-parity branches of CentML/vllm and
# CentML/flashinfer). If a default repository does not have the pinned commit, pass --vllm-repo /
# --flashinfer-repo with a local repository that has it.
DEFAULT_VLLM_REVISION=5bdd885b602efa7a4898f9a5f541dfcb9b04b0fa
vllm_revision=${DEFAULT_VLLM_REVISION}

DEFAULT_FLASHINFER_REPO=https://github.com/CentML/flashinfer.git
flashinfer_repo=${DEFAULT_FLASHINFER_REPO}

DEFAULT_FLASHINFER_REVISION=73a15cbe2e9e882f1d8258fccc0550acdc537a3f
flashinfer_revision=${DEFAULT_FLASHINFER_REVISION}

DEFAULT_VLLM_BUILD_MAX_JOBS=48
vllm_build_max_jobs=${DEFAULT_VLLM_BUILD_MAX_JOBS}

DEFAULT_VLLM_BUILD_NVCC_THREADS=2
vllm_build_nvcc_threads=${DEFAULT_VLLM_BUILD_NVCC_THREADS}

# Must match the two base images (both CUDA 13.4, the newest public CUDA).
DEFAULT_VLLM_BUILD_CUDA_VERSION=13.4
vllm_build_cuda_version=${DEFAULT_VLLM_BUILD_CUDA_VERSION}

# "The first NCCL release with Rubin (SM107) support" (vLLM release pipeline).
DEFAULT_NCCL_VERSION=2.32.3
nccl_version=${DEFAULT_NCCL_VERSION}

# docker.io/pytorch/manylinuxaarch64-builder:cuda13.4 and docker.io/nvidia/cuda:13.4.1-base-ubuntu24.04,
# pinned by OCI index digest (resolved 2026-10-26).
DEFAULT_BUILD_BASE_IMAGE=docker.io/pytorch/manylinuxaarch64-builder@sha256:4977b72cc73116cff1a9705e4040ac5d7a28ae0c0119068b0624b1bbefcce675
build_base_image=${DEFAULT_BUILD_BASE_IMAGE}

DEFAULT_FINAL_BASE_IMAGE=docker.io/nvidia/cuda@sha256:3eb2d952d9cff2d7a4e42a1cedc19cfa610a7f72c3b8428811731dcea5a175fd
final_base_image=${DEFAULT_FINAL_BASE_IMAGE}

DEFAULT_TRITON_REPO=https://github.com/triton-lang/triton.git
triton_repo=${DEFAULT_TRITON_REPO}

DEFAULT_TRITON_REVISION=3f6e41132b5edf639bfb872ad73d4688765e08b8
triton_revision=${DEFAULT_TRITON_REVISION}

# openucx/ucx branch v1.23.x (vLLM's default UCX_REF for the NIXL build) frozen to its commit on 2026-10-26.
DEFAULT_UCX_REF=425fb1bcf0f26a92364247a359130090fdc1ecd0
ucx_ref=${DEFAULT_UCX_REF}

# The Rubin image is arm64 only (VR-NVL72 hosts); an x86 host would emulate the whole build.
DETECTED_ARCH=$(uname -m)
if [[ "${DETECTED_ARCH}" != "aarch64" && "${DETECTED_ARCH}" != "arm64" ]]; then
    echo "[ERROR] ${DETECTED_ARCH} host; build the Rubin image on an arm64 host"
    exit 1
fi
vllm_build_arch=arm64

DEFAULT_VLLM_BUILD_IMAGE_REPO=gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/vllm-openai
vllm_build_image_repo=${DEFAULT_VLLM_BUILD_IMAGE_REPO}

vllm_build_push_flag=""
vllm_build_no_cache_flag=""
result_image_tag=""

DEFAULT_RESULT_IMAGE_REPO=gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv
result_image_repo=${DEFAULT_RESULT_IMAGE_REPO}

result_push_flag=""
result_no_cache_flag=""

function _exit_with_help_msg() {
  cat <<EOF
Build the docker image for NVIDIA's submission to the MLPerf Inference VLM benchmark on Rubin (VR-NVL72),
from public sources only. Needs an arm64 host with rootful Docker and the default buildx (docker) driver.

Usage: ${BASH_SOURCE[0]}
  [-h | --help]     Print this help message.
  [--vllm-repo <vllm_repo>]           The repository to use for vLLM (default: ${DEFAULT_VLLM_REPO}).
  [--vllm-revision <vllm_revision>]   The revision to use for vLLM (default: ${DEFAULT_VLLM_REVISION}).
  [--flashinfer-repo <flashinfer_repo>]           The repository to use for FlashInfer (default: ${DEFAULT_FLASHINFER_REPO}).
  [--flashinfer-revision <flashinfer_revision>]   The revision to use for FlashInfer (default: ${DEFAULT_FLASHINFER_REVISION}).
  [--vllm-build-max-jobs <vllm_build_max_jobs>]   The maximum number of processes to use for vLLM build (default: ${DEFAULT_VLLM_BUILD_MAX_JOBS}).
  [--vllm-build-nvcc-threads <vllm_build_nvcc_threads>]   The number of nvcc threads to use for vLLM build (default: ${DEFAULT_VLLM_BUILD_NVCC_THREADS}).
  [--vllm-build-cuda-version <vllm_build_cuda_version>]   The CUDA version of the vLLM build; must match the two base images (default: ${DEFAULT_VLLM_BUILD_CUDA_VERSION}).
  [--nccl-version <nccl_version>]   The NCCL version installed over torch's (default: ${DEFAULT_NCCL_VERSION}).
  [--build-base-image <image>]   The vLLM build base image (default: ${DEFAULT_BUILD_BASE_IMAGE}).
  [--final-base-image <image>]   The vLLM runtime base image (default: ${DEFAULT_FINAL_BASE_IMAGE}).
  [--triton-repo <triton_repo>]           The repository Triton is built from (default: ${DEFAULT_TRITON_REPO}).
  [--triton-revision <triton_revision>]   The Triton revision (default: ${DEFAULT_TRITON_REVISION}).
  [--ucx-ref <ucx_ref>]   The UCX revision bundled into the NIXL build (default: ${DEFAULT_UCX_REF}).
  [--vllm-build-image-repo <vllm_build_image_repo>]   The docker image repository to use for the resulting vLLM image (default: ${DEFAULT_VLLM_BUILD_IMAGE_REPO}).
  [--result-image-tag <result_image_tag>]   Full result image tag (overrides repo+tag).
  [--vllm-build-push]   Push the resulting vLLM image to the docker image repository (refuses a tag that already exists there).
  [--vllm-force-rebuild]   Force the rebuild of the FlashInfer wheels and the vLLM image.
  [--result-image-repo <result_image_repo>]   The docker image repository to use for the resulting MLPerf Inference VLM benchmark image (default: ${DEFAULT_RESULT_IMAGE_REPO}).
  [--result-push]   Push the resulting MLPerf Inference VLM benchmark image to the docker image repository (refuses a tag that already exists there).
  [--result-force-rebuild]   Force the rebuild of the MLPerf Inference VLM benchmark image.
EOF
  if [ -n "$1" ]; then
    echo "$(tput bold setab 1)$1$(tput sgr0)"
  fi
  exit "$2"
}

while [[ $# -gt 0 ]]; do
    case $1 in
    -h | --help)
        _exit_with_help_msg "" 0
        ;;
    --vllm-repo)
        vllm_repo=$2
        shift
        shift
        ;;
    --vllm-repo=*)
        vllm_repo=${1#*=}
        shift
        ;;
    --vllm-revision)
        vllm_revision=$2
        shift
        shift
        ;;
    --vllm-revision=*)
        vllm_revision=${1#*=}
        shift
        ;;
    --flashinfer-repo)
        flashinfer_repo=$2
        shift
        shift
        ;;
    --flashinfer-repo=*)
        flashinfer_repo=${1#*=}
        shift
        ;;
    --flashinfer-revision)
        flashinfer_revision=$2
        shift
        shift
        ;;
    --flashinfer-revision=*)
        flashinfer_revision=${1#*=}
        shift
        ;;
    --vllm-build-max-jobs)
        vllm_build_max_jobs=$2
        shift
        shift
        ;;
    --vllm-build-max-jobs=*)
        vllm_build_max_jobs=${1#*=}
        shift
        ;;
    --vllm-build-nvcc-threads)
        vllm_build_nvcc_threads=$2
        shift
        shift
        ;;
    --vllm-build-nvcc-threads=*)
        vllm_build_nvcc_threads=${1#*=}
        shift
        ;;
    --vllm-build-cuda-version)
        vllm_build_cuda_version=$2
        shift
        shift
        ;;
    --vllm-build-cuda-version=*)
        vllm_build_cuda_version=${1#*=}
        shift
        ;;
    --nccl-version)
        nccl_version=$2
        shift
        shift
        ;;
    --nccl-version=*)
        nccl_version=${1#*=}
        shift
        ;;
    --build-base-image)
        build_base_image=$2
        shift
        shift
        ;;
    --build-base-image=*)
        build_base_image=${1#*=}
        shift
        ;;
    --final-base-image)
        final_base_image=$2
        shift
        shift
        ;;
    --final-base-image=*)
        final_base_image=${1#*=}
        shift
        ;;
    --triton-repo)
        triton_repo=$2
        shift
        shift
        ;;
    --triton-repo=*)
        triton_repo=${1#*=}
        shift
        ;;
    --triton-revision)
        triton_revision=$2
        shift
        shift
        ;;
    --triton-revision=*)
        triton_revision=${1#*=}
        shift
        ;;
    --ucx-ref)
        ucx_ref=$2
        shift
        shift
        ;;
    --ucx-ref=*)
        ucx_ref=${1#*=}
        shift
        ;;
    --vllm-build-image-repo)
        vllm_build_image_repo=$2
        shift
        shift
        ;;
    --vllm-build-image-repo=*)
        vllm_build_image_repo=${1#*=}
        shift
        ;;
    --result-image-tag)
        result_image_tag=$2
        shift
        shift
        ;;
    --result-image-tag=*)
        result_image_tag=${1#*=}
        shift
        ;;
    --vllm-build-push)
        vllm_build_push_flag="--push"
        shift
        ;;
    --vllm-force-rebuild)
        vllm_build_no_cache_flag="--no-cache"
        shift
        ;;
    --result-image-repo)
        result_image_repo=$2
        shift
        shift
        ;;
    --result-image-repo=*)
        result_image_repo=${1#*=}
        shift
        ;;
    --result-push)
        result_push_flag="--push"
        shift
        ;;
    --result-force-rebuild)
        result_no_cache_flag="--no-cache"
        shift
        ;;
    *)
        _exit_with_help_msg "[ERROR] Unknown option: $1" 1
        ;;
  esac
done

# Repo -> tag component: the GitHub owner_name, or local_<dir> for a local repository.
function _slug() {
    if [[ "$1" == /* || "$1" == .* ]]; then
        echo "local_$(basename "$1" .git)"
    else
        echo "$1" | sed -e 's|https://github.com/||' -e 's|\.git$||' -e 's|/|_|g'
    fi
}
# Revision -> tag component; a full commit SHA is shortened to 12 characters.
function _rev() {
    if [[ "$1" =~ ^[0-9a-f]{40}$ ]]; then
        echo "${1:0:12}"
    else
        echo "$1" | sed -e 's|/|_|g'
    fi
}
function _sha256() {
    if command -v sha256sum > /dev/null; then sha256sum; else shasum -a 256; fi | cut -c1-64
}
# Refuse to push over an image that is already in the registry: only "not found" lets a push go on.
function _refuse_existing_tag() {
    local out
    if out=$(docker manifest inspect "$1" 2>&1); then
        _exit_with_help_msg "[ERROR] $1 already exists in the registry; pass a new tag" 1
    fi
    case "${out}" in
        *"no such manifest"*|*"manifest unknown"*|*"not found"*) ;;
        *) _exit_with_help_msg "[ERROR] cannot tell whether $1 exists (log in to the registry?): ${out}" 1 ;;
    esac
}
# Shallow, tag-less fetch of one revision (a SHA, branch or tag) into a fresh directory: vLLM
# then versions itself 0.1.dev<depth>+g<sha>.d<build date> (the build edits the checkout).
function _fetch_source() {
    local dir=$1 repo=$2 revision=$3 depth=$4 flag=$5
    if [ -d "${repo}" ] && [ -d "${dir}" ] && [ "$(cd "${repo}" && pwd -P)" = "$(cd "${dir}" && pwd -P)" ]; then
        _exit_with_help_msg "[ERROR] ${dir} is the ${flag} repository itself; run from another directory" 1
    fi
    rm -rf "${dir}"
    git init -q "${dir}"
    git -C "${dir}" remote add origin "${repo}"
    if ! git -C "${dir}" fetch --depth "${depth}" --no-tags origin "${revision}"; then
        _exit_with_help_msg "[ERROR] cannot fetch ${revision} from ${repo}; if it is not pushed yet, pass ${flag} <a local repository that has it>" 1
    fi
    git -C "${dir}" -c advice.detachedHead=false checkout -q FETCH_HEAD
}

# The vLLM image carries the FlashInfer wheels, so its tag names both revisions, plus a digest of
# every other input of steps 1-2 so that changing one of them never reuses an older image.
vllm_build_inputs_digest=$(
    {
        echo "${vllm_build_cuda_version} ${nccl_version} ${build_base_image} ${final_base_image}"
        echo "${triton_repo} ${triton_revision} ${ucx_ref}"
        cat "${PROJECT_ROOT}/docker/vllm-dockerfile-rubin.patch" "${PROJECT_ROOT}/docker/flashinfer-wheels.Dockerfile" \
            "${PROJECT_ROOT}/docker/build_flashinfer_wheels.sh" "${PROJECT_ROOT}/docker/flashinfer-build-requirements.txt"
    } | _sha256 | cut -c1-8
)
vllm_build_image_tag=${vllm_build_arch}_cuda${vllm_build_cuda_version}_rubin_$(_slug "${vllm_repo}")-$(_rev "${vllm_revision}")_$(_slug "${flashinfer_repo}")-$(_rev "${flashinfer_revision}")_${vllm_build_inputs_digest}
if [ ${#vllm_build_image_tag} -gt 128 ]; then
    vllm_build_image_tag="${vllm_build_image_tag:0:128}"
fi
vllm_build_image_tag=${vllm_build_image_repo}:${vllm_build_image_tag}

if [ -z "${result_image_tag}" ]; then
    dynamo_version=$(sed -n 's/^ARG DYNAMO_VERSION=//p' "${PROJECT_ROOT}/docker/mpi-dynamo-vllm-rubin.Dockerfile")
    result_image_tag=${vllm_build_arch}_cuda${vllm_build_cuda_version}_rubin_$(_slug "${vllm_repo}")-$(_rev "${vllm_revision}")_$(_slug "${flashinfer_repo}")-$(_rev "${flashinfer_revision}")_${vllm_build_inputs_digest}_ai-dynamo-${dynamo_version}
    if [ ${#result_image_tag} -gt 128 ]; then
        result_image_tag="${result_image_tag:0:128}"
    fi
    result_image_tag=${result_image_repo}:${result_image_tag}
fi

# Check the push targets before the multi-hour build, not after it.
if [ "${result_push_flag}" = "--push" ]; then
    _refuse_existing_tag "${result_image_tag}"
fi
if [ "${vllm_build_push_flag}" = "--push" ] && { ! docker image inspect "${vllm_build_image_tag}" > /dev/null 2>&1 || [ "${vllm_build_no_cache_flag}" = "--no-cache" ]; }; then
    _refuse_existing_tag "${vllm_build_image_tag}"
fi

if ! docker image inspect "${vllm_build_image_tag}" > /dev/null 2>&1 || [ "${vllm_build_no_cache_flag}" = "--no-cache" ]; then
    VLLM_CLONE_DIR=${VLLM_CLONE_DIR:-${PWD}/vllm}
    FLASHINFER_CLONE_DIR=${FLASHINFER_CLONE_DIR:-${PWD}/flashinfer}
    _fetch_source "${VLLM_CLONE_DIR}" "${vllm_repo}" "${vllm_revision}" 2 --vllm-repo
    vllm_commit=$(git -C "${VLLM_CLONE_DIR}" rev-parse HEAD)

    ####################################################
    # 1. FlashInfer wheels (flashinfer-python, -cubin) #
    ####################################################
    _fetch_source "${FLASHINFER_CLONE_DIR}" "${flashinfer_repo}" "${flashinfer_revision}" 1 --flashinfer-repo
    flashinfer_commit=$(git -C "${FLASHINFER_CLONE_DIR}" rev-parse HEAD)
    # Both wheels carry <version.txt>+<local version>; flashinfer requires the two to match.
    flashinfer_commit_time=$(git -C "${FLASHINFER_CLONE_DIR}" log -1 --format=%ct HEAD)
    flashinfer_local_version=rubin.fmha.$(git -C "${FLASHINFER_CLONE_DIR}" log -1 --format=%cd --date=format:%Y%m%d HEAD).g${flashinfer_commit:0:8}
    flashinfer_wheel_version=$(cat "${FLASHINFER_CLONE_DIR}/version.txt")+${flashinfer_local_version}

    # The wheels are kept (keyed by commit and builder inputs) and reused: the vLLM build
    # bind-mounts requirements/, so new wheel bytes on a retry would invalidate its whole layer
    # cache. SOURCE_DATE_EPOCH fixes their zip timestamps when they are rebuilt.
    flashinfer_wheel_dir=${FLASHINFER_WHEEL_DIR:-${PWD}/flashinfer-wheels}/${flashinfer_commit:0:12}-${vllm_build_inputs_digest}
    if [ ! -s "${flashinfer_wheel_dir}/flashinfer_python-${flashinfer_wheel_version}-py3-none-any.whl" ] \
            || [ ! -s "${flashinfer_wheel_dir}/flashinfer_cubin-${flashinfer_wheel_version}-py3-none-any.whl" ] \
            || [ "${vllm_build_no_cache_flag}" = "--no-cache" ]; then
        git -C "${FLASHINFER_CLONE_DIR}" submodule update --init --depth 1 3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl \
            || git -C "${FLASHINFER_CLONE_DIR}" submodule update --init 3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl
        flashinfer_builder_context=$(mktemp -d)
        trap 'rm -rf "${flashinfer_builder_context}"' EXIT
        cp "${PROJECT_ROOT}/docker/flashinfer-build-requirements.txt" "${PROJECT_ROOT}/docker/build_flashinfer_wheels.sh" \
            "${flashinfer_builder_context}/"
        cp "${VLLM_CLONE_DIR}/requirements/rubin-prerelease.txt" "${flashinfer_builder_context}/rubin-prerelease.txt"
        cp "${FLASHINFER_CLONE_DIR}/requirements.txt" "${flashinfer_builder_context}/flashinfer-requirements.txt"
        flashinfer_builder_tag=q3vl-rubin/flashinfer-wheel-builder:${flashinfer_commit:0:12}-${vllm_build_inputs_digest}
        DOCKER_BUILDKIT=1 docker build \
            ${vllm_build_no_cache_flag} \
            --build-arg BUILD_BASE_IMAGE="${build_base_image}" \
            --platform "linux/${vllm_build_arch}" \
            --tag "${flashinfer_builder_tag}" \
            --progress plain \
            -f "${PROJECT_ROOT}/docker/flashinfer-wheels.Dockerfile" \
            "${flashinfer_builder_context}"
        rm -rf "${flashinfer_builder_context}"
        trap - EXIT

        rm -rf "${flashinfer_wheel_dir}"
        mkdir -p "${flashinfer_wheel_dir}"
        docker run --rm --pull=never \
            --user "$(id -u):$(id -g)" \
            -e HOME=/tmp \
            -e USER="$(id -un)" \
            -e LOGNAME="$(id -un)" \
            -e NVIDIA_VISIBLE_DEVICES=void \
            -e SOURCE_DATE_EPOCH="${flashinfer_commit_time}" \
            -e FLASHINFER_COMMIT="${flashinfer_commit}" \
            -e FLASHINFER_LOCAL_VERSION="${flashinfer_local_version}" \
            -e FLASHINFER_WHEEL_VERSION="${flashinfer_wheel_version}" \
            -v "${FLASHINFER_CLONE_DIR}:/src/flashinfer" \
            -v "${flashinfer_wheel_dir}:/out" \
            "${flashinfer_builder_tag}"
    fi

    ###################################
    # 2. vLLM image (Rubin build path) #
    ###################################
    # The wheels go where the vLLM build reads them (requirements/ is bound into its build).
    mkdir -p "${VLLM_CLONE_DIR}/requirements/local-wheel-inputs"
    cp "${flashinfer_wheel_dir}"/flashinfer_python-"${flashinfer_wheel_version}"-py3-none-any.whl \
        "${flashinfer_wheel_dir}"/flashinfer_cubin-"${flashinfer_wheel_version}"-py3-none-any.whl \
        "${VLLM_CLONE_DIR}/requirements/local-wheel-inputs/"
    {
        echo ""
        echo "# FlashInfer ${flashinfer_commit} (built by q3vl/rubin/scripts/build_image.sh)."
        echo "flashinfer-python @ file:///tmp/vllm-requirements/local-wheel-inputs/flashinfer_python-${flashinfer_wheel_version}-py3-none-any.whl"
        echo "flashinfer-cubin @ file:///tmp/vllm-requirements/local-wheel-inputs/flashinfer_cubin-${flashinfer_wheel_version}-py3-none-any.whl"
    } >> "${VLLM_CLONE_DIR}/requirements/rubin-prerelease.txt"
    git -C "${VLLM_CLONE_DIR}" apply "${PROJECT_ROOT}/docker/vllm-dockerfile-rubin.patch"

    # vLLM's aarch64 Rubin release arguments: the Dockerfile's default torch_cuda_arch_list, and
    # BUILD_NIXL=true instead of INSTALL_KV_CONNECTORS (the PyPI NIXL wheel has no NIXL EP for
    # torch 2.16). VLLM_IMAGE_TAG is recorded in the image, so it carries no registry host.
    cd "${VLLM_CLONE_DIR}"
    DOCKER_BUILDKIT=1 docker build \
        --pull \
        ${vllm_build_push_flag} \
        ${vllm_build_no_cache_flag} \
        --build-arg max_jobs="${vllm_build_max_jobs}" \
        --build-arg nvcc_threads="${vllm_build_nvcc_threads}" \
        --build-arg RUN_WHEEL_CHECK=false \
        --build-arg CUDA_VERSION="${vllm_build_cuda_version}" \
        --build-arg NCCL_VERSION="${nccl_version}" \
        --build-arg BUILD_BASE_IMAGE="${build_base_image}" \
        --build-arg FINAL_BASE_IMAGE="${final_base_image}" \
        --build-arg INSTALL_RUBIN_PRERELEASE=true \
        --build-arg BUILD_NIXL=true \
        --build-arg UCX_REF="${ucx_ref}" \
        --build-arg FLASHINFER_LOCAL_WHEELS=1 \
        --build-arg TRITON_INSTALL_FROM_SOURCE_REPO="${triton_repo}" \
        --build-arg TRITON_INSTALL_FROM_SOURCE_REVISION="${triton_revision}" \
        --build-arg VLLM_BUILD_COMMIT="${vllm_commit}" \
        --build-arg VLLM_IMAGE_TAG="${vllm_build_image_tag##*/}" \
        --platform "linux/${vllm_build_arch}" \
        --tag "${vllm_build_image_tag}" \
        --target vllm-openai \
        --progress plain \
        -f docker/Dockerfile \
        .

    cd -
    rm -rf "${VLLM_CLONE_DIR}" "${FLASHINFER_CLONE_DIR}"
fi

if ! docker image inspect "${vllm_build_image_tag}" > /dev/null 2>&1; then
    echo "[ERROR] ${vllm_build_image_tag} is not in the local image store (the default docker buildx driver is required)"
    exit 1
fi

####################################
# 3. MLPerf Q3VL image (Rubin)     #
####################################
DOCKER_BUILDKIT=1 docker build \
    --ulimit nofile=65536:65536 \
    ${result_push_flag} \
    ${result_no_cache_flag} \
    --build-arg VLLM_BASE_IMAGE="${vllm_build_image_tag}" \
    --build-arg ARCH="${vllm_build_arch}" \
    --build-arg VLLM_REVISION="${vllm_revision}" \
    --build-arg FLASHINFER_REVISION="${flashinfer_revision}" \
    --platform "linux/${vllm_build_arch}" \
    --tag "${result_image_tag}" \
    --progress plain \
    -f "${PROJECT_ROOT}/docker/mpi-dynamo-vllm-rubin.Dockerfile" \
    "${PROJECT_ROOT}/docker"
