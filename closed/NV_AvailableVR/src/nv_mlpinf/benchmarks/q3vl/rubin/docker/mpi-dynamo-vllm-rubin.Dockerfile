# Rubin (VR-NVL72) variant of the generic ../../vllm/docker/mpi-dynamo-vllm.Dockerfile, from public sources only.
#
# VLLM_BASE_IMAGE is the vllm-openai image that ../scripts/build_image.sh builds with vLLM's own
# docker/Dockerfile Rubin path: INSTALL_RUBIN_PRERELEASE=true (torch 2.16 nightly cu134, Triton
# from source), BUILD_NIXL=true (NIXL 1.5.0 + NIXL EP built for that torch, with a bundled UCX),
# CUDA 13.4, NCCL 2.32.3, and the CentML FlashInfer wheels. Same skeleton as the generic
# Dockerfile (Dynamo -> etcd/NATS), with these differences:
#   - No apt packages, Rust, maturin, or nixl/triton installs: Dynamo comes as wheels (nothing to
#     compile), and the base already has NIXL and Triton built from source for its torch. Hence
#     also no TRITON_PTXAS_PATH / TRITON_PTXAS_BLACKWELL_PATH (vLLM's Rubin path links Triton's
#     ptxas to the system CUDA 13.4 one).
#   - No Nsight Systems (the generic Dockerfile installs 2025.4.1): profiling on Rubin needs nsys
#     2026.5.2, which no public CUDA image or apt repository shipped in 2026-10, so profiling
#     needs a separate image that has it.
#   - Dynamo is the PyPI nightly ai-dynamo / ai-dynamo-runtime, hash-pinned and --no-deps, plus a
#     fixed dependency set (dynamo-deps.txt, not hash-pinned) for the names the base does not
#     provide. The 1.6.0 release imports vllm.v1.engine.utils.get_engine_zmq_addresses, which
#     this vLLM no longer has; the generic Dockerfile builds a Dynamo branch from source instead.
#   - pydantic 2.13.0: ai-dynamo-runtime requires pydantic<=2.13.5, the base resolves a newer one.
#   - No transformers pin: vLLM's requirements choose it.
#   - etcd from the same release URL, NATS from its GitHub release (the generic Dockerfile pipes
#     binaries.nats.dev into sh; NATS_VER is the bare version here), both sha256-checked.
#   - The tuned Rubin cute-dsl FMHA exports (FlashInfer ArtifactPath.DSL_FMHA, <cpu>/sm_107a) are
#     prefetched into FlashInfer's cubin dir with FlashInfer's own download code
#     (prefetch_dsl_fmha.py), so loading them needs no network and no
#     FLASHINFER_DSL_FMHA_LOCAL_DIR.
#   - Every install step proves which packages it changed (pkgset.py), and a CPU-only
#     validation (validate_image.py --build) ends the build.
#   - ENV UV_BREAK_SYSTEM_PACKAGES=1 (uv on Ubuntu 24.04's system Python); the working directory
#     is /tmp; no Dynamo source tree under /vllm-workspace.
ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

ARG ARCH=arm64
ARG VLLM_REVISION=
ARG FLASHINFER_REVISION=
ARG PYDANTIC_VERSION=2.13.0
ARG PYDANTIC_SHA256=ab0078b90da5f3e2fd2e71e3d9b457ddcb35d0350854fbda93b451e28d56baaf
ARG PYDANTIC_CORE_VERSION=2.46.0
ARG PYDANTIC_CORE_SHA256=f26a1032bcce6ca4b4670eb3f7d8195bd0a8b8f255f1307823e217ca3cfa7c27
ARG DYNAMO_VERSION=1.6.0.dev20261019
ARG DYNAMO_SHA256=d8291b3fa3895a3e53949cf47c3b7a305973663631f6e72de43aecd3a13b8b68
ARG DYNAMO_RUNTIME_SHA256=21c3eb9eeeaa77ab04e42a74b498cd90a8ae6740fe798ce5bd1495cde3c1aa45
ARG ETCD_VER=v3.6.6
ARG ETCD_DOWNLOAD_URL=https://storage.googleapis.com/etcd
ARG ETCD_SHA256=8a15f5427c111ff4692753682374970fb68878401d946c2c28bdad6857db652f
ARG ETCD_TARGET_DIR=/opt/etcd
ARG NATS_VER=v2.11.6
ARG NATS_DOWNLOAD_URL=https://github.com/nats-io/nats-server/releases/download
ARG NATS_SHA256=4fe3c40bddc58e31012f55865df0dcd24012995d2893a4d64a0ae5cb147d7a36
# The FMHA exports come from FlashInfer's own (public) cubin repository unless
# DSL_FMHA_REPOSITORY names an http(s) mirror in the published layout.
ARG DSL_FMHA_SM_ARCH=sm_107a
ARG DSL_FMHA_REPOSITORY=

LABEL org.opencontainers.image.title="Q3VL MLPerf server for Rubin (VR-NVL72)" \
      q3vl.vllm.revision="${VLLM_REVISION}" \
      q3vl.flashinfer.revision="${FLASHINFER_REVISION}" \
      q3vl.dynamo.version="${DYNAMO_VERSION}" \
      q3vl.dynamo.wheel.ai-dynamo.sha256="${DYNAMO_SHA256}" \
      q3vl.dynamo.wheel.ai-dynamo-runtime.sha256="${DYNAMO_RUNTIME_SHA256}" \
      q3vl.etcd.version="${ETCD_VER}" \
      q3vl.nats.version="${NATS_VER}"

# PYTHONDONTWRITEBYTECODE is set per RUN below (the build tools must not add .pyc files to the
# installed packages), not as image ENV.
ENV UV_BREAK_SYSTEM_PACKAGES=1
WORKDIR /opt/q3vl-rubin
COPY pkgset.py dynamo_deps_split.py prefetch_dsl_fmha.py validate_image.py dynamo-deps.txt ./
WORKDIR /tmp

#################################
# pydantic for ai-dynamo-runtime #
#################################
# The image ENV UV_OVERRIDE (vLLM's unhashed NCCL pin) breaks --require-hashes and does not
# matter under --no-deps. A temporary uv cache keeps /opt/uv/cache unchanged.
RUN set -eux; \
    export PYTHONDONTWRITEBYTECODE=1; \
    export UV_CACHE_DIR="$(mktemp -d /tmp/uv-cache.XXXXXX)"; \
    cd /opt/q3vl-rubin; \
    uv pip freeze --system > base-freeze.txt; \
    python3 pkgset.py snapshot pkgset-base.json; \
    printf 'pydantic==%s --hash=sha256:%s\npydantic-core==%s --hash=sha256:%s\n' \
        "${PYDANTIC_VERSION}" "${PYDANTIC_SHA256}" "${PYDANTIC_CORE_VERSION}" "${PYDANTIC_CORE_SHA256}" \
        > pydantic-pins.txt; \
    env -u UV_OVERRIDE uv pip install --system --no-deps --require-hashes -r pydantic-pins.txt; \
    python3 pkgset.py expect pkgset-base.json \
        --replaced "pydantic==${PYDANTIC_VERSION}" "pydantic-core==${PYDANTIC_CORE_VERSION}"; \
    rm -rf "${UV_CACHE_DIR}"

##################
# Install Dynamo #
##################
# The dependency set for every name the base does not already provide (base versions win and
# are recorded), resolved under the base's versions, then the two nightly wheels.
RUN set -eux; \
    export PYTHONDONTWRITEBYTECODE=1; \
    export UV_CACHE_DIR="$(mktemp -d /tmp/uv-cache.XXXXXX)"; \
    cd /opt/q3vl-rubin; \
    uv pip list --system --format=freeze | grep -E '^[A-Za-z0-9]' > /tmp/base-constraints.txt; \
    python3 dynamo_deps_split.py dynamo-deps.txt dynamo-deps-install.txt dynamo-deps-kept.txt; \
    python3 pkgset.py snapshot /tmp/pkgset-before-dynamo-deps.json; \
    if [ -s dynamo-deps-install.txt ]; then \
        uv pip install --system --constraint /tmp/base-constraints.txt -r dynamo-deps-install.txt; \
    fi; \
    python3 pkgset.py expect /tmp/pkgset-before-dynamo-deps.json \
        --added-file dynamo-deps-install.txt --allow-new > dynamo-deps-pkgset.txt \
        || { cat dynamo-deps-pkgset.txt; exit 1; }; \
    cat dynamo-deps-pkgset.txt; \
    printf 'ai-dynamo-runtime==%s --hash=sha256:%s\nai-dynamo==%s --hash=sha256:%s\n' \
        "${DYNAMO_VERSION}" "${DYNAMO_RUNTIME_SHA256}" "${DYNAMO_VERSION}" "${DYNAMO_SHA256}" \
        > dynamo-wheels.txt; \
    python3 pkgset.py snapshot /tmp/pkgset-before-dynamo.json; \
    env -u UV_OVERRIDE uv pip install --system --no-deps --require-hashes -r dynamo-wheels.txt; \
    python3 pkgset.py expect /tmp/pkgset-before-dynamo.json \
        --added "ai-dynamo==${DYNAMO_VERSION}" "ai-dynamo-runtime==${DYNAMO_VERSION}"; \
    rm -rf "${UV_CACHE_DIR}" /tmp/pkgset-before-dynamo-deps.json /tmp/pkgset-before-dynamo.json \
        /tmp/base-constraints.txt

#########################
# Install NATS and ETCD #
#########################
RUN set -eux; \
    curl -fsSL --retry 3 -o /tmp/etcd.tar.gz "${ETCD_DOWNLOAD_URL}/${ETCD_VER}/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz"; \
    echo "${ETCD_SHA256}  /tmp/etcd.tar.gz" | sha256sum -c -; \
    mkdir -p ${ETCD_TARGET_DIR}; \
    tar -xzf /tmp/etcd.tar.gz -C ${ETCD_TARGET_DIR} --strip-components=1 --no-same-owner; \
    curl -fsSL --retry 3 -o /tmp/nats.tar.gz "${NATS_DOWNLOAD_URL}/${NATS_VER}/nats-server-${NATS_VER}-linux-${ARCH}.tar.gz"; \
    echo "${NATS_SHA256}  /tmp/nats.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/nats.tar.gz -C /tmp --no-same-owner; \
    install -m 0755 "/tmp/nats-server-${NATS_VER}-linux-${ARCH}/nats-server" /opt/nats-server; \
    rm -rf /tmp/etcd.tar.gz /tmp/nats.tar.gz "/tmp/nats-server-${NATS_VER}-linux-${ARCH}"; \
    ${ETCD_TARGET_DIR}/etcd --version; \
    /opt/nats-server --version
ENV PATH="/opt/:/opt/etcd/:$PATH"

#############################################
# Prefetch the tuned cute-dsl FMHA exports  #
#############################################
RUN set -eux; \
    export PYTHONDONTWRITEBYTECODE=1; \
    python3 /opt/q3vl-rubin/prefetch_dsl_fmha.py --sm "${DSL_FMHA_SM_ARCH}" \
        ${DSL_FMHA_REPOSITORY:+--repository "${DSL_FMHA_REPOSITORY}"} \
        --record /opt/q3vl-rubin/dsl-fmha-prefetch.json

#######################
# Build-time checks   #
#######################
RUN set -eux; \
    export PYTHONDONTWRITEBYTECODE=1; \
    export UV_CACHE_DIR="$(mktemp -d /tmp/uv-cache.XXXXXX)"; \
    python3 /opt/q3vl-rubin/validate_image.py --build --dynamo-version "${DYNAMO_VERSION}" \
        --report /opt/q3vl-rubin/build-validation.json; \
    uv pip list --system --format=freeze > /opt/q3vl-rubin/pip-freeze.txt; \
    rm -rf "${UV_CACHE_DIR}"

ENTRYPOINT ["/bin/bash"]
