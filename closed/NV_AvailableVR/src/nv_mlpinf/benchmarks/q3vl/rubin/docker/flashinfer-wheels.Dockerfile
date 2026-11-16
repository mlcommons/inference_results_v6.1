# Builder for the FlashInfer wheels (flashinfer-python + flashinfer-cubin) that the Rubin vLLM
# image installs. scripts/build_image.sh runs it with the FlashInfer checkout mounted at
# /src/flashinfer and the output directory at /out (build_flashinfer_wheels.sh). The build
# context holds four files that build_image.sh assembles:
#   rubin-prerelease.txt               vLLM's torch / cuda-python pins for the Rubin path
#   flashinfer-requirements.txt        FlashInfer's requirements.txt
#   flashinfer-build-requirements.txt  vLLM's CuTe DSL / tvm-ffi pins and the build backends'
#   build_flashinfer_wheels.sh
# torch and the runtime deps are here only because `import flashinfer` (done by the
# flashinfer-cubin build backend) needs them; nothing is compiled.
ARG BUILD_BASE_IMAGE
FROM ${BUILD_BASE_IMAGE}
RUN dnf install -y --setopt=install_weak_deps=False git \
    && dnf clean all && rm -rf /var/cache/dnf
RUN /opt/python/cp312-cp312/bin/python3.12 -m venv /opt/fi-venv
ENV PATH=/opt/fi-venv/bin:$PATH \
    PIP_CONFIG_FILE=/dev/null \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
COPY rubin-prerelease.txt flashinfer-requirements.txt flashinfer-build-requirements.txt /opt/fi-req/
# torch/torchvision/torchaudio/cuda-python/cuda-bindings exactly as vLLM's Rubin pins (the
# nixl pin of that file is not needed to build FlashInfer and is dropped).
RUN set -eux; \
    grep -v '^nixl' /opt/fi-req/rubin-prerelease.txt > /tmp/rubin-torch.txt; \
    python3 -m pip install --pre -r /tmp/rubin-torch.txt; \
    python3 -m pip install -r /opt/fi-req/flashinfer-requirements.txt -r /opt/fi-req/flashinfer-build-requirements.txt; \
    python3 -m pip freeze > /opt/fi-req/builder-freeze.txt; \
    python3 -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
RUN git config --system --add safe.directory '*'
COPY build_flashinfer_wheels.sh /usr/local/bin/build_flashinfer_wheels.sh
WORKDIR /tmp
ENTRYPOINT ["bash", "/usr/local/bin/build_flashinfer_wheels.sh"]
