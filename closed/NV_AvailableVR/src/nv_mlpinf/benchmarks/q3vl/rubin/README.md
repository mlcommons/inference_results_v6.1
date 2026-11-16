# Q3VL server image for Rubin (VR-NVL72)

This folder builds the MLPerf Qwen3-VL-235B-A22B NVFP4 server image for VR-NVL72 (sm_107, arm64) from
public sources. It is the Rubin counterpart of [`../vllm/scripts/build_image.sh`](../vllm/scripts/build_image.sh)
and [`../vllm/docker/mpi-dynamo-vllm.Dockerfile`](../vllm/docker/mpi-dynamo-vllm.Dockerfile).
[`scripts/build_image.sh`](scripts/build_image.sh) produces two images (plus an intermediate FlashInfer
wheel-builder image):

- the **vLLM image**: vLLM's own `docker/Dockerfile` Rubin path (target `vllm-openai`), with
  FlashInfer wheels built from the pinned FlashInfer commit;
- the **benchmark image**: Dynamo, etcd and NATS on top of the vLLM image, with the tuned Rubin
  cute-dsl FMHA exports prefetched into FlashInfer's cubin dir.

The image that the [`VR-NVL72_VR-288GB_aarch64x72`](../../../../../configs/qwen3_vl_235b_a22b/VR-NVL72_VR-288GB_aarch64x72/README.md) recipes pin (`mlperf-inf-mm-q3vl-nv:arm64_cuda134_q3vlpublic_202610272018`) is in NVIDIA's internal registry and is not publicly available. Following this README builds the same software from public sources, apart from the unpinned packages noted under [Inputs](#inputs); the image digest differs.

| File | Purpose |
|---|---|
| `scripts/build_image.sh` | Runs the three build steps below |
| `docker/flashinfer-wheels.Dockerfile`, `docker/build_flashinfer_wheels.sh`, `docker/flashinfer-build-requirements.txt` | Builder for the `flashinfer-python` / `flashinfer-cubin` wheels |
| `docker/vllm-dockerfile-rubin.patch` | Two opt-in build args for vLLM's `docker/Dockerfile` (applied to the build checkout only) |
| `docker/mpi-dynamo-vllm-rubin.Dockerfile` | The benchmark image on top of the vLLM image |
| `docker/pkgset.py`, `docker/dynamo_deps_split.py`, `docker/dynamo-deps.txt` | Dynamo dependency set and the per-step package-change checks |
| `docker/prefetch_dsl_fmha.py` | Downloads FlashInfer's pinned cute-dsl FMHA exports into its cubin dir at build time |
| `docker/validate_image.py` | `--build` (CPU, the last build step) and `--gpu` (on a GPU host) checks |

## Prerequisites

- An arm64 host with rootful Docker 27+ and the default buildx (`docker`) driver. The script reads
  the vLLM image back from the local image store, so other buildx drivers do not work.
- About 200 GB of free disk and 48+ cores (`--vllm-build-max-jobs` defaults to 48).
- Network access to the hosts listed under [Inputs](#inputs). No credentials are needed to build.
- To push, `docker login` to the target registry.

## Inputs

| Component | Source | Pin |
|---|---|---|
| Build base | `docker.io/pytorch/manylinuxaarch64-builder:cuda13.4` | `@sha256:4977b72cc73116cff1a9705e4040ac5d7a28ae0c0119068b0624b1bbefcce675` |
| Runtime base | `docker.io/nvidia/cuda:13.4.1-base-ubuntu24.04` | `@sha256:3eb2d952d9cff2d7a4e42a1cedc19cfa610a7f72c3b8428811731dcea5a175fd` |
| vLLM | `https://github.com/CentML/vllm` (`q3vl-release-parity` branch) | `5bdd885b602efa7a4898f9a5f541dfcb9b04b0fa` |
| FlashInfer | `https://github.com/CentML/flashinfer` (`q3vl-release-parity` branch) | `73a15cbe2e9e882f1d8258fccc0550acdc537a3f`; wheels `0.7.2+rubin.fmha.20261028.g73a15cbe` |
| torch / torchvision / torchaudio | `https://download.pytorch.org/whl/nightly/cu134` (vLLM `requirements/rubin-prerelease.txt`) | `2.16.0.dev20261009+cu134` / `0.30.0.dev20261009+cu134` / `2.11.0.dev20261009+cu134` |
| CUDA libraries | `https://pypi.nvidia.com`, `https://pypi.org`, CUDA apt repository `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/` | cuBLAS 13.8.1.7, cuDNN 9.26.0.51, NVSHMEM 3.7.2 (torch's pins); NCCL 2.32.3 (`--nccl-version`); `cuda-nvcc-13-4` and related packages resolved at build time |
| Triton | `https://github.com/triton-lang/triton`, source build (LLVM prebuilt from Triton's public blob store) | `3f6e41132b5edf639bfb872ad73d4688765e08b8` |
| NIXL + NIXL EP | `https://github.com/ai-dynamo/nixl` `v1.5.0`, with UCX from `https://github.com/openucx/ucx` | UCX `425fb1bcf0f26a92364247a359130090fdc1ecd0` |
| CuTe DSL | `https://pypi.org/project/nvidia-cutlass-dsl/` (vLLM `requirements/cuda.txt`) | `4.8.0` |
| FlashInfer cubins (trtllm-gen FMHA, BMM, GEMM, DeepGEMM) | FlashInfer's public cubin repository `https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/` | the `flashinfer/artifacts.py` pins of the FlashInfer commit |
| Tuned Rubin cute-dsl FMHA exports | the same repository, `ArtifactPath.DSL_FMHA` leaf `<pin>/fmha/cute-dsl/aarch64/sm_107a/` | pin `96e6da5f40123e53146d8072dfd42f633ad6591a`, 690 `.so`; the FlashInfer commit pins the manifest sha256 |
| Dynamo | `https://pypi.org/project/ai-dynamo/` nightly | `1.6.0.dev20261019`, both wheels sha256-pinned |
| pydantic | `https://pypi.org` | `2.13.0` / `pydantic-core 2.46.0`, sha256-pinned |
| etcd | `https://storage.googleapis.com/etcd` | `v3.6.6`, sha256-pinned |
| NATS server | `https://github.com/nats-io/nats-server/releases` | `v2.11.6`, sha256-pinned |

The build also downloads from these public hosts: the Ubuntu (`ports.ubuntu.com`) and AlmaLinux 8 /
EPEL package mirrors, the deadsnakes PPA, `astral.sh` (uv), `bootstrap.pypa.io` (get-pip),
python-build-standalone on GitHub, `sh.rustup.rs` / `static.rust-lang.org` / crates.io (vLLM's Rust
build), and the GitHub repositories that vLLM's csrc build fetches with CMake FetchContent.

Not every input is pinned: the package mirrors, uv, get-pip, rustup, the PyPI ranges in vLLM's and
FlashInfer's requirements, and the transitive dependencies of `docker/dynamo-deps.txt` resolve at
build time, so a rebuild can differ from the image the `VR-NVL72_VR-288GB_aarch64x72` recipes pin in such
packages.

## Build

From `closed/NVIDIA/src/nv_mlpinf/benchmarks/q3vl/rubin`:

```bash
bash scripts/build_image.sh
```

The script works from any directory except `q3vl/`, whose `vllm/` folder it would delete: it
replaces `./vllm` and `./flashinfer` with fresh checkouts and keeps the wheels in
`./flashinfer-wheels/`; set `VLLM_CLONE_DIR`, `FLASHINFER_CLONE_DIR` or `FLASHINFER_WHEEL_DIR` to use
other paths. If a default repository does not have the pinned commit, the fetch stops with an error;
pass local clones that have it: `--vllm-repo /path/to/vllm --flashinfer-repo /path/to/flashinfer`.

1. **FlashInfer wheels.** A shallow fetch of `--flashinfer-revision` (with the cutlass, spdlog and
   cccl submodules), then both wheels built in the builder image as the calling user. Nothing is
   compiled; the `flashinfer-cubin` build downloads the trtllm-gen cubins the commit pins from
   FlashInfer's public cubin repository. The wheels are kept in `./flashinfer-wheels/` and reused.
2. **vLLM image.** A shallow, tag-less fetch of `--vllm-revision` (vLLM then versions itself
   `0.1.dev2+g<short sha>.d<build date>`; the date suffix comes from the patched checkout), then
   vLLM's `docker/Dockerfile`, target `vllm-openai`, with the aarch64 Rubin arguments: `INSTALL_RUBIN_PRERELEASE=true`, `BUILD_NIXL=true`, `CUDA_VERSION=13.4`,
   `NCCL_VERSION=2.32.3`, Triton from source and the two digest-pinned bases. The FlashInfer wheels
   enter through two `file://` lines appended to `requirements/rubin-prerelease.txt`.
   `docker/vllm-dockerfile-rubin.patch` adds `UCX_REF` (pins the UCX that NIXL bundles) and
   `FLASHINFER_LOCAL_WHEELS=1` (skips `flashinfer download-kernels`, which would replace the local
   `flashinfer-cubin` wheel with a public one of another version). Steps 1-2 are skipped when the
   local image store already has the vLLM image, unless `--vllm-force-rebuild`.
3. **Benchmark image.** `docker/mpi-dynamo-vllm-rubin.Dockerfile` on the vLLM image: pydantic and
   the Dynamo wheels (hash-pinned, with the dependency set in `docker/dynamo-deps.txt`), etcd and
   NATS, then `docker/prefetch_dsl_fmha.py`. FlashInfer's bulk cubin download carries only the FMHA
   manifests, not the `.so` exports, so the prefetch runs FlashInfer's own per-file download for
   every manifest entry. The image then loads every export with no network access;
   `FLASHINFER_NO_DOWNLOAD=1` is safe, and `FLASHINFER_DSL_FMHA_LOCAL_DIR` and
   `FLASHINFER_CUBIN_DIR` must stay unset. `ENTRYPOINT` is `/bin/bash`, `etcd` and `nats-server` are
   on `PATH`, and the image has no Nsight Systems.

### Options

Each option with a value accepts `--option value` and `--option=value`; `--help` prints the list.

| Option | Default | Effect |
|---|---|---|
| `--vllm-repo`, `--vllm-revision` | `https://github.com/CentML/vllm.git`, `5bdd885b602efa7a4898f9a5f541dfcb9b04b0fa` | vLLM source (a URL or a local path) |
| `--flashinfer-repo`, `--flashinfer-revision` | `https://github.com/CentML/flashinfer.git`, `73a15cbe2e9e882f1d8258fccc0550acdc537a3f` | FlashInfer source (a URL or a local path) |
| `--vllm-build-max-jobs`, `--vllm-build-nvcc-threads` | `48`, `2` | vLLM compile parallelism |
| `--vllm-build-cuda-version` | `13.4` | CUDA version of the vLLM build; must match the two base images |
| `--nccl-version` | `2.32.3` | NCCL installed over torch's |
| `--build-base-image`, `--final-base-image` | the two bases in [Inputs](#inputs) | vLLM build and runtime base images |
| `--triton-repo`, `--triton-revision` | `https://github.com/triton-lang/triton.git`, `3f6e41132b5edf639bfb872ad73d4688765e08b8` | Triton source build |
| `--ucx-ref` | `425fb1bcf0f26a92364247a359130090fdc1ecd0` | UCX revision bundled into the NIXL build |
| `--vllm-build-image-repo` | see `--help` | Registry repository of the vLLM image |
| `--vllm-build-push` | off | Push the vLLM image (only when steps 1-2 run) |
| `--vllm-force-rebuild` | off | Rebuild the FlashInfer wheels and the vLLM image without cache |
| `--result-image-repo` | see `--help` | Registry repository of the benchmark image |
| `--result-image-tag` | derived | Full benchmark image reference (overrides repository and tag) |
| `--result-push` | off | Push the benchmark image |
| `--result-force-rebuild` | off | Rebuild the benchmark image without cache |

The vLLM image is tagged
`arm64_cuda13.4_rubin_<vllm repo>-<rev>_<flashinfer repo>-<rev>_<inputs>`, where `<inputs>` is a
digest of the other inputs of steps 1-2 (NCCL, bases, Triton, UCX, the patch, the FlashInfer
builder). The benchmark image adds `_ai-dynamo-<version>`.

## Validate

At build time, and fatal on failure: every sha256 pin, the package changes of each install step
(`docker/pkgset.py`), the FMHA leaf against FlashInfer's pinned manifest, and
`docker/validate_image.py --build` (versions, `nixl_ep_cpp_torch216`, the vLLM NIXL connector, the
Dynamo and worker imports). `uv pip check` is recorded, not enforced; it reports 4 expected lines:
torch's `nvidia-nccl-cu13==2.30.7` (overridden by `NCCL_VERSION`), `triton==3.9.0+gitfc8d31a5`
(replaced by the source Triton), the `nvidia-cusparselt-cu13` platform tag, and ai-dynamo's
`aisimulate==0.13.0.dev2` (`docker/dynamo-deps.txt` keeps 0.12.0).

On a GPU host, run the smoke checks (torch and a bf16 matmul on every GPU, NCCL all-reduce, every
FMHA export loaded through FlashInfer's pinned path with `FLASHINFER_NO_DOWNLOAD=1`, NIXL EP and UCX
loopbacks, Dynamo and worker imports, `/proc/self/maps`). Add `--fmha-host-only` on a non-Rubin GPU.
The command prints `VALIDATE_IMAGE_PASS` on success:

```bash
docker run --rm --gpus all -v "$PWD:/out" --entrypoint python3 <image> \
    /opt/q3vl-rubin/validate_image.py --gpu --report /out/validate.json
```

## Push

Add `--result-push` (and `--vllm-build-push` for the vLLM image) with `--result-image-repo` (and
`--vllm-build-image-repo`) set to your registry, or pass the full reference with
`--result-image-tag`. Before building, the script refuses a tag that already exists in the
registry. The [`VR-NVL72_VR-288GB_aarch64x72`](../../../../../configs/qwen3_vl_235b_a22b/VR-NVL72_VR-288GB_aarch64x72/VLLM)
recipes read the image from `CONTAINER_IMAGE` in their config; pass
`--set CONTAINER_IMAGE=<image>` to `sflow batch` to run them on a pushed image.

## If a step loses network access

The csrc build clones about 15 GitHub repositories, and step 3 downloads the 690 FMHA exports one
file at a time. `docker/prefetch_dsl_fmha.py` retries each failed export download 3 more times,
60 s apart, after FlashInfer's own retries. If a build still fails on a download, rerun the same
command without `--vllm-force-rebuild` and `--result-force-rebuild`: BuildKit reuses every finished
layer, the FlashInfer wheels are reused, and once the vLLM image exists only step 3 runs again.
