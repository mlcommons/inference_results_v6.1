# DeepSeek-R1 on Vera Rubin NVL72: TensorRT-LLM + NVIDIA Dynamo image

This recipe builds the server image for the NVIDIA MLPerf Inference v6.1 DeepSeek-R1 submission on Vera Rubin (VR)
NVL72. The image is aarch64 and targets sm_107. How to run the benchmark, and the client and grader images, are
described in the [DeepSeek-R1 README](../../src/nv_mlpinf/benchmarks/deepseek_r1/README.md).

## Contents

All pins are in [`versions.env`](versions.env).

| Input | Pin |
| --- | --- |
| Base image | `nvcr.io/nvidia/tensorrt-llm/devel:1.3.0rc27`, arm64 digest `sha256:d9f42c1471892a847a1b01f108bcd188cd9be256f97da3113525ba678fa4b694` |
| Base contents | CUDA 13.4.1, PyTorch 2.14.0a0+4fdf77b, NCCL 2.30.7, cuDNN 9.25, cuBLAS 13.7, TensorRT 11.2.1.2, HPC-X UCX 1.22.0, NIXL 1.3.0, Nsight Systems 2026.5.1 |
| TensorRT-LLM | Public source: [NVIDIA/TensorRT-LLM, `feat/dsr1_26.08`](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/dsr1_26.08). Select a released commit as described below. Built for `107-real` with `--nvrtc_dynamic_linking --extra-cmake-vars NIXL_ROOT=/opt/nvidia/nvda_nixl`. |
| CuTe DSL | `nvidia-cutlass-dsl`, `nvidia-cutlass-dsl-libs-base`, `-libs-core` and `-libs-cu13`, all 4.8.0, from PyPI |
| FlashInfer | `flashinfer-python==0.6.18.post1` from PyPI, plus a one-line patch (step 45) |
| NVIDIA Dynamo | `ai-dynamo==1.5.1` and `ai-dynamo-runtime==1.5.1` from PyPI, unpatched |
| nats-server | `v2.11.9` GitHub release, sha256-checked |
| etcd | `v3.6.9` GitHub release, sha256-checked |
| Other Python packages | TensorRT-LLM `requirements-dev.txt`, resolved from pypi.org and the PyTorch cu130 index that it names. The base's torch, triton, tensorrt, flashinfer, cuda-python and related packages keep their base versions (`PROTECTED_PKGS`). |

## Build steps

| Step | Script | What it does |
| --- | --- | --- |
| 10 | `steps/10_pydeps.sh` | Installs the Python dependencies from public PyPI, with the FlashInfer and CuTe DSL pins. |
| 30 | `steps/30_trtllm.sh` | Builds and installs the TensorRT-LLM wheel at `TRTLLM_COMMIT`. |
| 45 | `steps/45_rubin_extras.sh` | Patches FlashInfer's trtllm-gen FMHA runner JIT header to map SM107 to the SM100 kernel family: `mSM(getSMVersion() == 107 ? kSM_100 : getSMVersion())`. |
| 40 | `steps/40_dynamo.sh` | Installs the Dynamo wheels with the runtime dependencies they use, nats-server and etcd. |
| 50 | `steps/50_finalize.sh` | Writes `/opt/dsr1/BUILD_INFO` and `/opt/dsr1/pip_freeze.final.txt`. Fails unless the protected base packages are unchanged, every installed `nvidia-cutlass-dsl*` distribution is at the public pin, and Dynamo, nats-server and etcd are present. |

`steps/check_pkgs.py` holds the package checks shared by steps 10 and 50.

## Build

Build TensorRT-LLM from the public [NVIDIA/TensorRT-LLM `feat/dsr1_26.08` branch](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/dsr1_26.08).
The branch must contain the released DeepSeek-R1 FP4 MLA and Push-Combine changes before building; a preview
branch without those changes is insufficient. The DSR1 feature publication is pending at preparation of this recipe.
Once published, resolve and record its full commit SHA before either build path:

```bash
export TRTLLM_REPO=https://github.com/NVIDIA/TensorRT-LLM.git
export TRTLLM_REF=feat/dsr1_26.08
export TRTLLM_COMMIT=$(git ls-remote "$TRTLLM_REPO" "refs/heads/$TRTLLM_REF" | awk '{print $1}')
test "${#TRTLLM_COMMIT}" -eq 40
printf 'TensorRT-LLM source: %s @ %s\n' "$TRTLLM_REPO" "$TRTLLM_COMMIT"
```

`versions.env` defaults to this public repository and branch and requires `TRTLLM_COMMIT` explicitly.
Path B exports it to the build job. Path A passes all three values as build arguments. Keep the resolved SHA with
your build logs; the original source SHA in the historical image table below is provenance, not a public build pin.

Network access: the host that runs the build steps (in Path B, the compute node) needs nvcr.io, pypi.org,
download.pytorch.org, github.com and gitlab.com (TensorRT-LLM's third-party sources, nats-server, etcd) and the
TensorRT-LLM repository; `prep_src.sh` also needs `git-lfs`. Packages without a pin in `versions.env` resolve at build
time, so a rebuild can pick newer patch releases of loose dependencies; `/opt/dsr1/pip_freeze.final.txt` records the
exact set of an image.

### Path B: Slurm + pyxis/enroot

This path built the image used for the results. It needs one Vera Rubin node and a work directory `RWD` on a
filesystem that both the login node and the compute nodes can see.

```bash
cd docker/vr-dsr1-trtllm-dynamo   # from closed/NVIDIA
export RWD=/path/to/work
# 1. TensorRT-LLM source with git-lfs content -> $RWD/src/trtllm-<sha8> (needs network access; a login node works)
bash prep_src.sh
# 2. Image -> $RWD/containers/dsr1-rubin-trtllm-<sha8>-dyn1.5.1.sqsh, plus .sha256 and a .txt summary
sbatch -A <account> -p <partition> -o $RWD/build_%j.log \
  --export=ALL,RWD=$RWD,RECIPE_DIR=$PWD build_dsr1_rubin_recipe.sbatch
```

The script imports the base by its arm64 digest, runs steps 10, 30 and 45 in one writable container, saves it with
pyxis `--container-save`, runs steps 40 and 50 on top of that, then runs `validate_image.sh`. The log ends with
`VALIDATE_RC=0` on success. Only `$RWD` is mounted into the build containers, so a `RECIPE_DIR` outside `$RWD` is
copied to `$RWD/recipe` first. `TAG=<tag>` in `--export` appends `-<tag>` to the image name.

### Path A: docker buildx

Path A has not been used to build the submission image. Steps 30, 40 and 50 import `tensorrt_llm` and
`dynamo.trtllm`, and step 30 queries the GPU, so the build steps need a GPU and `libcuda`; BuildKit (`docker buildx`)
does not give `RUN` steps a GPU. Run on an aarch64 Vera Rubin host with network access (above). `BUILD_JOBS` defaults
to 320; lower it on hosts with fewer cores or less memory.

```bash
cd docker/vr-dsr1-trtllm-dynamo   # from closed/NVIDIA
docker buildx build --platform linux/arm64 -f Dockerfile.dsr1-rubin --load \
  --build-arg TRTLLM_REPO="$TRTLLM_REPO" \
  --build-arg TRTLLM_REF="$TRTLLM_REF" \
  --build-arg TRTLLM_COMMIT="$TRTLLM_COMMIT" \
  --build-arg BUILD_JOBS=$(nproc) -t dsr1-rubin-trtllm:dyn1.5.1 .
```

To run the image with pyxis, push it to a registry or convert it with
`enroot import -o dsr1-rubin-trtllm.sqsh dockerd://dsr1-rubin-trtllm:dyn1.5.1`.

## Check an image

`validate_image.sh` runs inside the image on a Vera Rubin node. Its second argument is a TensorRT-LLM source tree at
the image's commit (`/opt/dsr1/TRTLLM_COMMIT`; `prep_src.sh` creates one).

```bash
srun -N 1 -A <account> -p <partition> --container-image=<image>.sqsh --container-mounts=$RWD:/mnt,$PWD:/recipe \
  bash /recipe/validate_image.sh /mnt/validate /mnt/src/trtllm-<sha8>
```

It checks the sm_107 capability and the TensorRT-LLM cubins and bindings, that `dynamo.trtllm`, `dynamo.frontend` and
`trtllm-serve` start, and runs a FlashInfer rmsnorm and a TensorRT-LLM fp4_quantize on the GPU. It prints
`VALIDATE_FAIL=0` and exits 0 when these pass. The sm_107 unit-test lanes that follow are informational. The image's
provenance is in `/opt/dsr1/BUILD_INFO` and `/opt/dsr1/pip_freeze.final.txt`.

## Server image

The image used for the results. A rebuild does not reproduce the `.sqsh` hash (file timestamps, and loose
dependencies can differ); compare `/opt/dsr1/BUILD_INFO` and `/opt/dsr1/pip_freeze.final.txt` instead.

| Item | Value |
| --- | --- |
| Original image source | `521c43be4e5f92186ab974328375dcbc634b0542` (historical source SHA; the public release has a different commit history) |
| Image file | `dsr1-rubin-trtllm-521c43be-dyn1.5.1.sqsh` |
| sha256 (of the `.sqsh` file) | `f59895d17ffb83ee147e323019745c1601b0ec9ece814ecd25c0cdd9b1f1fca2` |
| Source for rebuilds | [NVIDIA/TensorRT-LLM `feat/dsr1_26.08`](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/dsr1_26.08), after the DSR1 feature publication |
