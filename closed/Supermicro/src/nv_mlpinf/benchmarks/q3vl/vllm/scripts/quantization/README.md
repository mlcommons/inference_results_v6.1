# Qwen3-VL-235B NVFP4 + FP8-KV quantization (ModelOpt)

Produces the MLPerf q3vl submission checkpoint:

> `nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.1-FP8-KV` (served from the repo `main`).

Recipe: NVFP4 (W4A4) on every linear layer (static MSE weight scales plus dynamic NVFP4 inputs) with a
calibrated per-tensor FP8 KV cache, calibrated on the Shopify catalogue (the benchmark's own data). Built
with [NVIDIA TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

This supersedes the previous llm-compressor NVFP4-only checkpoint. Adding the FP8 KV cache and recovering
accuracy with MSE weight calibration matches the FP16-KV baseline within noise at higher throughput, while
clearing the 0.78240 accuracy target.

| checkpoint | F1 (Shopify) | QPS |
| --- | --- | --- |
| FP16-KV NVFP4-only baseline | 0.78725 | 64.5 |
| NVFP4 + calibrated FP8 KV (this) | 0.78708 | 67.2 |

## Files

* `quantize_qwen3vl_nvfp4_fp8kv.py`: one command (build calibration, run PTQ, export a vLLM-loadable checkpoint).
* `nvfp4_default_mse-kv_fp8.yaml`: the ModelOpt recipe.

## Environment

Quantization runs in its own Python environment (it does not use the vLLM base or submission container).
Create a fresh environment with the exact versions used for the submission checkpoint, so the result is
reproducible:

```bash
# ModelOpt source (the script wraps examples/llm_ptq/hf_ptq.py); pinned to the submission commit.
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
git -C TensorRT-Model-Optimizer checkout 0f61c984a27f

uv venv --python 3.12 .venv-modelopt
source .venv-modelopt/bin/activate
SETUPTOOLS_SCM_PRETEND_VERSION=0.46.0.dev0 uv pip install --torch-backend=auto \
    "torch==2.12.0" "torchvision==0.27.0" \
    ./TensorRT-Model-Optimizer[hf] "compressed-tensors==0.17.1" fire datasets pydantic
```

Pinned versions: `nvidia-modelopt 0.46.0.dev0` (commit `0f61c98`), `torch 2.12.0+cu130`,
`transformers 5.9.0`, `torchvision 0.27.0`, `compressed-tensors 0.17.1`. torch 2.12 or newer is required on
Blackwell (SM100). torchvision is needed by the VLM processor.

## Run

On a 4x GB200/GB300 node (the 235B model is sharded across the 4 GPUs):

```bash
source .venv-modelopt/bin/activate
python quantize_qwen3vl_nvfp4_fp8kv.py \
    --modelopt-repo ./TensorRT-Model-Optimizer \
    --output ./Qwen3-VL-235B-A22B-Instruct-NVFP4-FP8-KV
```

The export is directly loadable by the submission vLLM: the script adds the router-gate exclusion and copies
the VLM processor configs.

## ModelOpt shims (pending upstream)

The script adds four small Qwen3-VL-MoE workarounds for current ModelOpt gaps: expert routing, an async CUDA
race guard (`CUDA_LAUNCH_BLOCKING=1`), Shopify calibration injection, and a post-export metadata fix. Each is
documented inline and tracked for upstreaming, so a future ModelOpt release removes the need for them.
