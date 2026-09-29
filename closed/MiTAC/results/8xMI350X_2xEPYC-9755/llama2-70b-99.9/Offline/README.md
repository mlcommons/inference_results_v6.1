# Setup - llama2-70b-99

Setup steps run from the **repository root** (the `mlperf-inference/` folder). Once the
container is started, the run/package workflow is in
[`README.md`](../README.md) under **Running Experiments**.

> **MI350X vs MI355X:** these commands use the **MI355X** configs. On MI350X, replace
> `mi355x` with `mi350x` everywhere.

## 1. Download the dataset

```bash
bash setup/llama2-70b-99/download_dataset.sh
```

## 2. Download (and quantize) the model

If the unquantized model already sits at
`/data/inference/model/llama2-70b-chat-hf/orig/`, no Hugging Face token is needed;
otherwise pass one to download it first. Output:
`.../llama2-70b-chat-hf/fp4_quantized/`.

```bash
# If the model is already present:
MODEL_OPTION="_fp4"
bash setup/llama2-70b-99/download_model$MODEL_OPTION.sh

# If it needs to be downloaded first:
HUGGINGFACE_ACCESS_TOKEN="<your HF token>"
MODEL_OPTION="_fp4"
bash setup/llama2-70b-99/download_model$MODEL_OPTION.sh --token "$HUGGINGFACE_ACCESS_TOKEN"
```

## 3. Runtime tunables

Applies extra machine tuning. Run once after each reboot, before your first
performance test:

```bash
bash setup/runtime_tunables.sh
```

## 4. Build and start Docker

Build the image, then start the container. You land in
`/lab-mlperf-inference/submission`, where the Running Experiments workflow begins.

```bash
bash setup/llama2-70b-99/build_docker.sh
export EXTRA_ARGS="--rm --workdir /lab-mlperf-inference/submission"
bash setup/llama2-70b-99/start_docker.sh
```

> `llama2-70b-99` and `gpt-oss-120b` share the same container. If you've already built
> and started it for one, you can run both benchmarks from the same shell.

Next: [`README.md`](../README.md) -> **Running Experiments**.
