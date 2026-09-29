# Setup - gpt-oss-120b

Setup steps run from the **repository root** (the `mlperf-inference/` folder). Once the
container is started, the run/package workflow is in
[`README.md`](../README.md) under **Running Experiments**.

> **MI350X vs MI355X:** these commands use the **MI355X** configs. On MI350X, replace
> `mi355x` with `mi350x` everywhere.

## 1. Download the dataset

```bash
bash setup/gpt-oss-120b/download_dataset.sh
```

## 2. Download (and quantize) the model

Preferred: download the already-quantized model with your Hugging Face token (or drop
`--download-prequantized` to quantize from scratch). Output:
`.../gpt-oss-120b/fp4_quantized/`.

```bash
HUGGINGFACE_ACCESS_TOKEN="<your HF token>"
bash setup/gpt-oss-120b/download_model_fp4.sh --token "$HUGGINGFACE_ACCESS_TOKEN" --download-prequantized
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
bash setup/gpt-oss-120b/build_docker.sh
export EXTRA_ARGS="--rm --workdir /lab-mlperf-inference/submission"
bash setup/gpt-oss-120b/start_docker.sh
```

> `gpt-oss-120b` and `llama2-70b-99` share the same container. If you've already built
> and started it for one, you can run both benchmarks from the same shell.

Next: [`README.md`](../README.md) -> **Running Experiments**.

# Running GPT-OSS-120B

Run from `/lab-mlperf-inference/submission` inside the benchmark container.

## Offline

```bash
python3 submission.py --model gpt-oss-120b experiment \
  --scenario Offline \
  --model-conf ../code/gpt-oss-120b/offline_mi355x.yaml \
  --user-conf ../code/gpt-oss-120b/user_mi355x.conf

