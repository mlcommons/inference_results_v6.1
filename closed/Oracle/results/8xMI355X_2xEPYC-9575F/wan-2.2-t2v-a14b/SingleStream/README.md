# MLPerf Inference wan-2.2-t2v-a14b — SingleStream

Reproduction instructions for the **SingleStream** scenario submitted under
`closed/AMD/results/8xMI355X_2xEPYC_9575F/wan-2.2-t2v-a14b/`.

Harness source, Docker build pins, and data-fetch metadata ship in this
submission at `closed/AMD/src/wan-2.2-t2v-a14b/` (see
`REPRODUCIBILITY.json` in that directory).

## Prerequisites

* 8× AMD Instinct MI355X (or equivalent ROCm + xDiT stack)
* Docker with ROCm device access (`/dev/kfd`, `/dev/dri`)
* Hugging Face cache with `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (bind-mount
  at `/hf_cache` via `launch.sh`)

## Source

Harness source ships under `closed/AMD/src/wan-2.2-t2v-a14b/` in the
submission tree. From that directory on the **host**:

```bash
./launch.sh --build
./launch.sh            # interactive shell at /workspace/wan-harness
```

Inside the container:

```bash
python3 -m tools.fetch_data
```

## Single-scenario replay

Inside the container:

```bash
# Performance
./scripts/run_scenario.sh \
    --backend wan22 --scenario SingleStream --mode performance

# Accuracy (writes mp4 artefacts under the run directory)
./scripts/run_scenario.sh \
    --backend wan22 --scenario SingleStream --mode accuracy

# VBench scoring → accuracy.txt
with-vbench python -m tools.run_vbench \
    runs/wan22/latest/SingleStream/accuracy

# TEST04 compliance (Wan-specific audit config)
./scripts/verify_compliance.sh --scenario SingleStream
```

## Full submission matrix

To reproduce both Offline and SingleStream from the archived harness source,
inside the container:

```bash
./scripts/run_all.sh --backend wan22
```
