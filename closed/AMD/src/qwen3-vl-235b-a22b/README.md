# qwen3-vl-235b-a22b — AMD IMG-S recipe (build → quantize → run)

Self-contained recipe for the AMD closed/available Q3VL submission. The image builds and the
benchmark runs entirely from this directory (`docker/` + `scripts/` incl. `start_docker.sh` + `configs/`) —
no repo checkout needed. See "Contents of this directory" at the end for the file manifest.

## 1. Build the image
```bash
cd submission/src/qwen3-vl-235b-a22b/
docker build -f docker/Dockerfile --network=host -t amd-mlperf6.1-qwen3-vl-235b-a22b .
```
## 1.5 Launch the container (AMD / ROCm)
```bash
export HF_CACHE=/path/to/hf-cache   
export HF_TOKEN=hf_...              
scripts/start_docker.sh             # mounts src/qwen3-vl-235b-a22b/ at /work and submission/ at /submission
```

## 2. Quantize the model — ONE Quark pass (MXFP4 W4A4 LLM + SmoothQuant + FP8 ViT)
Inside the container (`/work`); needs a GPU for the SmoothQuant calibration forward:
```bash
scripts/build_submission_model.sh
# writes /root/.cache/huggingface/Qwen3-VL-235B-A22B-Instruct-MXFP4-mlperf6.1-closed
# (the quantized checkpoint used in step 3; override the name with OUT_NAME=... — see the script)
```

## 3. Run the reference benchmark (Offline + Server), compliant sampling
Runs land under `/submission/outputs/` (host: `submission/outputs/`), the same tree the packager reads:
```bash
python3 scripts/benchmark_mlperf6pt1.py scenario=offline_qwen3_vl_235b_a22b_shopify hydra.run.dir=/submission/outputs/offline
python3 scripts/benchmark_mlperf6pt1.py scenario=server_qwen3_vl_235b_a22b_shopify hydra.run.dir=/submission/outputs/server
```

## 4. Package + validate

**On the host** (after exiting the container):
```bash
RUNS=submission/outputs
python3 submission/packager.py --offline "$RUNS/offline" --server "$RUNS/server" --sut MI355X_8x
#   -> submission/outputs/<YYYYmmdd_HHMMSS>_submission_out/   (exact path printed at the end)
# validate with the pinned checker (mlcommons/inference @ cd92cfe7); use python3 + the .main module,
# run from the checker's tools/submission/ dir:
git clone https://github.com/mlcommons/inference.git && git -C inference checkout cd92cfe7
OUT=$(realpath "$(ls -dt submission/outputs/*_submission_out | head -1)")   # newest packaged tree
( cd inference/tools/submission && python3 -m submission_checker.main \
    --input "$OUT" --version v6.1 --submitter AMD )
```

**In-container** (`submission/` is mounted at `/submission`; the image already ships the checker's
only dep, PyYAML — but the checker itself is NOT baked in, so clone it inside; host networking is on):
```bash
python3 /submission/packager.py --offline /submission/outputs/offline --server /submission/outputs/server --sut MI355X_8x
git clone https://github.com/mlcommons/inference.git /tmp/inference && git -C /tmp/inference checkout cd92cfe7
OUT=$(ls -dt /submission/outputs/*_submission_out | head -1)          # newest packaged tree
( cd /tmp/inference/tools/submission && python3 -m submission_checker.main \
    --input "$OUT" --version v6.1 --submitter AMD )
```

## Contents of this directory

- **Image recipe** — `docker/Dockerfile` and `scripts/start_docker.sh` 
- **Model recipe** — `scripts/{build_submission_model.sh, quantize_mxfp4_quark.py}`
- **Run harness** — under `scripts/`
- **Config** — `configs/benchmark.yaml` with the `server` and `orchestrator` settings inlined 

Not included (intentionally): the packager (`submission/packager.py`), systems/measurements/documentation
(sibling dirs in this submission), and all dev-only code (other images, profiling configs, docs, scratch).
