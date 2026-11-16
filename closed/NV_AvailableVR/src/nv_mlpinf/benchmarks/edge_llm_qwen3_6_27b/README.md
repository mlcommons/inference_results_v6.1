# Edge-LLM Qwen3.6-27B (Edge-Agentic)

End-to-end edge benchmark for **Qwen3.6-27B** served by **TensorRT Edge-LLM**
([`release/0.9.1-mlpinf`](https://github.com/NVIDIA/TensorRT-Edge-LLM/tree/release/0.9.1-mlpinf))
on Jetson-class devices. Unlike the datacenter benchmarks in this directory, the
workload is driven by the **MLCommons Inference Endpoints harness**
([mlcommons/endpoints](https://github.com/mlcommons/endpoints)) against an
OpenAI-compatible HTTP server rather than by the nv_mlpinf LoadGen core; this
README outlines the full validated flow: **build → quantize → export → build
engines → serve → benchmark → verify**.

## Support Matrix

| System | Scenario | Precision | Status |
| --- | --- | --- | --- |
| Jetson AGX Thor (sm_110, JetPack 7.2, CUDA 13.2, TensorRT 10.16) | SingleStream (concurrency 1) | NVFP4 + NVFP4 lm_head + FP8 KV cache, tree-MTP | Validated end-to-end on device |

## Benchmark Definition

Two phases, both deterministic (`temperature 0`, `seed 42`), reasoning **off**,
single-stream:

| Phase | Dataset | Samples | Metric |
| --- | --- | --- | --- |
| Performance | `agentic_coding_2.5h.jsonl` — 20 recorded SWE-bench-style agentic-coding trajectories, teacher-forced multi-turn replay | 1,007 generated turns | Wall-clock / TPS + **inline IoU** (multiset IoU of normalized bash executables per turn vs. the recorded ground truth); a run is valid only with 0 dropped turns |
| Accuracy | BFCL v4 single-turn function calling (categories `non_live`/`live`/`hallucination`) | ~995 | Overall AST-match accuracy |

**Dataset Statistics** (tokens, measured with the served model's tokenizer):

| Dataset | Samples | ISL mean | ISL p90 | ISL max | OSL mean | OSL p90 | OSL max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Performance (agentic replay) | 1,007 | 10,554 | 17,681 | 23,456 | 100 | 225 | 1,022 |
| Accuracy (BFCL v4) | 995 | 519 | 746 | 3,059 | 81 | 158 | 1,024 |

The performance set is strongly prefill-dominated (ISL:OSL ≈ 105:1, accumulated
multi-turn context); it is built so peak ISL stays under a 32K served context.
Outputs are capped at `max_new_tokens: 1024`.

## Getting Started

### Model

Preferred starting checkpoint (already NVFP4 + MTP):
[`centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf`](https://huggingface.co/centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf).
Alternatively, start from unquantized
[`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) (carries MTP draft
weights) and quantize on device (below).

### Datasets

Both ship with / are fetched by the endpoints harness:

- Performance: `examples/11_Edge_Agentic_Example/agentic_coding_2.5h.jsonl` (in-repo, 2,014 rows / 20 conversations).
- Accuracy: BFCL v4 self-downloads on first run into `dataset_cache/bfcl_v4/bfcl_v4_single_turn.parquet`
  (also the source for the quantization calibration subset).

## Build TensorRT Edge-LLM

On the device, from
[`NVIDIA/TensorRT-Edge-LLM@release/0.9.1-mlpinf`](https://github.com/NVIDIA/TensorRT-Edge-LLM/tree/release/0.9.1-mlpinf):

```bash
export REPO=<tensorrt-edge-llm checkout>   # release/0.9.1-mlpinf
export VENV=<python3.12 venv>              # pip install "$REPO[tools]"
export WORK=$REPO/mlperf/artifacts
export B=$REPO/build
# Preferred: https://huggingface.co/centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
# Or unquantized Qwen/Qwen3.6-27B (quantize on device below).
export BASE_MODEL=<path-to>/Qwen3.6-27B-NVFP4-W4A4-mlpinf
sudo nvpmodel -m 0 && sudo jetson_clocks   # lock clocks for stable timing

cd "$REPO" && git submodule update --init --recursive
# CuTe DSL kernels (sm_110), C++ runtime + plugin, python bindings:
$VENV/bin/python kernelSrcs/build_cutedsl.py --kernels ALL --gpu_arch sm_110 --arch aarch64 -j 8
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-thor -DCUDA_CTK_VERSION=13.2 -DENABLE_CUTE_DSL=ALL && make -j"$(nproc)"
# then experimental/pybind (python bindings) — see $REPO/mlperf/README.md §1c for the exact cmake line
```

## Quantize, Export, Build Engines

```bash
# If starting from centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf, skip calib/quantize and set
# QUANT="$BASE_MODEL". Otherwise quantize on device from Qwen/Qwen3.6-27B:

# Calibration: 10% of the BFCL accuracy data, chat-templated {messages, tools} JSONL
$VENV/bin/python mlperf/make_calibration.py "$WORK/bfcl_calib.jsonl" <bfcl_v4_single_turn.parquet>

# NVFP4 backbone + NVFP4 lm_head + FP8 KV; MTP draft auto-detected and quantized
$VENV/bin/python -m tensorrt_edgellm.scripts.quantize llm \
  --model_dir "$BASE_MODEL" --output_dir "$WORK/quant-nvfp4" \
  --quantization nvfp4 --lm_head_quantization nvfp4 --kv_cache_quantization fp8 \
  --dataset "$WORK/bfcl_calib.jsonl" --num_samples 364
QUANT="$WORK/quant-nvfp4"

# Export ONNX with TREE-MTP bindings (--mtp-tree-base; plain --mtp is chain-only)
$VENV/bin/python -m tensorrt_edgellm.scripts.export "$QUANT" "$WORK/onnx" --mtp-tree-base --skip-visual

# Build base + draft engines (draft --maxInputLen MUST equal base's), combine into one dir.
# Prefill max=8192 with opt=1024 (classic 2-profile). Keep maxKVCacheCapacity=32768 so
# multi-turn agentic context reuse does not hit "Insufficient KV cache capacity".
$B/examples/llm/llm_build --onnxDir "$WORK/onnx/llm"       --engineDir "$WORK/engines/base" \
  --specBase  --maxInputLen 8192 --optInputLen 1024 --maxKVCacheCapacity 32768 --maxBatchSize 1 \
  --maxKVPoolPages 1024 --maxVerifyTreeSize 32
$B/examples/llm/llm_build --onnxDir "$WORK/onnx/mtp_draft" --engineDir "$WORK/engines/draft" \
  --specDraft --maxInputLen 8192 --optInputLen 1024 --maxKVCacheCapacity 32768 --maxBatchSize 1 \
  --maxKVPoolPages 512 --maxDraftTreeSize 32
cp "$WORK/engines/draft/"{spec_draft.engine,draft_config.json} "$WORK/engines/base/"
cp mlperf/chat_template_noreason.jinja "$WORK/engines/base/chat_template.jinja"   # reasoning-off template required
```

## Serve

```bash
bash mlperf/serve_edgellm.sh            # OpenAI-compatible server on :8001
curl -s http://localhost:8001/v1/models # -> {"data":[{"id":"base",...}]}
```

Serves the combined engine dir with **tree-MTP (top-k 8 / step 6 / verify 32)** —
validated fastest at full scale in a 13-point sweep of `top-k × steps × verify` —
and prefill-state-only KV context reuse (MLPerf replay semantics: generated-token
KV is never reused).

## Run the Benchmark

```bash
git clone https://github.com/mlcommons/endpoints && cd endpoints
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,bfcl]"

# Use this directory's config.yaml (also shipped as mlperf/config.yaml on
# release/0.9.1-mlpinf). Edit model_params.tokenizer_name to your combined
# engine dir ($WORK/engines/base).
CFG=<path-to>/closed/NVIDIA/src/nv_mlpinf/benchmarks/edge_llm_qwen3_6_27b/config.yaml
inference-endpoint benchmark from-config --config "$CFG"                  # both phases
inference-endpoint benchmark from-config --config "$CFG" --mode perf      # performance only
inference-endpoint benchmark from-config --config "$CFG" --mode acc       # accuracy only
# (--accuracy-only is an equivalent alias for --mode acc)
```

Outputs under `report_dir`: `scores.json` (inline IoU + turn validity),
`accuracy/accuracy_results.json` (BFCL score), `report.txt` (Duration / TPS /
TTFT breakdown).

## Known Gotchas

- Tree MTP (`--draft-top-k > 1`) requires `--mtp-tree-base` at ONNX export; plain
  `--mtp` is chain-only (`--draft-top-k 1`, `--verify-tree-size = draft-step + 1`).
- Draft engine `--maxInputLen` must equal the base engine's, or long-context
  prefill fails `satisfyProfile`.
- Keep `maxKVCacheCapacity=32768` (not 16384): agentic multi-turn reuse otherwise
  hits `Insufficient KV cache capacity` / page-table row capacity errors and IoU
  collapses.
- Both engines live in **one** `--engine-dir`.
- MTP + context reuse at concurrency 1 needs `--recurrent-capture-interval 0` and
  a one-slot (~160 MB) recurrent snapshot pool.
- The engine dir must contain a reasoning-off `chat_template.jinja`; the harness
  runs reasoning off, and `<think>` spans break tool-call extraction.

Full command-level detail (every step validated on device):
[`mlperf/README.md` on `release/0.9.1-mlpinf`](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/release/0.9.1-mlpinf/mlperf/README.md).
