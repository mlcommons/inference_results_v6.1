# DeepSeek-R1 FP8 → NVFP4 checkpoint (MLPerf calibration)

Reproduction recipe for an NVFP4 (FP4) DeepSeek-R1 checkpoint quantized from the
base FP8 model with NVIDIA TensorRT Model Optimizer, calibrated on the **MLPerf
Inference deepseek-r1 calibration dataset** (500 samples) instead of the stock
Model-Optimizer calibration data.

Published checkpoint:
**https://huggingface.co/centml/DeepSeek-R1-NVFP4-v2-mlpinf**

The deepseek-r1 benchmark uses this MLPerf-calibrated checkpoint. It reaches the
closed-division accuracy gate (exact_match 80.86, gate ≥ 80.5446).

## Pins

| What | Version |
|---|---|
| Base model | `deepseek-ai/DeepSeek-R1` @ `56d4cbb` (FP8 HF checkpoint) |
| Model-Optimizer | `089c06e41` (`examples/deepseek`; producer tag `0.46.0.dev0+g089c06e41`) |
| DeepSeek-V3 | `9b4e978` (`inference/convert.py`, `config_671B.json`) |
| Container | `nvcr.io/nvidia/pytorch:25.10-py3` |
| Calibration data | `mlperf_deepseek_r1_calibration_dataset_500_fp8_eval` (500 rows, `text` column) |

## Recipe

Upstream: https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/deepseek

The checkpoint is produced by the official three-stage DeepSeek recipe. The
**only** deviation is the calibration data: the calibration sweep reads the
MLPerf deepseek-r1 calibration set rather than cnn_dailymail + nemotron.

1. **Reshard** the HF FP8 checkpoint to model-parallel-8 — `DeepSeek-V3/inference/convert.py`
2. **Calibrate** — `deepseek_v3/ptq.py --quant_cfg NVFP4_DEFAULT_CFG --calib_size 500`:
   one forward sweep over the 500 MLPerf prompts recording per-tensor activation
   `amax`, plus a peer-max `amax` sync across MoE experts. No training / iterations.
3. **Convert** — `deepseek_v3/quantize_fp8_to_nvfp4.sh --world_size 8`: one-shot
   FP8 → NVFP4 weight rewrite using the recorded scales, emitting a deployable
   HF-format checkpoint.

Quant scheme (recipe defaults): dense MLP + MoE expert linears and `attn.wo`
in NVFP4 (fine-grained block scaling, group size 16); MLA attention projections
in bf16; KV cache in FP8; `lm_head`, MoE router gates, and the MTP layer excluded.

### The one deviation — MLPerf calibration data

`quantize_dsr1_mlperf_calib.py` encapsulates the whole recipe and injects the
MLPerf calibration dataloader into Model-Optimizer's `deepseek_v3/ptq.py`
(`apply_calib_override`). Key detail: the MLPerf calibration prompts are already
chat-templated (they contain the BOS / `<|begin_of_sentence|>` token), so they
are tokenized with **`add_special_tokens=False`** — adding a second BOS silently
skews the per-tensor `amax` statistics. Prompts are length-sorted (less padding),
truncated at 2048 tokens, batch size 4.

## Usage

```bash
# inspect the recipe / pins without any inputs
python quantize_dsr1_mlperf_calib.py --print-recipe

# unit self-check for the calibration batching
python quantize_dsr1_mlperf_calib.py --selftest

# run the pipeline (heavy: 8 GPUs of Blackwell generation, or 2× GB200 nodes;
# ~1.1 TB transient disk. Run inside the container above, under SLURM/pyxis or
# on a single 8-GPU node.)
python quantize_dsr1_mlperf_calib.py \
    --modelopt    /path/to/Model-Optimizer   `# checkout @ 089c06e41` \
    --deepseek-v3 /path/to/DeepSeek-V3        `# checkout @ 9b4e978` \
    --hf-fp8      /path/to/DeepSeek-R1        `# FP8 HF checkpoint, ~690 GB` \
    --calib       /path/to/data.parquet       `# MLPerf calib, 500 rows` \
    --out         /path/to/output             `# fp4 checkpoint lands in <out>/fp4_out`
```

Multi-node stage 2 uses `torchrun` with `--rdzv_backend c10d` (no `--node_rank`);
pass `--nnodes` and set `HEAD`/`--head` to the rendezvous host. `--stage 1|2|3|4`
runs a single stage (e.g. to chain them as separate SLURM jobs with
`--dependency=afterok`).

### Stage 4 — weight dtype normalization (required for wide-EP)

Model-Optimizer's `quantize_to_nvfp4` dequantizes the FP8 weights to BF16 but
passes the *non-quantized* weights (the excluded MLA projections and the MTP
layer) through at the source FP8 checkpoint's dtype — which for DeepSeek-R1 is
**FP32**, whereas the reference `nvidia/DeepSeek-R1-FP4-v2` stores them as BF16.
TensorRT-LLM's wide-EP loader (dep16 / attention-DP, i.e. the **Interactive**
scenario) trusts the encoded dtype: an FP32 `[2048,7168]` buffer is read as BF16
with 14336 columns and the load aborts with `The size of tensor a (7168) must
match the size of tensor b (14336)`. The dep8 (Offline/Server) loader tolerates
it, so this only surfaces on wide-EP.

Stage 4 casts every FP32 `*.weight` tensor to BF16 (round-to-nearest-even) — a
pure safetensors byte-rewrite. The NVFP4 quantized weights are packed uint8/FP8
(not FP32) and the quantization scales (`*_scale`, `*_scale_2`, `*.input_scale`)
do not end in `.weight`, so both are left untouched, reproducing the reference
checkpoint's dtype layout exactly. `--stage all` runs it automatically after
stage 3; `--stage 4 --out <dir>` applies it to an already-produced `fp4_out`.
The published `centml/DeepSeek-R1-NVFP4-v2-mlpinf` checkpoint already carries this
correction. The true source-side fix belongs upstream in Model-Optimizer
(`examples/deepseek/deepseek_v3/quantize_to_nvfp4.py`, which still emits the
passthrough weights unchanged) — stage 4 is the checkpoint-level equivalent.

## Validate

1. `<out>/amax_out/.quant_summary.txt` — NVFP4 on the expected MLP/MoE/`attn.wo`
   linears; MLA projections and `kv_bmm`/`pe_bmm` disabled; head, gates, MTP
   excluded; KV cache FP8.
2. Diff `<out>/fp4_out/hf_quant_config.json` + `config.json` against the
   published `centml/DeepSeek-R1-NVFP4-v2-mlpinf` checkpoint — only
   producer-version metadata should differ.
3. Serve-load with the existing deepseek-r1 configs (model path swapped), then
   run the MLPerf deepseek-r1 Offline AccuracyOnly flow.
4. Confirm the stage-4 dtype fix: no `*.weight` tensor is left as FP32 (only the
   quantization scales are). This is what lets the checkpoint load on the wide-EP
   (dep16) Interactive path in addition to dep8 Offline/Server.
