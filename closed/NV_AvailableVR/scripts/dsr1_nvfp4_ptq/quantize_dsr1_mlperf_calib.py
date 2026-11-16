#!/usr/bin/env python3
"""Create the DeepSeek-R1 FP8 -> NVFP4 checkpoint used for the MLPerf Inference
deepseek-r1 benchmark, calibrated with the MLPerf calibration dataset.

This is a thin, auditable driver around NVIDIA TensorRT Model Optimizer's
official DeepSeek recipe (`examples/deepseek`). The ONLY deviation from the
stock recipe is the calibration data: instead of cnn_dailymail + nemotron, the
calibration sweep reads the 500-sample MLPerf deepseek-r1 calibration set. That
override is the `build_mlperf_calib_dataloader` function below, injected into
Model-Optimizer's `deepseek_v3/ptq.py` by `apply_calib_override`.

Pins (reproduce exactly):
  - Model-Optimizer : commit 089c06e41 (examples/deepseek; producer tag 0.46.0.dev0+g089c06e41)
  - DeepSeek-V3     : commit 9b4e978   (inference/convert.py, config_671B.json)
  - Base model      : deepseek-ai/DeepSeek-R1 @ 56d4cbb (FP8 HF checkpoint)
  - Container       : nvcr.io/nvidia/pytorch:25.10-py3
  - Published result: https://huggingface.co/centml/DeepSeek-R1-NVFP4-v2-mlpinf

Stages 1-3 are heavy multi-GPU jobs (run under SLURM/pyxis or on an 8-GPU
Blackwell node); stage 4 is a fast CPU-only safetensors rewrite:
  1. reshard the HF FP8 checkpoint to model-parallel-8   (convert.py)
  2. calibration sweep with the MLPerf calib set -> amax (deepseek_v3/ptq.py)
  3. one-shot FP8->NVFP4 weight conversion               (quantize_fp8_to_nvfp4.sh)
  4. normalize the non-quantized weights' storage dtype FP32 -> BF16
     (see stage4_normalize_dtype for why this is required for wide-EP loading)

Usage:
  python quantize_dsr1_mlperf_calib.py \
      --modelopt   /path/to/Model-Optimizer \
      --deepseek-v3 /path/to/DeepSeek-V3 \
      --hf-fp8     /path/to/DeepSeek-R1            # FP8 HF checkpoint (~690G) \
      --calib      /path/to/data.parquet          # MLPerf calib, 500 rows, `text` col \
      --out        /path/to/output                # ~1.1T transient, fp4 ckpt in $out/fp4_out

  # inspect the recipe / commands without running:
  python quantize_dsr1_mlperf_calib.py --print-recipe
  # unit self-check for the calibration batching:
  python quantize_dsr1_mlperf_calib.py --selftest

See ./README.md for the full reproduction notes and cluster gotchas.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

MODELOPT_SHA = "089c06e41"
DEEPSEEK_V3_SHA = "9b4e978"
BASE_MODEL = "deepseek-ai/DeepSeek-R1@56d4cbb"
CONTAINER = "nvcr.io/nvidia/pytorch:25.10-py3"
CALIB_SIZE = 500
BATCH_SIZE = 4
MAX_LEN = 2048
WORLD_SIZE = 8

# Marker so apply_calib_override is idempotent and greppable.
_OVERRIDE_MARKER = "MLPERF_CALIB_PARQUET"


def build_mlperf_calib_dataloader(tokenizer, calib_parquet, device,
                                  calib_size=CALIB_SIZE, batch_size=BATCH_SIZE,
                                  max_len=MAX_LEN):
    """MLPerf calibration dataloader — the one recipe deviation.

    Reads a parquet with a single `text` column (the MLPerf deepseek-r1
    calibration set) and yields {"input_ids": tensor} batches shaped exactly
    as Model-Optimizer's calibrate_loop expects.

    The MLPerf calib prompts are already chat-templated (they contain the
    BOS / <|begin_of_sentence|> token), so they are tokenized with
    add_special_tokens=False. Adding another BOS here silently skews the
    per-tensor amax statistics and is the most common way to get this wrong.
    """
    import datasets as hf_datasets

    texts = hf_datasets.load_dataset(
        "parquet", data_files=calib_parquet, split="train")["text"]
    texts = list(texts)[:calib_size]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # length-sort to minimize intra-batch padding noise in the calibration stats
    texts.sort(key=len)

    batches = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,  # prompts already carry BOS / chat template
        )
        batches.append({"input_ids": enc["input_ids"].to(device)})
    return batches


# The block injected into Model-Optimizer's deepseek_v3/ptq.py, replacing the
# stock `get_dataset_dataloader(dataset_name=["cnn_dailymail", ...])` call.
_OVERRIDE_BLOCK = '''\
    ## MLPerf calibration override (injected by quantize_dsr1_mlperf_calib.py)
    import os as _os
    import datasets as _hf_datasets
    _calib_parquet = _os.environ["MLPERF_CALIB_PARQUET"]
    _max_len = int(_os.environ.get("MLPERF_CALIB_MAX_LEN", "2048"))
    _texts = _hf_datasets.load_dataset("parquet", data_files=_calib_parquet, split="train")["text"]
    _texts = list(_texts)[:calib_size]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _texts.sort(key=len)
    calib_dataset = []
    for _i in range(0, len(_texts), batch_size):
        _enc = tokenizer(_texts[_i:_i + batch_size], return_tensors="pt", padding=True,
                         truncation=True, max_length=_max_len, add_special_tokens=False)
        calib_dataset.append({"input_ids": _enc["input_ids"].to(device)})
'''


def apply_calib_override(modelopt_dir):
    """Patch Model-Optimizer's deepseek_v3/ptq.py to calibrate on the MLPerf set.

    Idempotent: replaces the stock get_dataset_dataloader(...) assignment to
    `calib_dataset` with the MLPerf override block. Returns the path patched.
    """
    ptq = os.path.join(modelopt_dir, "examples", "deepseek", "deepseek_v3", "ptq.py")
    if not os.path.isfile(ptq):
        raise FileNotFoundError(f"ptq.py not found at {ptq} (check --modelopt)")
    with open(ptq, encoding="utf-8") as f:
        src = f.read()
    if _OVERRIDE_MARKER in src:
        return ptq  # already patched

    key = "calib_dataset = get_dataset_dataloader("
    start = src.find(key)
    if start == -1:
        raise RuntimeError(
            "Could not find the stock `calib_dataset = get_dataset_dataloader(` "
            f"call in {ptq}; the recipe may have changed for a different "
            f"Model-Optimizer commit (pinned: {MODELOPT_SHA}).")
    # the stock call spans multiple lines ending in `)` at column 4
    end = src.find("\n    )", start)
    if end == -1:
        raise RuntimeError("Could not find the end of the get_dataset_dataloader(...) call.")
    end = end + len("\n    )")
    patched = src[:start] + _OVERRIDE_BLOCK.rstrip() + src[end:]
    with open(ptq, "w", encoding="utf-8") as f:
        f.write(patched)
    return ptq


def _run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def stage1_reshard(deepseek_v3, hf_fp8, out):
    """Reshard HF FP8 -> model-parallel-8 (CPU-only, holds ~660G in host RAM)."""
    _run([sys.executable, os.path.join(deepseek_v3, "inference", "convert.py"),
          "--hf-ckpt-path", hf_fp8,
          "--save-path", os.path.join(out, "ds_ckpt_mp8"),
          "--n-experts", "256", "--model-parallel", str(WORLD_SIZE)])


def stage2_calibrate(modelopt, deepseek_v3, calib_parquet, out, head, nnodes):
    """Calibration sweep over the MLPerf set -> per-tensor amax (WORLD_SIZE ranks)."""
    if WORLD_SIZE % nnodes != 0:
        raise ValueError(f"--nnodes ({nnodes}) must divide world size {WORLD_SIZE}")
    apply_calib_override(modelopt)
    env = dict(os.environ, MLPERF_CALIB_PARQUET=calib_parquet)
    cfg = os.path.join(deepseek_v3, "inference", "configs", "config_671B.json")
    # c10d rendezvous assigns node ranks; do NOT pass --node_rank
    cmd = ["torchrun", "--nnodes", str(nnodes),
           "--nproc-per-node", str(WORLD_SIZE // nnodes),
           "--rdzv_backend", "c10d",
           "--rdzv_id", os.environ.get("SLURM_JOB_ID", "0"),
           "--rdzv_endpoint", f"{head}:29513",
           "deepseek_v3/ptq.py",
           "--model_path", os.path.join(out, "ds_ckpt_mp8"),
           "--config", cfg,
           "--quant_cfg", "NVFP4_DEFAULT_CFG",
           "--output_path", os.path.join(out, "amax_out"),
           "--calib_size", str(CALIB_SIZE), "--batch_size", str(BATCH_SIZE)]
    print("+ (cwd=%s) " % os.path.join(modelopt, "examples", "deepseek")
          + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env,
                   cwd=os.path.join(modelopt, "examples", "deepseek"))


def stage3_convert(modelopt, hf_fp8, out):
    """One-shot FP8 -> NVFP4 weight conversion using the recorded amax (1 GPU)."""
    script = os.path.join(modelopt, "examples", "deepseek",
                          "deepseek_v3", "quantize_fp8_to_nvfp4.sh")
    _run(["bash", script,
          "--amax_path", os.path.join(out, "amax_out"),
          "--fp4_output_path", os.path.join(out, "fp4_out"),
          "--fp8_hf_path", hf_fp8,
          "--world_size", str(WORLD_SIZE)])
    if not os.path.isfile(os.path.join(out, "fp4_out", "hf_quant_config.json")):
        raise RuntimeError("no fp4 output produced (missing hf_quant_config.json)")
    print("NVFP4 checkpoint ready at " + os.path.join(out, "fp4_out"))


def _f32_to_bf16_rne(buf):
    """Round-to-nearest-even cast of a raw float32 byte buffer to bfloat16 bytes."""
    import numpy as np
    u = np.frombuffer(buf, dtype=np.uint32)
    # add the rounding bias (0x7FFF) plus the LSB of the retained mantissa, then
    # keep the high 16 bits -- this is IEEE round-to-nearest-even.
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return r.tobytes()


def stage4_normalize_dtype(out):
    """Cast the non-quantized passthrough weights from FP32 to BF16 (in place).

    Model-Optimizer's quantize_to_nvfp4 dequantizes the FP8 weights to BF16 but
    passes the *non-quantized* weights (the excluded MLA projections and the MTP
    layer) through at whatever dtype the source FP8 checkpoint stored them in --
    which for DeepSeek-R1 is FP32. The reference nvidia/DeepSeek-R1-FP4-v2 stores
    those same tensors as BF16, and TensorRT-LLM's wide-EP loader (dep16 /
    attention-DP, i.e. the Interactive scenario) trusts the encoded dtype: an
    FP32 [2048,7168] buffer is reinterpreted as BF16 with 14336 columns and the
    load aborts with
      "The size of tensor a (7168) must match the size of tensor b (14336)".
    The dep8 (Offline/Server) loader tolerates the mismatch, so this only bites
    wide-EP. Without this stage the produced checkpoint fails to load at dep16.

    Fix: cast every FP32 ``*.weight`` tensor to BF16 (round-to-nearest-even). The
    NVFP4 quantized weights are stored as packed uint8/FP8 (not FP32) and the
    quantization scales (``*_scale``, ``*_scale_2``, ``*.input_scale``) do not end
    in ``.weight`` -- so both are correctly left untouched, reproducing the
    reference checkpoint's dtype layout exactly (0 dtype diffs vs FP4-v2). Pure
    safetensors byte-rewrite; only shards containing such tensors are touched.
    """
    import json
    import struct
    from glob import glob

    ckpt = os.path.join(out, "fp4_out")
    shards = sorted(glob(os.path.join(ckpt, "*.safetensors")))
    if not shards:
        raise RuntimeError(f"no safetensors under {ckpt}; run stage 3 first")

    def _read_header(path):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            return json.loads(fh.read(n)), 8 + n

    rewritten = cast_total = 0
    for path in shards:
        hdr, data_start = _read_header(path)
        names = [k for k in hdr if k != "__metadata__"]
        cast = {k for k in names
                if hdr[k]["dtype"] == "F32" and k.endswith(".weight")}
        if not cast:
            continue
        order = sorted(names, key=lambda k: hdr[k]["data_offsets"][0])
        new_hdr = {}
        if "__metadata__" in hdr:
            new_hdr["__metadata__"] = hdr["__metadata__"]
        off = 0
        for k in order:
            v = hdr[k]
            size = v["data_offsets"][1] - v["data_offsets"][0]
            if k in cast:
                size //= 2  # F32 (4 bytes) -> BF16 (2 bytes)
                new_hdr[k] = {"dtype": "BF16", "shape": v["shape"],
                              "data_offsets": [off, off + size]}
            else:
                new_hdr[k] = {"dtype": v["dtype"], "shape": v["shape"],
                              "data_offsets": [off, off + size]}
            off += size
        hb = json.dumps(new_hdr, separators=(",", ":")).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)  # safetensors 8-byte header align
        tmp = path + ".tmp"
        with open(path, "rb") as fin, open(tmp, "wb") as fout:
            fout.write(struct.pack("<Q", len(hb)))
            fout.write(hb)
            for k in order:
                v = hdr[k]
                fin.seek(data_start + v["data_offsets"][0])
                raw = fin.read(v["data_offsets"][1] - v["data_offsets"][0])
                fout.write(_f32_to_bf16_rne(raw) if k in cast else raw)
        os.replace(tmp, path)
        rewritten += 1
        cast_total += len(cast)

    # the per-shard byte counts changed -> regenerate the index total_size
    idx_path = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            idx = json.load(f)
        total = 0
        for path in shards:
            hdr, _ = _read_header(path)
            total += sum(v["data_offsets"][1] - v["data_offsets"][0]
                         for k, v in hdr.items() if k != "__metadata__")
        idx.setdefault("metadata", {})["total_size"] = total
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(idx, f, indent=2)
    print(f"dtype normalize: cast {cast_total} FP32 .weight tensors -> BF16 "
          f"across {rewritten} shard(s)")


def print_recipe():
    print(__doc__)
    print(f"  Model-Optimizer  : {MODELOPT_SHA}")
    print(f"  DeepSeek-V3      : {DEEPSEEK_V3_SHA}")
    print(f"  base model       : {BASE_MODEL}")
    print(f"  container        : {CONTAINER}")
    print(f"  calib_size={CALIB_SIZE}  batch_size={BATCH_SIZE}  "
          f"max_len={MAX_LEN}  world_size={WORLD_SIZE}")


def selftest():
    """Verify the calibration batching: length-sorted, padded, no added BOS."""
    class FakeTok:
        pad_token = None
        eos_token = 0

        def __call__(self, texts, add_special_tokens=True, max_length=None, **kw):
            assert add_special_tokens is False, "must not add BOS to pre-templated prompts"
            ids = [[1] * min(len(t), max_length) for t in texts]
            width = max(len(x) for x in ids)
            padded = [x + [0] * (width - len(x)) for x in ids]

            class _T:
                def __init__(self, rows):
                    self.rows = rows

                def to(self, _dev):
                    return self.rows
            return {"input_ids": _T(padded)}

    import types
    from unittest import mock

    fake_ds = types.ModuleType("datasets")
    # split="train" -> real API returns a Dataset indexed by column name
    fake_ds.load_dataset = lambda *a, **k: {"text": ["xxxx", "x", "xxx", "xx"]}

    with mock.patch.dict(sys.modules, {"datasets": fake_ds}):
        batches = build_mlperf_calib_dataloader(
            FakeTok(), "unused", "cpu", calib_size=4, batch_size=2, max_len=8)
    assert len(batches) == 2, batches
    # first batch = the two shortest ("x","xx") -> padded to width 2
    assert batches[0]["input_ids"] == [[1, 0], [1, 1]], batches[0]
    assert batches[1]["input_ids"] == [[1, 1, 1, 0], [1, 1, 1, 1]], batches[1]

    # stage-4 cast: round-to-nearest-even FP32 -> BF16 (numpy-optional)
    try:
        import numpy as np
    except ImportError:
        print("selftest OK (numpy absent: skipped bf16-cast check)")
        return
    import struct as _struct
    vals = [1.0, -2.5, 3.5, 65504.0, 0.1]
    out16 = np.frombuffer(_f32_to_bf16_rne(_struct.pack("<%df" % len(vals), *vals)),
                          dtype=np.uint16)
    back = (out16.astype(np.uint32) << 16).view(np.float32)
    assert np.allclose(back, vals, rtol=2 ** -8), (back, vals)
    # tie: 1 + 2^-9 has the guard bit set with zero remainder -> rounds to even (1.0)
    tie = _struct.pack("<f", float(np.float32(1.0) + np.float32(2 ** -9)))
    assert np.frombuffer(_f32_to_bf16_rne(tie), dtype=np.uint16)[0] == 0x3F80
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelopt", help="NVIDIA/Model-Optimizer checkout @ %s" % MODELOPT_SHA)
    ap.add_argument("--deepseek-v3", help="deepseek-ai/DeepSeek-V3 checkout @ %s" % DEEPSEEK_V3_SHA)
    ap.add_argument("--hf-fp8", help="DeepSeek-R1 FP8 HF checkpoint dir")
    ap.add_argument("--calib", help="MLPerf calib parquet (500 rows, `text` column)")
    ap.add_argument("--out", help="output dir (~1.1T transient; fp4 ckpt lands in <out>/fp4_out)")
    ap.add_argument("--head", default=os.environ.get("HEAD", "localhost"),
                    help="rendezvous head host for multi-node stage 2")
    ap.add_argument("--nnodes", type=int, default=int(os.environ.get("SLURM_NNODES", "1")),
                    help="nodes for stage 2 (nnodes * gpus_per_node must be %d)" % WORLD_SIZE)
    ap.add_argument("--stage", choices=["1", "2", "3", "4", "all"], default="all")
    ap.add_argument("--print-recipe", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.print_recipe:
        print_recipe()
        return
    if args.selftest:
        selftest()
        return

    # stage 4 is a pure safetensors rewrite of an existing fp4_out -> only --out
    required = {
        "1": ("deepseek_v3", "hf_fp8", "out"),
        "2": ("modelopt", "deepseek_v3", "calib", "out"),
        "3": ("modelopt", "hf_fp8", "out"),
        "4": ("out",),
        "all": ("modelopt", "deepseek_v3", "hf_fp8", "calib", "out"),
    }[args.stage]
    for req in required:
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required for stage {args.stage} "
                     f"(use --print-recipe / --selftest to inspect without inputs)")
    os.makedirs(args.out, exist_ok=True)

    if args.stage in ("1", "all"):
        stage1_reshard(args.deepseek_v3, args.hf_fp8, args.out)
    if args.stage in ("2", "all"):
        stage2_calibrate(args.modelopt, args.deepseek_v3, args.calib, args.out,
                         args.head, args.nnodes)
    if args.stage in ("3", "all"):
        stage3_convert(args.modelopt, args.hf_fp8, args.out)
    if args.stage in ("4", "all"):
        stage4_normalize_dtype(args.out)


if __name__ == "__main__":
    main()
