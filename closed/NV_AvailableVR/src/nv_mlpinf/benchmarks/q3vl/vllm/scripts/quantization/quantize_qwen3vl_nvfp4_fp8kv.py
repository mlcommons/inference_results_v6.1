#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantize Qwen3-VL-235B-A22B-Instruct to NVFP4 (W4A4) + calibrated per-tensor FP8 KV cache.

Reproduces the MLPerf q3vl submission checkpoint:

    nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.1-FP8-KV  (served from `main`)

Recipe: NVFP4 on every linear layer (static MSE weight scales plus dynamic NVFP4 inputs) with a
calibrated per-tensor FP8 KV cache, calibrated on the Shopify catalogue (the benchmark's own data).
Compared with the previous llm-compressor NVFP4-only checkpoint, this adds the FP8 KV cache and
recovers accuracy with MSE weight calibration (see README.md for the accuracy/throughput table).

The script wraps NVIDIA TensorRT-Model-Optimizer's ``examples/llm_ptq/hf_ptq.py`` (the standard
modelopt PTQ entry point) and adds four small Qwen3-VL-MoE workarounds. Each is a known modelopt gap
we intend to upstream (see README.md); until then they live here so the checkpoint is reproducible
from a stock modelopt checkout:

  1. route_moe_experts_to_fused_path(): modelopt's model-specific expert class is not HF-exportable,
     so reroute to the generic, exportable fused-experts class.
  2. CUDA_LAUNCH_BLOCKING=1 (set below, before CUDA init): the 235B multimodal calibration trips an
     async CUDA race that surfaces (misleadingly) at export; serializing kernels avoids it.
  3. serve_shopify_calibration(): hf_ptq's image-calibration path is hardcoded to a different dataset,
     so we build the Shopify calibration batches and inject them.
  4. postprocess_for_vllm(): the export leaves the MoE router gate in BF16 but omits it from the
     exclusion list (so vLLM mis-allocates it) and drops the VLM processor configs.

Usage (in a dedicated modelopt environment; see README.md):
    python quantize_qwen3vl_nvfp4_fp8kv.py --output <export_dir> --modelopt-repo <path/to/modelopt>
"""

from __future__ import annotations

import os

# Shim #2: serialize CUDA kernel launches to avoid the multimodal-calibration race. Must be set
# before any CUDA context is created (i.e. before torch initializes CUDA at model load). Uses
# setdefault so it can be disabled from the environment (CUDA_LAUNCH_BLOCKING=0) once a modelopt
# release fixes the underlying race.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import argparse
import base64
import json
import re
import runpy
import shutil
import sys
from io import BytesIO
from pathlib import Path

import torch

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"
RECIPE_NAME = "nvfp4_default_mse-kv_fp8"  # the .yaml ships next to this script
RECIPE_FILE = Path(__file__).resolve().parent / f"{RECIPE_NAME}.yaml"

# Shopify calibration set: the exact sample indices used for the submission checkpoint. The model
# routes every token to top-k experts, so these 20 multi-thousand-token samples cover all experts.
SHOPIFY_DATASET = "Shopify/the-catalogue-public-beta"
SHOPIFY_SPLITS = ["train", "test"]
CALIBRATION_SAMPLE_INDICES = [
    20232, 21162, 33584, 46825, 45190, 46143, 14189, 16658, 26406, 9565,
    33733, 31057, 47465, 33503, 42293, 7768, 1962, 39746, 13568, 22527,
]


# --------------------------------------------------------------------------------------------------
# Shim #1: route Qwen3-VL-MoE experts onto modelopt's generic, exportable fused-experts class.
# --------------------------------------------------------------------------------------------------
def route_moe_experts_to_fused_path() -> None:
    """Register Qwen3-VL-MoE experts to modelopt's generic ``_QuantFusedExperts``.

    modelopt ships a model-specific ``_QuantQwen3VLMoeTextExperts`` that splits experts into
    per-expert ``nn.Linear`` layers, which HF export does not support
    (``NotImplementedError: ... not supported in export``). The generic ``_QuantFusedExperts``
    targets the standard fused HF layout, quantizes by intercepting ``F.linear`` (keeping the
    model's own forward), and *is* exportable. ``register_fused_experts_on_the_fly`` skips
    already-registered types, so we unregister the specific class first.
    """
    from modelopt.torch.quantization.nn import QuantModuleRegistry
    from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

    if QuantModuleRegistry.get(Qwen3VLMoeTextExperts) is not None:
        QuantModuleRegistry.unregister(Qwen3VLMoeTextExperts)
    QuantModuleRegistry.register({Qwen3VLMoeTextExperts: "hf.Qwen3VLMoeTextExperts"})(_QuantFusedExperts)
    print("[quantize] routed Qwen3VLMoeTextExperts -> _QuantFusedExperts (exportable fused path)")


# --------------------------------------------------------------------------------------------------
# Shim #3: build + serve the Shopify calibration batches (hf_ptq's image path is hardcoded).
# --------------------------------------------------------------------------------------------------
def _calibration_messages(sample: dict, schema_json: str) -> list[dict]:
    """Format one Shopify product as the system+user chat used for calibration (image + text).

    The wording and line breaks are byte-identical to the original llm-compressor calibration script
    (including its ``specifc`` / ``followng`` typos): the submission checkpoint was calibrated on exactly
    these prompts, so reproducing it requires identical calibration inputs.
    """
    image = BytesIO()
    image_format = sample["product_image"].format
    sample["product_image"].save(image, format=image_format)
    image_b64 = base64.b64encode(image.getvalue()).decode("utf-8")
    return [
        {"role": "system", "content": [{"type": "text", "text": f"""Please analyze the product from the user prompt
and provide the following fields in a valid JSON object:
- category
- brand
- is_secondhand

You must choose only one, which is the most appropriate, correct, and specifc
category out of the list of possible product categories.

The description of the product sometimes contains various types of source code
(e.g., JavaScript, CSS, HTML, etc.), where useful product information is embedded
somewhere inside the source code. For this task, you should extract the useful
product information from the source code and leverage it, and discard the
programmatic parts of the source code.

Your response should only contain a valid JSON object and nothing more, e.g.,
you should not fence the JSON object inside a ```json code block.
The JSON object should match the followng JSON schema:
```json
{schema_json}
```
"""}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"""The title of the product is the following:
```text
{sample['product_title']}
```

The description of the product is the following:
```text
{sample['product_description']}
```

The following are the possible product categories:
```json
{sample['potential_product_categories']}
```
"""},
            {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_b64}"}},
        ]},
    ]


def build_shopify_calibration(model_id: str, max_seq_len: int, num_samples: int) -> list[dict]:
    """Load the fixed Shopify samples and tokenize them with the model's processor.

    Returns a list of batch dicts (one per sample, batch size 1) on CPU; ``serve_shopify_calibration``
    moves them to the calibration device.
    """
    from datasets import load_dataset
    from pydantic import BaseModel
    from transformers import AutoProcessor

    class ProductMetadata(BaseModel):
        category: str
        brands: list[str]
        is_secondhand: bool

    schema_json = json.dumps(ProductMetadata.model_json_schema(), indent=2)
    indices = CALIBRATION_SAMPLE_INDICES[:num_samples]
    dataset = load_dataset(SHOPIFY_DATASET, split="+".join(SHOPIFY_SPLITS), revision="main").select(indices)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    batches: list[dict] = []
    for sample in dataset:
        encoded = processor.apply_chat_template(
            _calibration_messages(sample, schema_json),
            return_tensors="pt", padding=False, truncation=True, max_length=max_seq_len,
            tokenize=True, add_special_tokens=False, return_dict=True, add_generation_prompt=False,
        )
        batch = {k: (v if torch.is_tensor(v) else torch.as_tensor(v)) for k, v in encoded.items()}
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        batches.append({k: v.cpu() for k, v in batch.items()})
    print(f"[quantize] built {len(batches)} Shopify calibration batches (indices {indices[0]}..{indices[-1]})")
    return batches


def serve_shopify_calibration(batches: list[dict]) -> None:
    """Monkeypatch modelopt's VLM calibration dataloader to serve our pre-built Shopify batches.

    hf_ptq calls ``get_vlm_dataset_dataloader(...)`` under ``--calib_with_images`` with a hardcoded
    (non-Shopify) dataset. We patch the module attribute *before* hf_ptq runs; hf_ptq's
    ``from ... import get_vlm_dataset_dataloader`` then binds to this replacement at its import time,
    so every call returns our batches (moved to the requested device). modelopt's forward loop simply
    iterates the returned sequence and calls ``model(**batch)``.
    """
    import modelopt.torch.utils.vlm_dataset_utils as vlm_utils

    def _shopify_dataloader(dataset_name=None, processor=None, batch_size=1, num_samples=None,
                            device=None, max_length=None, **_unused):
        dev = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        served = batches[: int(num_samples)] if num_samples else batches
        print(f"[quantize] serving {len(served)} Shopify calibration batches on {dev}")
        return [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()} for b in served]

    vlm_utils.get_vlm_dataset_dataloader = _shopify_dataloader


# --------------------------------------------------------------------------------------------------
# Shim #4: make the export loadable in vLLM (router-gate exclusion + VLM processor configs).
# --------------------------------------------------------------------------------------------------
def _gate_exclusion_entries(export_dir: Path) -> list[str]:
    """Wildcard + per-layer ``*.mlp.gate`` entries to exclude (derived from the export's weight map)."""
    index = export_dir / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"export looks incomplete: {index} not found (did PTQ/export succeed?)")
    weight_map = json.loads(index.read_text())["weight_map"]
    layers = sorted({int(m.group(1)) for k in weight_map
                     if (m := re.search(r"language_model\.layers\.(\d+)\.mlp\.gate", k))})
    return ["*mlp.gate"] + [f"model.language_model.layers.{i}.mlp.gate" for i in layers]


def postprocess_for_vllm(export_dir: Path, model_id: str) -> None:
    """Two metadata-only fixes the modelopt export needs before vLLM can load it.

    1. Router gate: the export leaves ``*.mlp.gate`` in BF16 but does not list it as excluded, so
       vLLM's modelopt_fp4 loader mis-allocates a packed buffer for it. Add the gate (wildcard +
       per-layer) to the exclusion lists in both config files (the loader reads either).
    2. Processor configs: the export saves only the tokenizer + chat template; copy the VLM
       image/video preprocessor configs from the source model so ``AutoProcessor`` works.
    """
    entries = _gate_exclusion_entries(export_dir)

    def add_entries(cfg_path: Path, container_keys: list[str], list_key: str) -> None:
        if not cfg_path.is_file():
            return
        cfg = json.loads(cfg_path.read_text())
        node = cfg
        for key in container_keys:
            if not isinstance(node, dict):
                return
            node = node.get(key, {})
        current = node.get(list_key)
        if current is None:  # only patch lists the export already declares
            return
        merged = current + [e for e in entries if e not in current]
        if merged != current:
            node[list_key] = merged
            cfg_path.write_text(json.dumps(cfg, indent=4))
            print(f"[quantize] {cfg_path.name}[{list_key}]: +{len(merged) - len(current)} gate entries")

    add_entries(export_dir / "hf_quant_config.json", ["quantization"], "exclude_modules")
    add_entries(export_dir / "config.json", ["quantization_config"], "exclude_modules")
    add_entries(export_dir / "config.json", ["quantization_config"], "ignore")

    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(model_id, allow_patterns=["*preprocessor_config.json"]))
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        if (src / name).is_file():
            shutil.copyfile(src / name, export_dir / name)
            print(f"[quantize] copied {name}")


def install_recipe() -> None:
    """Install the recipe into ``modelopt_recipes`` so ``--recipe general/ptq/<name>`` resolves it."""
    import modelopt_recipes
    dst_dir = Path(modelopt_recipes.__file__).parent / "general" / "ptq"
    if not dst_dir.is_dir():
        raise FileNotFoundError(f"modelopt_recipes ptq dir not found at {dst_dir} (is modelopt installed?)")
    shutil.copyfile(RECIPE_FILE, dst_dir / RECIPE_FILE.name)
    print(f"[quantize] installed recipe -> {dst_dir / RECIPE_FILE.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Export directory for the quantized checkpoint.")
    parser.add_argument("--modelopt-repo", default=os.environ.get("MODELOPT_REPO"),
                        help="Path to the TensorRT-Model-Optimizer checkout (for examples/llm_ptq/hf_ptq.py).")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Source model id or path.")
    parser.add_argument("--calib-size", type=int, default=len(CALIBRATION_SAMPLE_INDICES),
                        help="Number of Shopify calibration samples (default: all fixed indices).")
    parser.add_argument("--calib-seq", type=int, default=65536, help="Max calibration sequence length.")
    args = parser.parse_args()

    if not args.modelopt_repo:
        parser.error("--modelopt-repo (or $MODELOPT_REPO) is required: path to the modelopt checkout.")
    hf_ptq = Path(args.modelopt_repo).expanduser().resolve() / "examples" / "llm_ptq" / "hf_ptq.py"
    if not hf_ptq.is_file():
        parser.error(f"hf_ptq.py not found at {hf_ptq}; check the --modelopt-repo path.")
    if not 1 <= args.calib_size <= len(CALIBRATION_SAMPLE_INDICES):
        parser.error(f"--calib-size must be in 1..{len(CALIBRATION_SAMPLE_INDICES)} (the fixed Shopify indices).")

    route_moe_experts_to_fused_path()
    install_recipe()
    serve_shopify_calibration(build_shopify_calibration(args.model, args.calib_seq, args.calib_size))

    # Hand off to modelopt's standard PTQ entry point (it loads the model, calibrates, and exports).
    sys.argv = [
        str(hf_ptq),
        "--pyt_ckpt_path", args.model,
        "--recipe", f"general/ptq/{RECIPE_NAME}",
        "--calib_with_images", "--calib_size", str(args.calib_size),
        "--batch_size", "1", "--use_seq_device_map", "--skip_generate",
        "--export_path", args.output,
    ]
    sys.path.insert(0, str(hf_ptq.parent))  # so hf_ptq can import its sibling example_utils
    print(f"[quantize] running modelopt PTQ -> {args.output}")
    try:
        runpy.run_path(str(hf_ptq), run_name="__main__")
    except SystemExit as exit_:
        if exit_.code:  # hf_ptq exits 0 on success; only propagate real failures
            raise

    postprocess_for_vllm(Path(args.output), args.model)
    print(f"[quantize] done. Quantized checkpoint at {args.output}")


if __name__ == "__main__":
    main()
