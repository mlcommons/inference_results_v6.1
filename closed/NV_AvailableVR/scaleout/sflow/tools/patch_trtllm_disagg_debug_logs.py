#!/usr/bin/env python3
"""Patch TensorRT-LLM disagg debug logs removed by NVIDIA/TensorRT-LLM@7212c43."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = "tensorrt_llm.serve.openai_disagg_server"
REMOVED_LINES = (
    '        logger.debug(f"Received context response from {ctx_server} for request {response.choices[0].disaggregated_params.ctx_request_id}")\n',
    '        logger.debug(f"Received first token from {gen_server} for request {request.disaggregated_params.ctx_request_id}")\n',
)


def main() -> None:
    spec = importlib.util.find_spec(MODULE)
    if spec is None or spec.origin is None:
        raise SystemExit(f"Could not locate {MODULE}")

    path = Path(spec.origin)
    text = path.read_text()
    patched = text
    for line in REMOVED_LINES:
        patched = patched.replace(line, "")

    if patched == text:
        print(f"Patch not needed: {path}")
        return

    path.write_text(patched)
    print(f"Patch successful: {path}")


if __name__ == "__main__":
    main()
