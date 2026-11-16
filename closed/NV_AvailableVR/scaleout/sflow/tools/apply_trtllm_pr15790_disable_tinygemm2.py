#!/usr/bin/env python3
"""Apply NVIDIA/TensorRT-LLM PR #15790 to an installed TensorRT-LLM package.

The PR adds a TLLM_DISABLE_TINYGEMM2 gate for the gpt-oss tinygemm2
small-M router GEMM. SFlow runs this inside release containers where
TensorRT-LLM is installed as a package, not as a git checkout, so this script
performs the equivalent idempotent source edits in-place.
"""

from __future__ import annotations

import fcntl
import site
import sysconfig
from pathlib import Path


# Mirrors Bofeng's TensorRT-LLM PR context:
# https://github.com/NVIDIA/TensorRT-LLM/pull/15790#issuecomment-4848789511
LOCK = Path("/tmp/trtllm_pr15790_disable_tinygemm2.lock")


def package_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            roots.append(Path(value))
    try:
        roots.extend(Path(path) for path in site.getsitepackages())
    except AttributeError:
        pass
    try:
        roots.append(Path(site.getusersitepackages()))
    except AttributeError:
        pass

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            unique.append(root)
            seen.add(root)
    return unique


def package_path(*parts: str) -> Path:
    relpath = Path("tensorrt_llm", *parts)
    checked = []
    for root in package_roots():
        candidate = root / relpath
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Could not locate installed TensorRT-LLM file "
        f"{relpath}; checked: {', '.join(checked)}"
    )


def replace_once(text: str, old: str, new: str, path: Path) -> tuple[str, bool]:
    if old not in text:
        raise SystemExit(f"Could not find expected text in {path}")
    return text.replace(old, new, 1), True


def write_if_changed(path: Path, original: str, patched: str) -> bool:
    if patched == original:
        return False
    path.write_text(patched)
    return True


def patch_trtllm_moe(path: Path) -> bool:
    text = path.read_text()
    if "is_tinygemm2_disabled" in text:
        return False

    # TensorRT-LLM 1.3.0rc14 does not contain this PR's auto_deploy router
    # tinygemm2 dispatch site. In that case, the GPT-OSS model patch below is
    # still required, but this hunk is legitimately not applicable.
    if not any(
        needle in text
        for needle in ("tinygemm2", "_TINYGEMM_SM", "_router_use_tinygemm")
    ):
        print(f"Skipping {path}: tinygemm2 router dispatch not present")
        return False

    patched = text
    patched, _ = replace_once(
        patched,
        "from tensorrt_llm._utils import get_sm_version\n",
        "from tensorrt_llm._utils import get_sm_version, is_tinygemm2_disabled\n",
        path,
    )
    patched, _ = replace_once(
        patched,
        "    return (\n        bias is not None\n        and get_sm_version() in _TINYGEMM_SM\n",
        "    return (\n        not is_tinygemm2_disabled()  # set TLLM_DISABLE_TINYGEMM2=1 to fall back to F.linear\n        and bias is not None\n        and get_sm_version() in _TINYGEMM_SM\n",
        path,
    )
    return write_if_changed(path, text, patched)


def patch_modeling_gpt_oss(path: Path) -> bool:
    text = path.read_text()
    patched = text

    if "is_tinygemm2_disabled" not in patched:
        if "from tensorrt_llm._utils import get_sm_version\n" in patched:
            patched, _ = replace_once(
                patched,
                "from tensorrt_llm._utils import get_sm_version\n",
                "from tensorrt_llm._utils import get_sm_version, is_tinygemm2_disabled\n",
                path,
            )
        else:
            patched, _ = replace_once(
                patched,
                "from tensorrt_llm._utils import get_hf_rope_theta, get_sm_version\n",
                "from tensorrt_llm._utils import (get_hf_rope_theta, get_sm_version,\n"
                "                                 is_tinygemm2_disabled)\n",
                path,
            )

    if "not is_tinygemm2_disabled()" not in patched:
        old_gate = (
            "        # Skip tinygemm2 optimization when LoRA is active (tinygemm2 doesn't support LoRA)\n"
            "        use_tinygemm = (get_sm_version() in [90, 100, 103]\n"
        )
        new_gate = (
            "        # Skip tinygemm2 optimization when LoRA is active (tinygemm2 doesn't support LoRA)\n"
            "        # Set TLLM_DISABLE_TINYGEMM2=1 to fall back to self.gate.\n"
            "        use_tinygemm = (not is_tinygemm2_disabled()\n"
            "                        and get_sm_version() in [90, 100, 103]\n"
        )
        patched, _ = replace_once(patched, old_gate, new_gate, path)

    return write_if_changed(path, text, patched)


def patch_utils(path: Path) -> bool:
    text = path.read_text()
    if "def is_tinygemm2_disabled()" in text:
        return False

    insertion = (
        "\n\ndef is_tinygemm2_disabled() -> bool:\n"
        "    \"\"\"True if the tinygemm2 kernel (gpt-oss small-M router GEMM) is disabled.\n\n"
        "    Set ``TLLM_DISABLE_TINYGEMM2=1`` to disable tinygemm2. When disabled,\n"
        "    every dispatch site falls back to its standard GEMM path (cuBLAS /\n"
        "    ``F.linear``), so there is no functional change beyond kernel choice.\n"
        "    \"\"\"\n"
        "    return os.environ.get(\"TLLM_DISABLE_TINYGEMM2\", \"0\") == \"1\"\n\n\n"
        "def mpi_rank():\n"
    )
    patched, _ = replace_once(text, "\n\ndef mpi_rank():\n", insertion, path)
    return write_if_changed(path, text, patched)


def main() -> None:
    with LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        apply_patch()


def apply_patch() -> None:
    paths = {
        "trtllm_moe": package_path(
            "_torch",
            "auto_deploy",
            "custom_ops",
            "fused_moe",
            "trtllm_moe.py",
        ),
        "modeling_gpt_oss": package_path("_torch", "models", "modeling_gpt_oss.py"),
        "utils": package_path("_utils.py"),
    }

    changed = []
    if patch_trtllm_moe(paths["trtllm_moe"]):
        changed.append(paths["trtllm_moe"])
    if patch_modeling_gpt_oss(paths["modeling_gpt_oss"]):
        changed.append(paths["modeling_gpt_oss"])
    if patch_utils(paths["utils"]):
        changed.append(paths["utils"])

    if changed:
        for path in changed:
            print(f"Applied TensorRT-LLM PR #15790 patch to {path}")
    else:
        print("TensorRT-LLM PR #15790 patch already applied")


if __name__ == "__main__":
    main()
