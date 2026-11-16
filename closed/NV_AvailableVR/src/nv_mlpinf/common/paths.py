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


import dataclasses
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import Final
import os
import shutil
import sys
import yaml


# Package installation root — points to site-packages/nv_mlpinf/ after pip install.
# Use this to locate bundled files (scripts, requirements, etc.) inside the package.
CODE_DIR: Final = _pkg_files("nv_mlpinf")

# Bundled template — ships with the package, never modified at runtime.
CONFIG_TEMPLATE: Final[Path] = Path(str(CODE_DIR)) / "nv_mlpinf_paths.yml"

# User config — edit this file to set persistent paths; survives reinstalls.
_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "nv_mlpinf" / "paths.yml"


@dataclasses.dataclass(frozen=True)
class _PathSpec:
    """Descriptor for a single resolvable path.

    Resolution order: env var → YAML config key → hardcoded default.
    ``env`` defaults to ``key.upper()`` when omitted, e.g. key='build_dir' → env='BUILD_DIR'.
    """
    key: str
    default: str
    env: str = ""

    def __post_init__(self):
        if not self.env:
            object.__setattr__(self, 'env', self.key.upper())

    def resolve(self, cfg: dict, sources: dict) -> Path:
        if self.env in os.environ:
            sources[self.key] = f"env: {self.env}"
            return Path(os.environ[self.env])
        if self.key in cfg:
            sources[self.key] = "config"
            return Path(cfg[self.key])
        sources[self.key] = "default"
        return Path(self.default)


def _init_user_path_config() -> None:
    """Create ~/.config/nv_mlpinf/paths.yml from the bundled template if it doesn't exist.

    Skipped when NV_MLPINF_PATHS_CONFIG is set (user manages their own config location).
    """
    if "NV_MLPINF_PATHS_CONFIG" in os.environ:
        p = Path(os.environ["NV_MLPINF_PATHS_CONFIG"])
        if not p.exists():
            raise FileNotFoundError(f"NV_MLPINF_PATHS_CONFIG points to a non-existent file: {p}")
        return
    if _DEFAULT_CONFIG_PATH.exists():
        return
    if not CONFIG_TEMPLATE.exists():
        return
    _DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(CONFIG_TEMPLATE, _DEFAULT_CONFIG_PATH)
    print(
        f"nv_mlpinf: created default config at {_DEFAULT_CONFIG_PATH}\n"
        f"  Edit this file to set your runtime paths.",
        file=sys.stderr,
    )


_init_user_path_config()


def _find_config_path() -> Path | None:
    """Return the YAML config path: env var → default location → None."""
    if "NV_MLPINF_PATHS_CONFIG" in os.environ:
        p = Path(os.environ["NV_MLPINF_PATHS_CONFIG"])
        if not p.exists():
            raise FileNotFoundError(f"NV_MLPINF_PATHS_CONFIG points to a non-existent file: {p}")
        return p
    if _DEFAULT_CONFIG_PATH.exists():
        return _DEFAULT_CONFIG_PATH
    return None


_CONFIG_PATH: Path | None = _find_config_path()  # None means no YAML config → use hardcoded defaults


def _load_yaml_config() -> dict:
    if _CONFIG_PATH is not None and _CONFIG_PATH.exists():
        try:
            return yaml.safe_load(_CONFIG_PATH.read_text()) or {}
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse config file {_CONFIG_PATH}: {e}") from e
    return {}


_cfg = _load_yaml_config()
_path_sources: dict[str, str] = {}  # populated by _PathSpec.resolve, used by show_paths

PROJECT_BASE_DIR:    Final[Path] = _PathSpec("project_base_dir",    "/work").resolve(_cfg, _path_sources)
BUILD_DIR:           Final[Path] = _PathSpec("build_dir",           "/work/build").resolve(_cfg, _path_sources)
MLPERF_SCRATCH_PATH: Final[Path] = _PathSpec("mlperf_scratch_path", "/home/mlperf_inference_storage").resolve(_cfg, _path_sources)
TRTLLM_DIR:          Final[Path] = _PathSpec("trtllm_dir",          "/work/3rdparty/trtllm").resolve(_cfg, _path_sources)
MLCOMMONS_INF_REPO:  Final[Path] = _PathSpec("mlcommons_inf_repo",  "/work/3rdparty/mlc-inference").resolve(_cfg, _path_sources)

# env names intentionally differ from key.upper() for these two:
RESULTS_SUBMISSION_DIR: Final[Path] = _PathSpec("results_submission_dir", str(BUILD_DIR / "artifacts"),         env="ARTIFACTS_DIR").resolve(_cfg, _path_sources)
RESULTS_STAGING_DIR:    Final[Path] = _PathSpec("results_staging_dir",    str(BUILD_DIR / "submission-staging"), env="ARTIFACTS_STAGING").resolve(_cfg, _path_sources)

# MODEL_DIR, DATA_DIR, PREPROCESSED_DATA_DIR: always derived from MLPERF_SCRATCH_PATH;
# override via env var only — not exposed as config file keys.
MODEL_DIR:              Final[Path] = _PathSpec("model_dir",             str(MLPERF_SCRATCH_PATH / "models")).resolve({}, _path_sources)
DATA_DIR:               Final[Path] = _PathSpec("data_dir",              str(MLPERF_SCRATCH_PATH / "data")).resolve({}, _path_sources)
PREPROCESSED_DATA_DIR:  Final[Path] = _PathSpec("preprocessed_data_dir", str(MLPERF_SCRATCH_PATH / "preprocessed_data")).resolve({}, _path_sources)

# Fix source labels for derived paths: distinguish env-override from scratch-path derivation.
for _key, _subdir in (("model_dir", "models"), ("data_dir", "data"), ("preprocessed_data_dir", "preprocessed_data")):
    if _path_sources.get(_key) == "default":
        _path_sources[_key] = "derived from mlperf_scratch_path"


# Paths expected to exist before running (input paths, not output dirs).
# build_dir / results_* are created on demand and intentionally excluded.
_INPUT_PATHS: Final[list[tuple[str, Path, str]]] = [
    ("project_base_dir",       PROJECT_BASE_DIR,       "PROJECT_BASE_DIR"),
    ("mlperf_scratch_path",    MLPERF_SCRATCH_PATH,    "MLPERF_SCRATCH_PATH"),
    ("trtllm_dir",             TRTLLM_DIR,             "TRTLLM_DIR"),
    ("mlcommons_inf_repo",     MLCOMMONS_INF_REPO,     "MLCOMMONS_INF_REPO"),
    ("model_dir",              MODEL_DIR,              "MODEL_DIR"),
    ("data_dir",               DATA_DIR,               "DATA_DIR"),
    ("preprocessed_data_dir",  PREPROCESSED_DATA_DIR,  "PREPROCESSED_DATA_DIR"),
]


def validate_paths() -> list[str]:
    """Return a warning string for each input path that does not exist on disk."""
    config_hint = f"edit {_CONFIG_PATH}" if _CONFIG_PATH is not None else f"edit {_DEFAULT_CONFIG_PATH}"

    warnings = []
    for key, path, env_var in _INPUT_PATHS:
        if not path.exists():
            warnings.append(
                f"WARNING: path '{key}' = {path} does not exist.\n"
                f"  Fix: {config_hint}  or  export {env_var}=<path>"
            )
    return warnings


# Emit warnings at import time so they appear before any downstream import failures.
for _w in validate_paths():
    print(_w, file=sys.stderr)
