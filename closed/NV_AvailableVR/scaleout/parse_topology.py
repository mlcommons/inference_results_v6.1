#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Parse server-topology.json and output shell-sourceable variable assignments.

Usage (from shell):
    eval "$(python3 parse_topology.py --host-vol /host/path --container-vol /work topology.json)"

Dispatches by server_type:

  trtllm-ifb  →  IFB (in-flight batching) topology
    Shell vars: TOPO_SERVER_TYPE, TOPO_MODEL_PATH, TOPO_DP_MULTIPLICITY,
                TOPO_GPUS_PER_DP_RANK, TOPO_GPUS_PER_NODE, TOPO_TRTLLM_YML,
                TOPO_ENV_EXPORTS

  trtllm-disagg  →  Disaggregated (context/generation split) topology
    Shell vars: TOPO_SERVER_TYPE, TOPO_MODEL_PATH, TOPO_GPUS_PER_NODE,
                TOPO_NUM_CTX_WORKERS, TOPO_NUM_GEN_WORKERS,
                TOPO_GPUS_PER_CTX_WORKER, TOPO_GPUS_PER_GEN_WORKER,
                TOPO_NUM_MASTER_SERVERS,
                TOPO_CTX_TRTLLM_YML, TOPO_CTX_ENV_EXPORTS,
                TOPO_GEN_TRTLLM_YML, TOPO_GEN_ENV_EXPORTS

Paths in the topology JSON (*_yml) are stored as relative paths.
They are resolved to absolute paths using --host-vol and --container-vol:
    host path:      <host-vol>/<relative-path>      (for reading files on the host)
    container path: <container-vol>/<relative-path>  (for passing to commands inside the container)
"""

import argparse
import json
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

KNOWN_SERVER_TYPES = ("trtllm-ifb", "trtllm-disagg")


# ── Helpers ───────────────────────────────────────────────────────────────


def shell_escape(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def resolve_path(relative_path: str, base: str) -> str:
    """Resolve a relative path against a base directory. Absolute paths pass through unchanged."""
    if not relative_path:
        return ""
    if relative_path.startswith("/"):
        return relative_path
    return f"{base}/{relative_path}"


def parse_env_yml(path: str) -> str:
    """Parse a flat YAML file (KEY: VALUE) into comma-separated KEY=VALUE.

    Only handles simple single-level KEY: VALUE or KEY: 'VALUE' pairs.
    No dependency on PyYAML.
    """
    p = Path(path)
    if not p.is_file():
        print(f"ERROR: env_yml not found: {path}", file=sys.stderr)
        sys.exit(1)

    pairs = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*['\"]?(.+?)['\"]?\s*$", line)
            if m:
                pairs.append(f"{m.group(1)}={m.group(2)}")
            else:
                print(f"WARNING: skipping unparseable env_yml line: {line}", file=sys.stderr)

    return ",".join(pairs)


def _validate_positive_int(name: str, value) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


# ── Base class ────────────────────────────────────────────────────────────


class BaseTopology(ABC):
    server_type: str
    model_path: str
    gpus_per_node: int

    @abstractmethod
    def to_shell_vars(self, host_vol: str, container_vol: str) -> str:
        ...

    @abstractmethod
    def log_summary(self, topo_path: Path, host_vol: str, container_vol: str) -> None:
        ...


# ── IFB topology (trtllm-ifb) ────────────────────────────────────────────
#
# Example server-topology.json:
#
#   {
#       "server_type": "trtllm-ifb",
#       "model_path": "/home/mlperf_inference_storage/models/gpt-oss/gpt-oss-120b",
#       "dp_multiplicity": 8,
#       "gpus_per_dp_rank": 1,
#       "gpus_per_node": 4,
#       "trtllm_yml": "configs/.../trtllm-serve-ifb-1gpu.yaml",
#       "env_yml": "configs/.../trtllm-serve-ifb-1gpu-env.yaml"
#   }
#
# Total GPUs = dp_multiplicity * gpus_per_dp_rank  (e.g. 8 * 1 = 8 GPUs)
# Total nodes = ceil(total_gpus / gpus_per_node)    (e.g. ceil(8 / 4) = 2 nodes)


@dataclass
class IFBTopology(BaseTopology):
    server_type: str
    model_path: str
    dp_multiplicity: int
    gpus_per_dp_rank: int
    gpus_per_node: int
    trtllm_yml: str
    env_yml: str

    def __post_init__(self):
        required = {
            "server_type": self.server_type,
            "model_path": self.model_path,
            "dp_multiplicity": self.dp_multiplicity,
            "gpus_per_dp_rank": self.gpus_per_dp_rank,
            "gpus_per_node": self.gpus_per_node,
            "trtllm_yml": self.trtllm_yml,
            "env_yml": self.env_yml,
        }
        missing = [k for k, v in required.items() if v is None or v == "" or v == 0]
        if missing:
            raise ValueError(f"Missing required topology fields: {', '.join(missing)}")
        _validate_positive_int("dp_multiplicity", self.dp_multiplicity)
        _validate_positive_int("gpus_per_dp_rank", self.gpus_per_dp_rank)
        _validate_positive_int("gpus_per_node", self.gpus_per_node)

    @classmethod
    def from_dict(cls, data: dict) -> "IFBTopology":
        return cls(
            server_type=data.get("server_type", ""),
            model_path=data.get("model_path", ""),
            dp_multiplicity=data.get("dp_multiplicity", 0),
            gpus_per_dp_rank=data.get("gpus_per_dp_rank", 0),
            gpus_per_node=data.get("gpus_per_node", 0),
            trtllm_yml=data.get("trtllm_yml", ""),
            env_yml=data.get("env_yml", ""),
        )

    def to_shell_vars(self, host_vol: str, container_vol: str) -> str:
        container_trtllm_yml = resolve_path(self.trtllm_yml, container_vol)
        env_exports = ""
        if self.env_yml:
            env_exports = parse_env_yml(resolve_path(self.env_yml, host_vol))

        return "\n".join([
            f"TOPO_SERVER_TYPE={shell_escape(self.server_type)}",
            f"TOPO_MODEL_PATH={shell_escape(self.model_path)}",
            f"TOPO_DP_MULTIPLICITY={shell_escape(str(self.dp_multiplicity))}",
            f"TOPO_GPUS_PER_DP_RANK={shell_escape(str(self.gpus_per_dp_rank))}",
            f"TOPO_GPUS_PER_NODE={shell_escape(str(self.gpus_per_node))}",
            f"TOPO_TRTLLM_YML={shell_escape(container_trtllm_yml)}",
            f"TOPO_ENV_EXPORTS={shell_escape(env_exports)}",
        ])

    def log_summary(self, topo_path: Path, host_vol: str, container_vol: str) -> None:
        container_trtllm_yml = resolve_path(self.trtllm_yml, container_vol)
        env_exports = ""
        if self.env_yml:
            env_exports = parse_env_yml(resolve_path(self.env_yml, host_vol))
        print(f"Parsed IFB topology from {topo_path}:", file=sys.stderr)
        print(f"  server_type={self.server_type} model_path={self.model_path}", file=sys.stderr)
        print(f"  dp_multiplicity={self.dp_multiplicity} gpus_per_dp_rank={self.gpus_per_dp_rank} gpus_per_node={self.gpus_per_node}", file=sys.stderr)
        print(f"  trtllm_yml={self.trtllm_yml} -> {container_trtllm_yml}", file=sys.stderr)
        print(f"  env_yml={self.env_yml}", file=sys.stderr)
        if env_exports:
            print(f"  env_exports={env_exports}", file=sys.stderr)


# ── Disagg topology (trtllm-disagg) ──────────────────────────────────────
#
# Example server-topology.json:
#
#   {
#       "server_type": "trtllm-disagg",
#       "model_path": "/home/mlperf_inference_storage/models/gpt-oss/gpt-oss-120b",
#       "gpus_per_node": 4,
#       "num_ctx_workers": 24,
#       "num_gen_workers": 12,
#       "num_master_servers": 1,
#       "gpus_per_ctx_worker": 1,
#       "gpus_per_gen_worker": 4,
#       "ctx_trtllm_yml": "configs/.../trtllm-serve-disagg-ctx-1gpu.yaml",
#       "ctx_env_yml": "configs/.../trtllm-serve-disagg-ctx-1gpu-env.yaml",
#       "gen_trtllm_yml": "configs/.../trtllm-serve-disagg-gen-4gpu.yaml",
#       "gen_env_yml": "configs/.../trtllm-serve-disagg-gen-4gpu-env.yaml"
#   }
#
# Total GPUs = (num_ctx_workers * gpus_per_ctx_worker) + (num_gen_workers * gpus_per_gen_worker)
#              e.g. (24 * 1) + (12 * 4) = 24 + 48 = 72 GPUs
# Total nodes = ceil(total_gpus / gpus_per_node)  (e.g. ceil(72 / 4) = 18 nodes)


@dataclass
class DisaggTopology(BaseTopology):
    server_type: str
    model_path: str
    gpus_per_node: int
    num_ctx_workers: int
    num_gen_workers: int
    gpus_per_ctx_worker: int
    gpus_per_gen_worker: int
    ctx_trtllm_yml: str
    ctx_env_yml: str
    gen_trtllm_yml: str
    gen_env_yml: str
    num_master_servers: int = 1

    def __post_init__(self):
        required = {
            "server_type": self.server_type,
            "model_path": self.model_path,
            "gpus_per_node": self.gpus_per_node,
            "num_ctx_workers": self.num_ctx_workers,
            "num_gen_workers": self.num_gen_workers,
            "gpus_per_ctx_worker": self.gpus_per_ctx_worker,
            "gpus_per_gen_worker": self.gpus_per_gen_worker,
            "ctx_trtllm_yml": self.ctx_trtllm_yml,
            "ctx_env_yml": self.ctx_env_yml,
            "gen_trtllm_yml": self.gen_trtllm_yml,
            "gen_env_yml": self.gen_env_yml,
        }
        missing = [k for k, v in required.items() if v is None or v == "" or v == 0]
        if missing:
            raise ValueError(f"Missing required topology fields: {', '.join(missing)}")
        _validate_positive_int("gpus_per_node", self.gpus_per_node)
        _validate_positive_int("num_ctx_workers", self.num_ctx_workers)
        _validate_positive_int("num_gen_workers", self.num_gen_workers)
        _validate_positive_int("gpus_per_ctx_worker", self.gpus_per_ctx_worker)
        _validate_positive_int("gpus_per_gen_worker", self.gpus_per_gen_worker)
        _validate_positive_int("num_master_servers", self.num_master_servers)

    @classmethod
    def from_dict(cls, data: dict) -> "DisaggTopology":
        return cls(
            server_type=data.get("server_type", ""),
            model_path=data.get("model_path", ""),
            gpus_per_node=data.get("gpus_per_node", 0),
            num_ctx_workers=data.get("num_ctx_workers", 0),
            num_gen_workers=data.get("num_gen_workers", 0),
            gpus_per_ctx_worker=data.get("gpus_per_ctx_worker", 0),
            gpus_per_gen_worker=data.get("gpus_per_gen_worker", 0),
            ctx_trtllm_yml=data.get("ctx_trtllm_yml", ""),
            ctx_env_yml=data.get("ctx_env_yml", ""),
            gen_trtllm_yml=data.get("gen_trtllm_yml", ""),
            gen_env_yml=data.get("gen_env_yml", ""),
            num_master_servers=data.get("num_master_servers", 1),
        )

    def to_shell_vars(self, host_vol: str, container_vol: str) -> str:
        ctx_container_yml = resolve_path(self.ctx_trtllm_yml, container_vol)
        gen_container_yml = resolve_path(self.gen_trtllm_yml, container_vol)
        ctx_env_exports = ""
        if self.ctx_env_yml:
            ctx_env_exports = parse_env_yml(resolve_path(self.ctx_env_yml, host_vol))
        gen_env_exports = ""
        if self.gen_env_yml:
            gen_env_exports = parse_env_yml(resolve_path(self.gen_env_yml, host_vol))

        return "\n".join([
            f"TOPO_SERVER_TYPE={shell_escape(self.server_type)}",
            f"TOPO_MODEL_PATH={shell_escape(self.model_path)}",
            f"TOPO_GPUS_PER_NODE={shell_escape(str(self.gpus_per_node))}",
            f"TOPO_NUM_CTX_WORKERS={shell_escape(str(self.num_ctx_workers))}",
            f"TOPO_NUM_GEN_WORKERS={shell_escape(str(self.num_gen_workers))}",
            f"TOPO_GPUS_PER_CTX_WORKER={shell_escape(str(self.gpus_per_ctx_worker))}",
            f"TOPO_GPUS_PER_GEN_WORKER={shell_escape(str(self.gpus_per_gen_worker))}",
            f"TOPO_NUM_MASTER_SERVERS={shell_escape(str(self.num_master_servers))}",
            f"TOPO_CTX_TRTLLM_YML={shell_escape(ctx_container_yml)}",
            f"TOPO_CTX_ENV_EXPORTS={shell_escape(ctx_env_exports)}",
            f"TOPO_GEN_TRTLLM_YML={shell_escape(gen_container_yml)}",
            f"TOPO_GEN_ENV_EXPORTS={shell_escape(gen_env_exports)}",
        ])

    def log_summary(self, topo_path: Path, host_vol: str, container_vol: str) -> None:
        ctx_container_yml = resolve_path(self.ctx_trtllm_yml, container_vol)
        gen_container_yml = resolve_path(self.gen_trtllm_yml, container_vol)
        total_gpus = (self.num_ctx_workers * self.gpus_per_ctx_worker
                      + self.num_gen_workers * self.gpus_per_gen_worker)
        print(f"Parsed disagg topology from {topo_path}:", file=sys.stderr)
        print(f"  server_type={self.server_type} model_path={self.model_path}", file=sys.stderr)
        print(f"  gpus_per_node={self.gpus_per_node} total_gpus={total_gpus} num_master_servers={self.num_master_servers}", file=sys.stderr)
        print(f"  ctx: {self.num_ctx_workers} workers x {self.gpus_per_ctx_worker} GPUs = {self.num_ctx_workers * self.gpus_per_ctx_worker} GPUs", file=sys.stderr)
        print(f"    trtllm_yml={self.ctx_trtllm_yml} -> {ctx_container_yml}", file=sys.stderr)
        print(f"    env_yml={self.ctx_env_yml}", file=sys.stderr)
        print(f"  gen: {self.num_gen_workers} workers x {self.gpus_per_gen_worker} GPUs = {self.num_gen_workers * self.gpus_per_gen_worker} GPUs", file=sys.stderr)
        print(f"    trtllm_yml={self.gen_trtllm_yml} -> {gen_container_yml}", file=sys.stderr)
        print(f"    env_yml={self.gen_env_yml}", file=sys.stderr)


# ── Factory ───────────────────────────────────────────────────────────────


def load_topology(path: Path) -> BaseTopology:
    """Load topology JSON, dispatch to the correct dataclass based on server_type."""
    with open(path) as f:
        data = json.load(f)

    server_type = data.get("server_type", "")
    if server_type == "trtllm-ifb":
        return IFBTopology.from_dict(data)
    elif server_type == "trtllm-disagg":
        return DisaggTopology.from_dict(data)
    else:
        raise ValueError(
            f"Unknown server_type: {server_type!r}. "
            f"Expected one of: {', '.join(KNOWN_SERVER_TYPES)}")


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse server-topology.json and output shell-sourceable variable assignments.",
        epilog='Usage from shell:  eval "$(python3 parse_topology.py --host-vol /host/path --container-vol /work topology.json)"',
    )
    parser.add_argument(
        "topology_json",
        type=Path,
        help="Path to server-topology.json",
    )
    parser.add_argument(
        "--host-vol",
        required=True,
        help="Host-side workspace path (used to resolve relative paths for reading files)",
    )
    parser.add_argument(
        "--container-vol",
        required=True,
        help="Container-side workspace path (used to generate absolute paths for in-container commands)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    topo_path: Path = args.topology_json
    if not topo_path.is_file():
        print(f"ERROR: server-topology.json not found: {topo_path}", file=sys.stderr)
        sys.exit(1)

    try:
        topo = load_topology(topo_path)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(topo.to_shell_vars(args.host_vol, args.container_vol))
    topo.log_summary(topo_path, args.host_vol, args.container_vol)


if __name__ == "__main__":
    main()
