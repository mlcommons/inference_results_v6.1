#!/usr/bin/env python3
"""Print the CPU NUMA node closest to the first CUDA-visible GPU.

Used by `numactl --cpunodebind=<N> --membind=<N>` in the vllm_worker
sflow task to pin each dynamo.vllm worker process to its GPU's local
NUMA domain.

Strategy mirrors `_get_device_numa_node` in
src/nv_mlpinf/benchmarks/q3vl/vllm/src/mlperf_inf_mm_q3vl_nv/deploy.py:

1. Resolve the first CUDA-visible GPU to an NVML handle, honoring
   CUDA_VISIBLE_DEVICES when it contains physical indices or GPU UUIDs.
2. Ask NVML directly via nvmlDeviceGetNumaNodeId. On Grace Hopper this
   returns the *GPU-memory* NUMA node which often has no CPUs.
3. If that node has no CPUs (sysfs cpulist empty), fall through to
   nvmlDeviceGetCpuAffinity, pick the first set CPU, and look up which
   /sys/devices/system/node/nodeN/cpulist contains it.
4. If both fail, print 0 (caller can choose to skip NUMA binding).

Prints the resolved NUMA node id (int) to stdout on success.
Exits 0 always; never raises so the caller can `NUMA=$(... || echo 0)`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _cpus_on_node(node_id: int) -> str:
    try:
        return Path(f"/sys/devices/system/node/node{node_id}/cpulist").read_text().strip()
    except OSError:
        return ""


def _cpu_in_cpulist(cpu_id: int, cpulist: str) -> bool:
    for part in cpulist.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            if int(lo) <= cpu_id <= int(hi):
                return True
        elif part.isdigit() and int(part) == cpu_id:
            return True
    return False


def _numa_node_for_cpu(cpu_id: int) -> int | None:
    node_path = Path("/sys/devices/system/node")
    if not node_path.exists():
        return None
    for entry in node_path.iterdir():
        if not entry.name.startswith("node") or not entry.name[4:].isdigit():
            continue
        cpulist = _cpus_on_node(int(entry.name[4:]))
        if cpulist and _cpu_in_cpulist(cpu_id, cpulist):
            return int(entry.name[4:])
    return None


def _first_cuda_visible_device() -> str | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return None
    first = visible_devices.split(",", 1)[0].strip()
    return first or None


def _handle_for_first_visible_gpu(pynvml):
    first_visible = _first_cuda_visible_device()

    if first_visible:
        if first_visible.isdigit():
            # In this Slurm/Pyxis flow CUDA_VISIBLE_DEVICES is set to the
            # physical GPU index assigned to the worker. NVML index 0 would
            # otherwise pin every single-GPU worker to physical GPU 0's NUMA.
            return pynvml.nvmlDeviceGetHandleByIndex(int(first_visible))
        if first_visible.startswith(("GPU-", "MIG-")):
            try:
                return pynvml.nvmlDeviceGetHandleByUUID(first_visible)
            except TypeError:
                return pynvml.nvmlDeviceGetHandleByUUID(first_visible.encode())

    return pynvml.nvmlDeviceGetHandleByIndex(0)


def main() -> int:
    try:
        import pynvml
    except ImportError:
        # pynvml not available — skip; caller can interpret "0" as "no binding".
        print(0)
        return 0

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        print(0)
        return 0

    try:
        handle = _handle_for_first_visible_gpu(pynvml)
    except pynvml.NVMLError:
        pynvml.nvmlShutdown()
        print(0)
        return 0

    # Try direct NUMA query first.
    try:
        numa_id = pynvml.nvmlDeviceGetNumaNodeId(handle)
        if _cpus_on_node(numa_id):
            print(numa_id)
            pynvml.nvmlShutdown()
            return 0
    except pynvml.NVMLError:
        pass

    # Fallback: derive NUMA from CPU affinity mask (Grace Hopper path).
    try:
        cpu_set_size = ((os.cpu_count() or 1) + 63) // 64
        affinity_mask = pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_set_size)
        for i, mask in enumerate(affinity_mask):
            if not mask:
                continue
            first_cpu = i * 64 + (mask & -mask).bit_length() - 1
            numa_node = _numa_node_for_cpu(first_cpu)
            if numa_node is not None:
                print(numa_node)
                pynvml.nvmlShutdown()
                return 0
    except pynvml.NVMLError:
        pass

    pynvml.nvmlShutdown()
    # All else fails — default to 0.
    print(0, file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
