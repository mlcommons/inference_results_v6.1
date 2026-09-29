# MLPerf Inference v6.1 — Cisco Submission

This is the Cisco closed-division submission for MLPerf Inference v6.1.  It
covers five systems, four LLM benchmarks, and the Offline, Server, and
Interactive scenarios.

---

## Systems

| System ID | Nodes | Accelerators | Memory | Framework |
|---|---|---|---|---|
| `Cisco_UCS_B300x16_G200_4x2_RO` | 2 | 16× NVIDIA B300-SXM-270GB | HBM3e | TensorRT-LLM 1.3.0rc0 |
| `Cisco_UCS_B300x64_G200_4x2_RO` | 8 | 64× NVIDIA B300-SXM-270GB | HBM3e | TensorRT-LLM 1.3.0rc0 |
| `h200` | 1 | 8× NVIDIA H200 141 GB | HBM3e | vLLM |
| `mi350x` | 1 | 8× AMD Instinct MI350X 288 GB | HBM3e | vLLM |
| `mi350x-h200` | 2 | 8× H200 + 8× MI350X | HBM3e | vLLM |

All B300 nodes use Intel Xeon 6776P CPUs and a Cisco G200 dual-plane
rail-optimized Ethernet fabric (16× 400 Gb/s GPU-direct interfaces per node).
The H200, MI350X, and MI350X-H200 systems use AMD EPYC CPUs.

---

## Benchmarks and scenarios

| Benchmark | Systems | Scenarios |
|---|---|---|
| `deepseek-r1` | B300x16 | Offline, Server |
| `gpt-oss-120b` | B300x16, B300x64, h200, mi350x, mi350x-h200 | Offline, Server |
| `llama2-70b-99` / `llama2-70b-99.9` | all | Offline, Server, Interactive |
| `llama3.1-8b` | B300x16, h200, mi350x, mi350x-h200 | Offline, Server, Interactive |

---

## Software stack

| Component | B300 systems | H200 / MI350X systems |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS (kernel 6.17) | Ubuntu 22.04 |
| GPU driver | NVIDIA 595.71.05 | NVIDIA 12.4 / ROCm 6.4 |
| CUDA / ROCm | CUDA 13.1.80 | CUDA 12.4 / ROCm 6.4 |
| Inference framework | TensorRT 10.14.1.48 + TensorRT-LLM 1.3.0rc0 | vLLM |
| LoadGen | 6.0.16 | 6.0.16 |

---

## Repository layout

```
closed/Cisco/
├── configs/          # Per-profile sflow configs and run instructions
├── results/          # LoadGen logs and accuracy results
├── systems/          # System descriptor JSON files
├── scaleout/         # portable SFlow workflows and scheduler template
├── scripts/          # Utility scripts (e.g. print_harness_result.py)
└── src/
    ├── nv_mlpinf/    # Shared TensorRT-LLM harness (Python package)
    └── vllm-heterogeneous/  # vLLM harness for H200/MI350X/MI350X-H200
```

`src/nv_mlpinf/` is the Python package declared by `pyproject.toml`.  It is
used for all B300 runs.  The `src/vllm-heterogeneous/` tree is the harness
for the vLLM-based systems.

---

## Running benchmarks

### B300 systems (TensorRT-LLM)

Set common environment variables:

```bash
export CISCO_SUBMISSION_ROOT=/path/to/submission/closed/Cisco
export CISCO_RUN_ROOT="${CISCO_SUBMISSION_ROOT}"
export MLPERF_DATA_DIR=/path/to/mlperf-data
export MLPERF_CONTAINER_IMAGE=/path/to/pinned-container-image
export MLPERF_SFLOW_BIN=/path/to/sflow
export MLPERF_SLURM_ACCOUNT=your-account
export MLPERF_SLURM_PARTITION=your-partition
export MLPERF_OUTPUT_DIR=/path/to/output
```

Install the harness:

```bash
python3 -m venv "${CISCO_RUN_ROOT}/.venv"
source "${CISCO_RUN_ROOT}/.venv/bin/activate"
python -m pip install -e "${CISCO_RUN_ROOT}[llm]"
```

See `configs/README.md` for the full per-profile run commands.  Each profile
directory (e.g. `configs/deepseek_r1/Cisco_UCS_B300x16_G200_4x2_RO/TRTLLM/Offline/`) contains a
`README.md` with the exact `export` values and audit instructions.

### H200 / MI350X / MI350X-H200 (vLLM)

See `src/vllm-heterogeneous/README.md` for container setup and run
instructions.
