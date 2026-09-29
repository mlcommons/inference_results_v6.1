# DLRM-v3 Submission Source Payload

This directory duplicates the source payload for the AMD q12,200 DLRM-v3 ROCm
submission branch. It is intentionally self-contained for submission assembly:

- `harness/`: pinned `AMD-AGI/dlrm-v3-harness-rocm` source (`5088d7c`).
- `gr/`: pinned `AMD-AGI/dlrm-v3-gr-rocm` source (`7eb52e9`).
- `pynve-rocm/`: pinned NVE ROCm source (`d34a5fe`).

Runner setup and orchestration wrappers live under `submission/setup/dlrm-v3/`;
assembly/checker helpers live under `submission/tools/dlrm-v3/`.
