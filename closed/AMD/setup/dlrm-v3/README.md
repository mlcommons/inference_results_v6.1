# DLRM-v3 Submission Setup Payload

This folder contains runner setup and execution wrappers for the AMD q12,200
DLRM-v3 ROCm submission staging tree.

- `run.sh`: high-level stage orchestrator.
- `scripts/build/`: data, workspace, container, FBGEMM, pynve, and stack setup.
- `scripts/run/`: GOLD Server, AccuracyOnly, and TEST08 launch wrappers.
- `RUNNER_README.md`: source runner quickstart and host requirements.
- `path_conventions.md`: final submission path normalization notes.

These are setup/run orchestration files, so they live in `setup/` rather than
`src/`.

Before final submission, adapt any runner-centric paths to the generated
submission runtime root (for example `/lab-mlperf-inference/code` and
`/lab-mlperf-inference/results`), following the AMD v6.0 convention.
