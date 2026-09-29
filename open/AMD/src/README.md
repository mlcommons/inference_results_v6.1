# Submission Source Staging

This runner-owned folder stages benchmark/runtime source intended for the final
`closed/AMD/src/` submission area.

For q12,200 DLRM-v3, `src/dlrm-v3/` should contain the code needed to build and
run the benchmark itself:

- harness benchmark source (`benchmarks/`, `inference_harness/`, benchmark-local `tools/`)
- GR model/kernel source (`generative_recommenders/`, configs, local dataset shims)
- pynve ROCm source required to rebuild the NVE path
- q12,200 LoadGen configs that are consumed by the harness

Do **not** put top-level runner orchestration here just because it is a shell
script. Host setup/build/run wrappers belong under `submission/setup/`; assembly
and validation helpers belong under `submission/tools/`.

Exclude `.git`, build products, caches, datasets, checkpoints, and raw benchmark
artifacts. The exact copy/prune rules are tracked in
`plans/plan_5_submission_assembly.md`.
