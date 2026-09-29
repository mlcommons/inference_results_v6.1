# Submission Setup Staging

This runner-owned folder stages setup/build/run material intended for the final
`closed/AMD/setup/` submission area.

For q12,200 DLRM-v3, setup material should describe how to reproduce the ROCm
container/runtime environment from the runner:

- data/checkpoint staging instructions (`setup_data.sh` flow)
- private/vendored source setup instructions (`setup_workspace.sh` flow)
- container creation and FBGEMM/pynve build instructions (`setup_submission.sh` flow)
- GOLD Server and AccuracyOnly launch commands (`run_gold.sh`, `run_accuracy.sh`)
- TEST08 launch/verification commands (`_test08_chain.sh`)
- environment defaults and host override notes for MI355X certification vs non-cert probes

The setup area may include runner shell wrappers because they are setup/run
orchestration, not benchmark implementation source.

Do not copy Docker layers, datasets, checkpoints, build outputs, or local
artifacts here.
