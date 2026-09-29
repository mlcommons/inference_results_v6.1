# Llama2-70B x64 Offline IFB

This profile uses 64 TP1 endpoints and target QPS 3600. It supports
Performance, Accuracy, and TEST06.

```bash
export PROFILE="${CISCO_SUBMISSION_ROOT}/configs/llama2_70b/Cisco_UCS_B300x64_G200_4x2_RO/TRTLLM/Offline"
export MLPERF_OUTPUT_DIR=/path/to/output/llama2-x64-offline
export MLPERF_SERVER_HOSTS=host01,host01,host01,host01,host01,host01,host01,host01,host02,host02,host02,host02,host02,host02,host02,host02,host03,host03,host03,host03,host03,host03,host03,host03,host04,host04,host04,host04,host04,host04,host04,host04,host05,host05,host05,host05,host05,host05,host05,host05,host06,host06,host06,host06,host06,host06,host06,host06,host07,host07,host07,host07,host07,host07,host07,host07,host08,host08,host08,host08,host08,host08,host08,host08
```

Performance or Accuracy:

```bash
MLPERF_TEST_MODE=PerformanceOnly sbatch \
  --account="${MLPERF_SLURM_ACCOUNT}" \
  --partition="${MLPERF_SLURM_PARTITION}" \
  "${PROFILE}/launchers/run_x64_submission.sh"
```

Use `MLPERF_TEST_MODE=AccuracyOnly` for Accuracy. For TEST06 set
`MLPERF_TEST_MODE=PerformanceOnly` and
`MLPERF_HARNESS_EXTRA_ARGS=--audit_test=TEST06`.
