# Llama2-70B x64 Server PD

This profile uses 24 context workers, 40 generation workers, eight frontends,
and target QPS 885. It supports Performance, Accuracy, and TEST06.

```bash
export PROFILE="${CISCO_SUBMISSION_ROOT}/configs/llama2_70b/Cisco_UCS_B300x64_G200_4x2_RO/TRTLLM/Server"
export MLPERF_OUTPUT_DIR=/path/to/output/llama2-x64-server
export MLPERF_CTX_HOSTS=ctx01,ctx01,ctx01,ctx01,ctx01,ctx01,ctx01,ctx01,ctx02,ctx02,ctx02,ctx02,ctx02,ctx02,ctx02,ctx02,ctx03,ctx03,ctx03,ctx03,ctx03,ctx03,ctx03,ctx03
export MLPERF_GEN_HOSTS=gen01,gen01,gen01,gen01,gen01,gen01,gen01,gen01,gen02,gen02,gen02,gen02,gen02,gen02,gen02,gen02,gen03,gen03,gen03,gen03,gen03,gen03,gen03,gen03,gen04,gen04,gen04,gen04,gen04,gen04,gen04,gen04,gen05,gen05,gen05,gen05,gen05,gen05,gen05,gen05
export MLPERF_FRONTEND_HOSTS=frontend01,frontend02,frontend03,frontend04,frontend05,frontend06,frontend07,frontend08

MLPERF_TEST_MODE=PerformanceOnly sbatch \
  --account="${MLPERF_SLURM_ACCOUNT}" \
  --partition="${MLPERF_SLURM_PARTITION}" \
  "${PROFILE}/launchers/run_x64_submission.sh"
```

Use `MLPERF_TEST_MODE=AccuracyOnly` for Accuracy. For TEST06 set
`MLPERF_TEST_MODE=PerformanceOnly` and
`MLPERF_HARNESS_EXTRA_ARGS=--audit_test=TEST06`.
