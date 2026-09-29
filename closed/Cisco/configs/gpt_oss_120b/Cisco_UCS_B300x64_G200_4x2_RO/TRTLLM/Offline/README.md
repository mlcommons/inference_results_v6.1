# GPT-OSS-120B x64 Offline IFB

This implementation runs 64 independent TP1/EP1 TensorRT-LLM endpoints and
uses the MLPerf Offline harness at target QPS 635. It supports Performance,
Accuracy, TEST07, and TEST09.

Site-specific filesystem, scheduler, host, and network defaults are exposed as
launcher environment variables.

Set the required `MLPERF_*` variables documented by
`launchers/run_x64_offline_ifb.sh`, then launch the checked-in profile:

```bash
export PROFILE="${CISCO_SUBMISSION_ROOT}/configs/gpt_oss_120b/Cisco_UCS_B300x64_G200_4x2_RO/TRTLLM/Offline"
export MLPERF_OUTPUT_DIR=/path/to/output/gptoss-x64-offline
export MLPERF_SERVER_HOSTS=host01,host01,host01,host01,host01,host01,host01,host01,host02,host02,host02,host02,host02,host02,host02,host02,host03,host03,host03,host03,host03,host03,host03,host03,host04,host04,host04,host04,host04,host04,host04,host04,host05,host05,host05,host05,host05,host05,host05,host05,host06,host06,host06,host06,host06,host06,host06,host06,host07,host07,host07,host07,host07,host07,host07,host07,host08,host08,host08,host08,host08,host08,host08,host08

MLPERF_TEST_MODE=PerformanceOnly sbatch \
  --account="${MLPERF_SLURM_ACCOUNT}" \
  --partition="${MLPERF_SLURM_PARTITION}" \
  "${PROFILE}/launchers/run_x64_offline_ifb.sh"
```

Choose `AccuracyOnly`, `TEST07`, or `TEST09` with `MLPERF_TEST_MODE`; the
launcher preserves QPS 635 and selects the matching official harness target.
