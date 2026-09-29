# GPT-OSS-120B x16 Offline IFB

This profile uses 16 TP1 endpoints and target QPS 162. It supports Performance,
Accuracy, TEST07, and TEST09.

```bash
export PROFILE="${CISCO_SUBMISSION_ROOT}/configs/gpt_oss_120b/Cisco_UCS_B300x16_G200_4x2_RO/TRTLLM/Offline"
export MAIN_CONFIG=gptoss_config_sflow.yaml
export WORKFLOW=trtllm_ifb_portable.yaml
export MLPERF_OUTPUT_DIR=/path/to/output/gptoss-x16-offline
```

Execute the standard x16 command in `../../../../README.md`. For TEST07 or
TEST09 keep `MLPERF_TEST_MODE=PerformanceOnly` and set
`MLPERF_HARNESS_EXTRA_ARGS=--audit_test=TEST07` or `--audit_test=TEST09`.
