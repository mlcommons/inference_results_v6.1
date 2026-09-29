# GPT-OSS-120B x16 Server IFB

This profile uses 16 TP1 endpoints and target QPS 170. It supports Performance,
Accuracy, TEST07, and TEST09.

```bash
export PROFILE="${CISCO_SUBMISSION_ROOT}/configs/gpt_oss_120b/Cisco_UCS_B300x16_G200_4x2_RO/TRTLLM/Server"
export MAIN_CONFIG=gptoss_config_sflow.yaml
export WORKFLOW=trtllm_ifb_portable.yaml
export MLPERF_OUTPUT_DIR=/path/to/output/gptoss-x16-server
export MLPERF_HARNESS_EXTRA_ARGS=""
```

For TEST07 or TEST09, set `MLPERF_HARNESS_EXTRA_ARGS` to
`--audit_test=TEST07` or `--audit_test=TEST09`. Accuracy uses
`MLPERF_TEST_MODE=AccuracyOnly`; all other modes use `PerformanceOnly`.

Execute the standard x16 command in `../../../../README.md`.
