# DeepSeek-R1 x16 Offline IFB

This profile uses two DEP8 IFB endpoints and target QPS 38. It supports
Performance, Accuracy, and TEST06.

```bash
export PROFILE="${CISCO_SUBMISSION_ROOT}/configs/deepseek_r1/Cisco_UCS_B300x16_G200_4x2_RO/TRTLLM/Offline"
export MAIN_CONFIG=deepseek_config_sflow.yaml
export WORKFLOW=trtllm_ifb_portable.yaml
export MLPERF_OUTPUT_DIR=/path/to/output/deepseek-x16-offline
```

Execute the standard x16 command in `../../../../README.md`. For TEST06 set
`MLPERF_TEST_MODE=PerformanceOnly` and
`MLPERF_HARNESS_EXTRA_ARGS=--audit_test=TEST06`.
