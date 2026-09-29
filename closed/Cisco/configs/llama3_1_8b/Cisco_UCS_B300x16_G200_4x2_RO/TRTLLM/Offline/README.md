# Llama3.1-8B x16 Offline IFB

This profile uses 16 TP1 endpoints and target QPS 2800. It supports
Performance, Accuracy, and TEST06.

```bash
export PROFILE="${CISCO_SUBMISSION_ROOT}/configs/llama3_1_8b/Cisco_UCS_B300x16_G200_4x2_RO/TRTLLM/Offline"
export MAIN_CONFIG=llama3_1_8b_config_sflow.yaml
export WORKFLOW=trtllm_ifb_portable.yaml
export MLPERF_OUTPUT_DIR=/path/to/output/llama3-x16-offline
```

Execute the standard x16 command in `../../../../README.md`. For TEST06 set
`MLPERF_TEST_MODE=PerformanceOnly` and
`MLPERF_HARNESS_EXTRA_ARGS=--audit_test=TEST06`.
