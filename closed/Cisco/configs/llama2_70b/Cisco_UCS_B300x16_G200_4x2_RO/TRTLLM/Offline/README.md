# Llama2-70B x16 Offline IFB

This profile uses 16 TP1 IFB endpoints and target QPS 900. It supports
Performance, Accuracy, and TEST06.

After the common setup in `../../../../README.md`, select:

```bash
export PROFILE="${CISCO_SUBMISSION_ROOT}/configs/llama2_70b/Cisco_UCS_B300x16_G200_4x2_RO/TRTLLM/Offline"
export MAIN_CONFIG=llama2_config_sflow.yaml
export WORKFLOW=trtllm_ifb_portable.yaml
export MLPERF_OUTPUT_DIR=/path/to/output/llama2-x16-offline
```

Then execute the standard x16 command in `../../../../README.md`. For TEST06,
set `MLPERF_TEST_MODE=PerformanceOnly` and
`MLPERF_HARNESS_EXTRA_ARGS=--audit_test=TEST06`.
