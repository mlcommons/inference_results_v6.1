# Llama3.1-8B x16 Server PD

This profile uses six context workers, ten generation workers, two frontends,
and target QPS 740. It supports Performance, Accuracy, and TEST06.

```bash
export PROFILE="${CISCO_SUBMISSION_ROOT}/configs/llama3_1_8b/Cisco_UCS_B300x16_G200_4x2_RO/TRTLLM/Server"
export MAIN_CONFIG=llama3_1_8b_config_sflow.yaml
export WORKFLOW=trtllm_disagg_portable.yaml
export MLPERF_OUTPUT_DIR=/path/to/output/llama3-x16-server
```

Execute the standard x16 command in `../../../../README.md`. For TEST06 use
`--audit_test=TEST06 --server_target_qps_adj_factor=0.92`.
