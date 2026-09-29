# DeepSeek-R1 x16 Server PD

This profile uses two DEP2 context workers, three DEP4 generation workers, two
frontends, and target QPS 24. It supports Performance, Accuracy, and TEST06.

Select the profile:

```bash
export PROFILE="${CISCO_SUBMISSION_ROOT}/configs/deepseek_r1/Cisco_UCS_B300x16_G200_4x2_RO/TRTLLM/Server"
export MAIN_CONFIG=deepseek_config_sflow-2ctx-3gen-dep4.yaml
export WORKFLOW=trtllm_disagg_portable.yaml
export MLPERF_OUTPUT_DIR=/path/to/output/deepseek-x16-server
```

Execute the standard x16 command in `../../../../README.md`. For TEST06 use
`--audit_test=TEST06 --server_target_qps_adj_factor=0.92`.
