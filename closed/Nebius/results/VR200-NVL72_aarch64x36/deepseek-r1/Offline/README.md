# DeepSeek-R1 Offline - VR200-NVL72_aarch64x36

Results collected with the MLPerf Endpoints client (mlcommons/endpoints @598dbfde) against
NVIDIA Dynamo + TensorRT-LLM disaggregated serving on a Vera Rubin NVL72 half rack
(9 nodes x 4 GPUs = 36 GPUs, one NVLink segment).

Setup, model/data preparation and launch instructions:
`closed/Nebius/src/nv_mlpinf/benchmarks/deepseek_r1/README.md`.

Run configuration used for this result:
- `closed/Nebius/configs/deepseek_r1/VR-NVL72_VR-288GB_aarch64x36/Offline/` (sflow point config, Slurm/container
  settings, TRT-LLM ctx/gen engine configs and env overlays, endpoint client config)
- `closed/Nebius/scaleout/sflow/templates/dynamo_disagg_half_rack9_endpoints_dsr1_perf_acc.yaml` (launch template)

Layout: 2 context servers x 2 GPUs (DEP2) + 4 generation servers x 8 GPUs (DEP8), 8 Dynamo frontends.
