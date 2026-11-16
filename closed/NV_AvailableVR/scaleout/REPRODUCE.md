# MLPerf Inference Scaleout - nv-sflow Reproduction Commands

This branch contains the DeepSeek-R1 Vera Rubin NVL72 submission (`VR-NVL72_VR-288GB_aarch64x72`), run with
TensorRT-LLM + NVIDIA Dynamo disaggregated serving and the MLPerf Endpoints client.

The model, data and image preparation and the `sflow batch` command for each scenario (Offline, Server, Interactive)
are in the [DeepSeek-R1 README](../src/nv_mlpinf/benchmarks/deepseek_r1/README.md).

The branch also holds the Qwen3-VL-235B-A22B Vera Rubin NVL72 recipes, run with vLLM + NVIDIA Dynamo. Their
`sflow batch` commands are in the scenario READMEs listed in
[configs/qwen3_vl_235b_a22b/VR-NVL72_VR-288GB_aarch64x72](../configs/qwen3_vl_235b_a22b/VR-NVL72_VR-288GB_aarch64x72/README.md).

## Reference

- [nv-sflow Documentation](sflow/README.md) - Detailed nv-sflow scaleout documentation
- [../docs/ENV_SETUP.md](../docs/ENV_SETUP.md) - Environment setup guide
- [../configs/SLURM_SUPPORT.md](../configs/SLURM_SUPPORT.md) - SLURM config support matrix
