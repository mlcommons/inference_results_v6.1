# SLURM Support Matrix

This branch holds the DeepSeek-R1 Vera Rubin NVL72 submission and the Qwen3-VL-235B-A22B Vera Rubin NVL72 recipes.
Run them from a SLURM login node on a cluster with pyxis/enroot and `--segment` support. Setup and launch commands
are in the [DeepSeek-R1 README](../src/nv_mlpinf/benchmarks/deepseek_r1/README.md) and in the
[Qwen3-VL scenario READMEs](qwen3_vl_235b_a22b/VR-NVL72_VR-288GB_aarch64x72/README.md).

## deepseek_r1 (3 configs)

| System                         | Scenario    | SLURM                                                  |
| ------------------------------ | ----------- | ------------------------------------------------------ |
| `VR-NVL72_VR-288GB_aarch64x72` | Offline     | ✅ (Dynamo disagg, one NVL72 rack, 18 nodes `--segment=18`) |
| `VR-NVL72_VR-288GB_aarch64x72` | Server      | ✅ (Dynamo disagg, one NVL72 rack, 18 nodes `--segment=18`) |
| `VR-NVL72_VR-288GB_aarch64x72` | Interactive | ✅ (Dynamo disagg, one NVL72 rack, 18 nodes `--segment=18`) |

## qwen3_vl_235b_a22b (3 configs)

| System                         | Scenario    | SLURM                                                                   |
| ------------------------------ | ----------- | ----------------------------------------------------------------------- |
| `VR-NVL72_VR-288GB_aarch64x72` | Offline     | ✅ (Dynamo + vLLM, 72 aggregated workers, 18 nodes `--segment=1`)        |
| `VR-NVL72_VR-288GB_aarch64x72` | Server      | ✅ (Dynamo + vLLM, 72 aggregated workers, 18 nodes `--segment=1`)        |
| `VR-NVL72_VR-288GB_aarch64x72` | Interactive | ✅ (Dynamo P/D disagg + vLLM, 48 prefill + 24 decode, 18 nodes `--segment=1`) |
