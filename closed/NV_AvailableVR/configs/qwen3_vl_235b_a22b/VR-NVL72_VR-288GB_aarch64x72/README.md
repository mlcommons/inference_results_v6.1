# Qwen3-VL-235B-A22B on NVIDIA Vera Rubin NVL72

vLLM + NVIDIA Dynamo recipes for Qwen3-VL-235B-A22B on Vera Rubin NVL72 (18 nodes × 4 GPUs), one folder per scenario.
Each README has the launch command, the configuration and the expected result:

- [VLLM/Offline](VLLM/Offline/README.md): 72 aggregated TP=1 workers
- [VLLM/Server](VLLM/Server/README.md): 72 aggregated TP=1 workers
- [VLLM/Interactive](VLLM/Interactive/README.md): prefill/decode disaggregation, 48 prefill + 24 decode TP=1 workers

The server image is built from public sources by
[src/nv_mlpinf/benchmarks/q3vl/rubin](../../../src/nv_mlpinf/benchmarks/q3vl/rubin/README.md).
