# GPT-OSS-120b - AMD MLPerf Inference v6.1 (72-GPU distributed ZMQ)

Distributed asynchronous ZMQ backend running vLLM across 9 nodes x 8 MI355X (72 GPUs).

- Head (SUT): run_harness.sh ... --backend zmq harness_config.device_count=72
- Worker per node: harness_llm/backends/vllm/zmq/distributed_async_server.py (Server)
  and distributed_sync_offline.py (Offline)
- Configs: gpt-oss-120b/{offline_mi355x_mn.yaml, server_mi355x_mn.yaml, user_mi355x_mn_72.conf}
- Quantization: MXFP4 (W4A4) weights, FP8 KV cache (Quark)
