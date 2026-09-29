# GPT-OSS-120b Server - closed division

- System: 9x8xMI355X_2xEPYC_9575F (9 nodes x 8x AMD Instinct MI355X = 72 GPUs)
- Backend: distributed asynchronous ZMQ (vLLM), MXFP4 weights + FP8 KV cache
- Scenario: Server
- Compliance: TEST07 (gpqa accuracy) and TEST09 (output-token-length) PASS

Run via run_harness.sh with the ZMQ head (device_count=72) plus one distributed
worker per node; see src/ for the harness implementation and configs.
