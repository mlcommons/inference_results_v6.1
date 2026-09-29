# AMD Llama-3.1-8B (Interactive) - MLPerf Inference v6.1

System: 8x AMD Instinct MI355X + 2x AMD EPYC 9575F
Backend: vLLM 0.22.0 (ROCm 7.2.2), MXFP4 (W4A4) weights + FP8 KV cache (Quark).

## Reproduce

Inside the vLLM container (`/lab-mlperf-inference`):

```bash
# Performance
python3 code/main.py --config-path code/llama3.1-8b \
    --config-name interactive_mi355x --backend vllm \
    test_mode=performance \
    harness_config.output_log_dir=results/llama3.1-8b/Interactive/performance/run_1

# Accuracy
python3 code/main.py --config-path code/llama3.1-8b \
    --config-name interactive_mi355x --backend vllm \
    test_mode=accuracy \
    harness_config.output_log_dir=results/llama3.1-8b/Interactive/accuracy
bash code/scripts/llama3.1-8b/run_accuracy.sh \
    results/llama3.1-8b/Interactive/accuracy/mlperf_log_accuracy.json > accuracy.txt

# Compliance TEST06 (audit.config performance run + verification)
bash submission/run_test06.sh Interactive
```
