## MangoBoost MLPerf Inference v6.1 Submission
---

In MLPerf Inference v6.1 Submission, MangoBoost submits two datapoint:
1. **Prefill Decode (P/D) disaggregated 2-node MI355X**: For the interactive scenario, we deploy a P/D setup and transfer the KV cache using MoriIO. The result meets the GPT-OSS interactive requirement of P99 TTFT < 2s, and P99 TPOT < 15ms.

2. **Single Node - MI300X GPT-OSS**: We quantize the GPT-OSS-120B into fp8 so that it will have a better performance on MI300X. Both Offline and Server results meet MLPerf requirement on GPT-OSS-120B.

**Accuracy**: All the Acuccracy also pass the GPT-OSS-120B threshold > 82.30% (99% of 83.13%). Compliance TEST07 and TEST09 also pass.

---

### Reproduction

- **Results of 2-node MI355X GPT-OSS-120B**: refer to `../results/16xMI355X_2xEPYC_9575F/gpt-oss-120b/Interactive/README.md` and `../results/16xMI355X_2xEPYC_9575F/gpt-oss-120b/Offline/README.md`.

- **Results of MI300X GPT-OSS-120B**: refer to `../results/8xMI300X_2xEPYC_9534/gpt-oss-120b/Offline/README.md` and `../results/8xMI300X_2xEPYC_9534/gpt-oss-120b/Server/README.md`.