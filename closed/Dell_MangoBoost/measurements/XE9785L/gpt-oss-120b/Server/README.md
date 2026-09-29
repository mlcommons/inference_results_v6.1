## Dell_MangoBoost MLPerf Inference v6.1 Submission
---

In MLPerf Inference v6.1 Submission, Dell_MangoBoost submits three datapoint:
1. **Multi Region 4-Cluster Inference**: We connect four clusters in different regions to make a scalable inference, which is comprised of MI300X from MangoBoost cluster (in Korea), MI355X from Dell cluster, MI355X from TensorWave cluster and MI300X Azure cluster. The scaling achieves **97%**. 

2. **Intra-Node P/D - MI355X GPT-OSS with liquid-cooling**: We submit single node MI355X on XE9785L, which is a liquid cooling MI355X node. To make a performant goodput, we applied ***Intra-node P/D disaggregation*** optimization in Interactive Scenario, with 4-GPUs dedicating working on prefill, plus 4-GPUs dedicating working on decode, connected with XGMI KV connector. The result shows great goodput on interactive Scenario, which achieves **29K** throughput and at the same time maintain P99 TTFT < 2s and P99 TPOT < 15ms.

3. **Single Node - MI355X GPT-OSS with air-cooling**: We submit single node MI355X on XE9785, which is a air cooling MI355X node. Same as XE9785L, we applied ***Intra-node P/D disaggregation*** optimization in Interactive Scenario, with 4-GPUs dedicating working on prefill, plus 4-GPUs dedicating working on decode, connected with XGMI KV connector. The result shows great goodput on interactive Scenario, which achieves **28K** throughput and at the same time maintain P99 TTFT < 2s and P99 TPOT < 15ms.

**Performance requirement** All the result meets the GPT-OSS Offline and Server requirement of P99 TTFT < 3s, and P99 TPOT < 80ms.

**Accuracy**: All the Acuccracy also pass the GPT-OSS-120B threshold > 82.30% (99% of 83.13%). Compliance TEST07 and TEST09 also pass.

***More reproduction details will be soon added.***