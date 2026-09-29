# DeepSeek-R1 - 512xMI355X Offline (perf)

512-GPU offline-scenario run: 64 nodes x 8 AMD Instinct MI355X (288 GB HBM3E) = 512 GPUs on Crusoe
Managed Kubernetes. AMD pre-quantized DeepSeek-R1 (`amd/Deepseek-S3_sq_a05_v2_mlperf6_1`: OCP MXFP4
experts / BF16 MLA-absorb / FP8-e4m3 KV) served with SGLang 0.5.15.post1 (ROCm 7.2.0). The 671B MoE does
not fit one GPU, so each node runs ONE replica sharded across all 8 GPUs (TP8 + EP8 + 8-way DP-attention,
intra-node over XGMI) -> 64 replicas total. LoadGen sees one SUT: an AMD ZMQ distributed harness with
1 dedicated head (dispatch + LoadGen, no model) + 64 worker nodes, `device_count = 512`. No cross-node
collective; only tokenized I/O streams over 8x400 Gb RoCE Ethernet. The same manifests run at N=1 (8 GPU)
and N=8 (64 GPU) for validation before the full 64-node run.

## Launch

Config in this submission: `closed/Crusoe/src/deepseek-r1/offline_mi355x_mn.yaml` + `user_mi355x_mn.conf`
+ the tuning files `bf16_tuned_gemm_withhip.csv` and `hipblaslt.txt` (plus this directory's `mlperf.conf`).
Harness: the ZMQ distributed SUT at `closed/Crusoe/src/harness_llm/backends/vllm/zmq/README.md` (it is
engine-agnostic and picks SGLang from the `sglang_engine_config` in the yaml) - run `run_harness.sh
--config-name offline_mi355x_mn --backend zmq` as a ZMQ head (`device_count=512`) plus one worker per node.

Kubernetes-native (as submitted). Full manifests and launchers:
`https://github.com/martin-cala1/crusoe-mlperf-mi355x-inference-v6.1`

```bash
# One-time prereqs (see the repo README): create the namespace + shared PVC and an HF-token secret;
# build/push the deepseek image from AMD's public v6.1 code; download the model (~366 GB) + dataset;
# and stage the harden-*.py ZMQ patches to /shared/patches-ds/. Substitute <your-registry>/<your-project>.
cd k8s/deepseek-r1
DEDICATED_HEAD=1 SCENARIO=offline bash launch-mn-deepseek.sh 64   # device_count=512, target_qps 760
```

The launcher selects a free MI355X node for the head, reads its internal IP, templates the head + worker
manifests, and applies them. Results land in `/shared/results/<run>`.

## Results

Offline, VALID (see `performance/run_1/mlperf_log_summary.txt`):
- Throughput: 2,901,950 tokens/s (735.23 samples/s), `target_qps 760`.
- Accuracy (one AccuracyOnly run per model, `accuracy/`): exact_match 80.70, tokens_per_sample 3898.37
  (gates: exact_match >= 80.5446 AND tokens_per_sample in [3497.6, 4274.85]) - PASS.
- Compliance: TEST06 PASS (`TEST06/`).

## Accuracy & compliance

Accuracy and compliance are scale-independent (each sample is issued/scored exactly once), so they run
with a `DEVICE_COUNT=496` (= 512 - 16, a 2-node tolerance) registration barrier so a rare per-replica
engine-startup flake does not force a relaunch. From `k8s/deepseek-r1`:

```bash
MODE=accuracy DEDICATED_HEAD=1 SCENARIO=offline bash launch-mn-deepseek.sh 64   # exact_match ~80.70
DEDICATED_HEAD=1 SCENARIO=offline TEST06=1 bash launch-mn-deepseek.sh 64        # TEST06 compliance
```

Tunables live in `offline_mi355x_mn.yaml` (TP8/EP8/DP-attention, MoRI-EP, decode CUDA-graph batch list,
prefill-delayer) and `user_mi355x_mn.conf` (`target_qps`, `min_duration`).
