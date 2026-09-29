# gpt-oss-120b - 512xMI355X Server (perf)

512-GPU server-scenario run under a p99 latency SLA: 64 nodes x 8 AMD Instinct MI355X (288 GB HBM3E) =
512 GPUs on Crusoe Managed Kubernetes. Native MXFP4 `openai/gpt-oss-120b` (checkpoint `b5c939d`, FP8-e4m3
KV cache) served with vLLM 0.22.1 + AITER (ROCm 7.2.2). The model fits in one GPU at MXFP4, so each node
runs 8 independent single-GPU replicas (tensor-parallel size 1) -> 512 replicas total. LoadGen sees one
SUT: an AMD ZMQ distributed harness with 1 dedicated head (dispatch + LoadGen, no model) + 64 worker
nodes, `device_count = 512`. No cross-node collective; only tokenized I/O streams over 8x400 Gb RoCE
Ethernet. The same manifests run at N=1 (8 GPU) and N=8 (64 GPU) for validation before the full run.

## Launch

Config in this submission: `closed/Crusoe/src/gpt-oss-120b/server_mi355x_mn.yaml` + `user_mi355x_mn.conf`
(plus this directory's `mlperf.conf`). Harness: the ZMQ distributed SUT documented at
`closed/Crusoe/src/harness_llm/backends/vllm/zmq/README.md` - run `run_harness.sh --config-name
server_mi355x_mn --backend zmq` as a ZMQ head (`device_count=512`) plus one worker per node.

Kubernetes-native (as submitted). Full manifests and launchers:
`https://github.com/martin-cala1/crusoe-mlperf-mi355x-inference-v6.1`

```bash
# One-time prereqs (see the repo README): create the namespace + shared PVC and an HF-token secret;
# build/push the gpt-oss image from AMD's public v6.1 code; download the model + dataset; and stage the
# harden-*.py ZMQ patches to /shared/patches/. Substitute <your-registry>/<your-project> in the manifests.
cd k8s/gpt-oss
DEDICATED_HEAD=1 bash launch-mn.sh 64 "" 4000      # target_qps MUST be 4000 for a VALID Server run
```

`target_qps` is tuned to the highest value that still passes the SLA; pushing past 4000 creates a
scheduling backlog and blows p99. The launcher templates the head + worker manifests and applies them.
Results land in `/shared/results/<run>`.

## Results

Server, VALID (see `performance/run_1/mlperf_log_summary.txt`):
- Throughput: 5,392,930 tokens/s (3956.29 completed samples/s) at `target_qps 4000`.
- p99 latency (SLA limit): TTFT 2.69 s (< 3.0 s); TPOT 36.5 ms (< 80 ms).
- Accuracy (one AccuracyOnly run per model, `accuracy/`): 83.09% exact-match (3652/4395), above the 82.30
  closed-division floor.
- Compliance: TEST07 + TEST09 PASS (`TEST07/`, `TEST09/`).

## Accuracy & compliance

Accuracy and compliance are scale-independent (each sample is issued/scored exactly once), so they run
with a `DEVICE_COUNT=496` (= 512 - 16, a 2-node tolerance) registration barrier so a rare per-replica
engine-startup flake does not force a relaunch. From `k8s/gpt-oss`:

```bash
DEDICATED_HEAD=1 DEVICE_COUNT=496 bash mn-accuracy-offline-64.sh    # exact-match ~83.1 (shared run)
DEVICE_COUNT=496 bash mn-compliance-64.sh TEST07                    # gpqa accuracy-in-performance
DEVICE_COUNT=496 bash mn-compliance-64.sh TEST09                    # output-token-length
```

Tunables live in `server_mi355x_mn.yaml` (batch / scheduler) and `user_mi355x_mn.conf` (`target_qps`,
`min_duration`). Engine knobs such as `enforce_eager` are set in AMD's upstream harness config, not in
this orchestration layer.
