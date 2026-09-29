# DeepSeek-R1 — VR200 NVL72

Serving configurations for the three MLPerf Inference scenarios — Offline, Server and
Interactive — one directory per scenario.

## Common to all three

Disaggregated prefill/decode serving of `DeepSeek-R1-NVFP4-v2-mlpinf` on 72 GPUs, greedy decoding
(`temperature 0.0`, `top_k 1`, `top_p 1.0`), `max_new_tokens 20000`, and the official MLPerf seeds
(scheduler `16159082839903944936`, dataloader `2747215439041700203`). Prefix-cache reuse is
disabled so repeated prompts cannot be served from cache.

The scenarios differ in load pattern — Offline issues at maximum throughput with streaming off;
Server and Interactive issue Poisson-distributed arrivals with streaming on, at their respective
target rates — and in the server-side parameters tuned for each latency target. Only the load
pattern is recorded here.

## Directory contents

Each scenario directory holds a single `client.yaml`: the load pattern, run duration, sample
count, dataset, sampling parameters, random seeds and accuracy configuration for that scenario.

Server-side engine, environment, deployment and scheduler configuration is not included here.
`<MODEL_ROOT>` is a placeholder for the location of the model weights.

## Running

`client.yaml` is the benchmark client configuration, consumed by `inference-endpoint benchmark
from-config`. Reproducing a measurement also requires a server-side deployment, which is not
described here; see [`../../../sflow/README.md`](../../../sflow/README.md) for how the harness is
driven.
