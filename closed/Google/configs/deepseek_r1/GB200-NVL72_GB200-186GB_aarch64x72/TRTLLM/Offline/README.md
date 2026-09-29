# DeepSeek-R1 Offline benchmark on GB200 NVL72

This directory defines the NVIDIA TensorRT-LLM configuration used to run the
MLPerf Inference DeepSeek-R1 Offline benchmark on a full GB200 NVL72 system.
The final Offline-plus-Server results and diagnostic history are collected in
[`docs/DP_R1_BENCHMARK_SUMMARY.md`](../../../../../docs/DP_R1_BENCHMARK_SUMMARY.md).
It explains the execution flow, MLPerf LoadGen, the scale-out topology, and the
settings in:

- `trtllm-serve-ifb-dep8.yaml`
- `trtllm-serve-ifb-dep8-env.yaml`
- `harness.py`
- `deepseek_config_sflow.yaml`
- `slurm_env_sflow.yaml`
- `scaleout/sflow/templates/trtllm_ifb_loadgen.yaml`

The values are a tuned configuration set. A setting that is beneficial in this
combination is not necessarily beneficial when changed independently or moved
to another GPU, model, scenario, or TensorRT-LLM release.

## Job topology

The full system contains 72 GPUs across 18 nodes, with 4 GPUs per node.
DeepSeek-R1 is served by nine independent data-parallel replicas:

```text
18 Slurm nodes × 4 GPUs/node = 72 GPUs
9 server replicas × 8 GPUs/replica = 72 GPUs
1 server replica = 2 nodes = 8 GPUs
```

Within each eight-GPU replica:

```text
tensor parallel size       = 8
pipeline parallel size     = 1
MoE expert parallel size   = 8
attention data parallelism = enabled
```

The filename suffix `dep8` means MoE expert parallelism of eight with attention
data parallelism enabled. It is not the number of Slurm replicas.

## End-to-end execution flow

```text
sbatch allocation: 18 nodes / 72 GPUs
                 |
                 v
nv-sflow creates nine parallel server tasks
                 |
                 v
each task starts 8 ranks with PMI-2 and trtllm-serve
                 |
                 v
nine HTTP endpoint URLs are recorded in server_urls.txt
                 |
                 v
sflow waits for every server readiness probe
                 |
                 v
nv-mlpinf starts one LoadGen harness against all endpoints
                 |
                 v
LoadGen issues Offline samples; the asynchronous client sends requests
                 |
                 v
TensorRT-LLM performs prefill + decode with in-flight batching
                 |
                 v
responses are returned to LoadGen and performance logs are written
```

The server tasks start first. The harness task depends on all server replicas
and cannot begin until each server log contains `Application startup complete`.
The readiness timeout is 600 seconds and is checked every 10 seconds.

## What MLPerf LoadGen does

MLPerf LoadGen is the benchmark controller. It provides a consistent workload
and measurement procedure across implementations. It does not run the model.
The NVIDIA harness connects LoadGen to the TensorRT-LLM HTTP endpoints.

The important interfaces are:

- **QSL (Query Sample Library):** owns the benchmark samples. Here, the samples
  are read from `input_ids_padded.npy` and `input_lens.npy` under
  `preprocessed_data/deepseek-r1/`.
- **SUT (System Under Test):** receives sample IDs from LoadGen, sends the
  corresponding tokenized prompts to the server endpoints, and reports each
  completed response back to LoadGen.
- **LoadGen scheduler:** selects sample order, constructs the Offline workload,
  enforces duration/query constraints, timestamps completions, and writes the
  standard MLPerf logs.

In the Offline scenario, all work is assumed to be available at once. There is
no simulated online arrival process and no Server-scenario TTFT/TPOT latency
constraint. The harness can keep many requests in flight to maximize aggregate
throughput. LoadGen still requires every issued sample to complete correctly and
reports the effective samples-per-second result.

`PerformanceOnly` means LoadGen measures performance without producing the full
accuracy log or running the DeepSeek exact-match accuracy evaluation. A separate
`AccuracyOnly` run is required for a complete MLPerf result.

### LoadGen settings used here

`harness.py` supplies these values:

| Setting | Value | Purpose |
| --- | ---: | --- |
| `min_duration` | `600000` ms | Run for at least 10 minutes. |
| `min_query_count` | `70208 * 8 = 561664` | Ensure enough issued samples for a stable full-system measurement. Samples may repeat from the performance set. |
| `offline_expected_qps` | `12 * 9 = 108` | Expected aggregate Offline samples/s across nine replicas; used to size/configure the LoadGen run. |
| `max_concurrency` | `5120` | Bound outstanding asynchronous client requests and avoid unbounded host/network queues. |
| `use_token_latencies` | `true` | Collect token-level timing information needed by LLM reporting. |
| `warmup_iterations` | `0` | Do not add harness-controlled warmup iterations; server startup and the configured workflow establish readiness. |
| `precision` | `fp4` | Select the FP4 model/configuration. |
| `input_dtype` | `int32` | Match the preprocessed token-ID arrays. |

`offline_expected_qps` is a sizing expectation, not proof that the system will
achieve 108 samples/s. The measured LoadGen result is authoritative. The large
minimum query count can make the run substantially longer than the 10-minute
minimum if achieved throughput is below the configured expectation.

The effective performance set contains 4,388 samples, and the configured
561,664 sample count repeats that set 128 times. At exactly 108 samples/s, query
drain alone takes about 86.7 minutes, so the Slurm wall time must include model
startup and result finalization as well.

## Scale-out configuration

`deepseek_config_sflow.yaml` defines:

| Variable | Value | Meaning |
| --- | ---: | --- |
| `DP_MULTIPLICITY` | 9 | Number of independent TensorRT-LLM server replicas. |
| `GPUS_PER_DP_RANK` | 8 | GPU allocation for each replica. |
| `TOTAL_GPUS` | 72 | Product of replicas and GPUs per replica. |
| `BASE_PORT` | 8336 | Replica `i` listens on port `8336 + i`. |
| `SCENARIO` | `Offline` | Select the Offline LoadGen behavior and harness config. |
| `TEST_MODE` | `PerformanceOnly` | Measure performance without accuracy evaluation. |

`slurm_env_sflow.yaml` maps the workflow onto Slurm:

- 4 GPUs per node, so 72 GPUs require 18 nodes.
- The allocation is exclusive.
- The batch allocation requests `--gpus-per-node=4`, and each of the four
  TensorRT-LLM ranks per node requests `--gpus-per-task=1`. Both levels are
  required for Slurm to expose a GPU to every rank.
- The one-rank LoadGen harness also requests one GPU because nvmitten performs
  CUDA system detection while selecting the runtime system configuration.
- The container is read-only, mounts the repository at `/work`, and remaps the
  container user to root.
- The a4x run uses the pinned image from the shared
  `/lustre/alisachen/containers/...aarch64.sqsh` file. Direct Pyxis imports
  require `nvcr.io#nvidia/...`, but nv-sflow 0.2.1 rejects that syntax; the
  shared squashfs is accepted by both layers.
- `mpi: pmi2` is used on the a4x domain. It was validated across two nodes with
  container-side `mpi4py`; direct PMIx failed because the host and container
  carry incompatible PMIx generations.
- The pinned aarch64 MLPerf TensorRT-LLM image supplies the runtime.

Long full-cluster runs also need enough Slurmd connection capacity. Server
attempt 170537 reached the configured query count but was invalidated before
finalization when nodes 0 and 1 repeatedly hit Slurm 25.11.2's compiled
50-connection default (`51/50 connections`), stopped answering controller
heartbeats, and caused `NODE_FAIL`. This was infrastructure failure, not a
LoadGen result, and its partial output must not be submitted or archived as a
result. The cluster-level mitigation applied before retrying was:

```text
SlurmdParameters=conmgr_max_connections=256
```

Place the setting in the administrator-managed portion of `slurm.conf`, run
`sudo scontrol reconfigure`, and restart Slurmd on every idle compute node so
the configless cache is reloaded. Clean only the failed job's exact step
cgroups/processes before restarting; Slurmd's systemd unit intentionally leaves
step daemons alive across an ordinary daemon restart. Before another measured
run, require all 18 nodes to be `IDLE`, verify the cached parameter, and run a
full 18-node/72-rank PMI-2 smoke. Recovery smoke job 170538 completed `0:0` and
showed no new connection-ceiling warnings.

For the a4x run, the generated job overrides the default mounts so shared model
and raw data remain on shared Lustre while user-owned preprocessed data is
overlaid at `/home/mlperf_inference_storage/preprocessed_data`. Because an
override replaces the complete mount list, it must retain all three mappings:

```text
/home/alisachen_google_com/nv-mlpinf-partner/closed/NVIDIA:/work
/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone:/home/mlperf_inference_storage
/lustre/alisachen/mlperf_inference_storage/preprocessed_data:/home/mlperf_inference_storage/preprocessed_data
```

The server command enables shell `pipefail` before piping TensorRT-LLM output
through `tee`, and the harness exits with the captured `run_harness` status.
Engine and LoadGen failures therefore remain failed workflow steps instead of
being masked by successful log writes. nv-sflow 0.2.1's generated outer batch
wrapper also needs its return-code fix so later artifact copies cannot turn the
parent job green:

```bash
python3 scripts/slurm_llm/deepseek_r1/fix_sflow_batch_exit.py \
  build/sbatch_scripts_sflow/<generated-job>.sh
```

Run this immediately after `sflow batch -o` and before `bash -n` or `sbatch`.
Generate the wrapper with
`--sflow-version=6efbf59c4a835473b08362eb623358e3ace31d6e`, the exact nv-sflow
0.2.1 source revision validated by job 170528, rather than letting compute nodes
resolve the moving `main` branch.

The workflow also runs `repo_preflight` before any server. It verifies that
`3rdparty/mlc-inference/tools/submission/submission_checker/constants.py` is
available through `/work`; otherwise it fails with the exact submodule action
instead of loading the model on 72 GPUs and failing later in the harness.

## MNNVL communication path

Each `dep8` replica is an eight-GPU communication domain spanning two a4x nodes.
The complete 18-node allocation is one 72-GPU A4X NVLink domain; the controller
reports `SwitchType=switch/nvidia_imex`. Each replica uses a subset of that
domain rather than crossing an external fabric boundary.
TensorRT-LLM automatically detects the GB200 multi-node NVLink fabric and, in a
healthy run, logs:

```text
[MnnvlMemory] creating address
Selected communication strategy: NVLinkOneSided
```

Those markers show that the model's cross-rank path selected MNNVL-backed
one-sided NVLink rather than a plain NCCL fallback. The runtime also creates a
NIXL connection manager with `NIXL backend: UCX`; that is a separate cache/data
transfer layer and a UCX marker alone does not prove RDMA transport selection.
Do not force UCX RC/RDMA or an NCCL network override in this measured in-domain
workload: MNNVL/IMEX is the intended primary path, while RDMA is relevant when
traffic crosses NVLink subblocks.

Validate the underlying fabric after, not during, a performance measurement:

```bash
sbatch scripts/slurm_llm/deepseek_r1/mnnvl_fabric_smoke.sbatch
```

The smoke test checks eight GPU ranks across two nodes, requires completed and
successful fabric registration, eight distinct GPU UUIDs (four per node), and
verifies that MNNVL cluster UUID and clique ID agree across ranks when the
driver exposes them. Driver 580 may omit both identifiers; omission is reported
but does not weaken the required fabric state, status, or GPU-identity checks.
Running the smoke concurrently would consume GPU/fabric resources and
contaminate the LoadGen result.

The checker first requests the filtered `nvidia-smi -q -d FABRIC` view. Driver
branches such as 580 can expose the same fields while rejecting that display
selector; in that case the checker automatically parses full `nvidia-smi -q`
output and applies the same state, status, and cluster-UUID requirements.

## TensorRT-LLM server settings

### Runtime and parallelism

```yaml
backend: pytorch
tensor_parallel_size: 8
pipeline_parallel_size: 1
moe_expert_parallel_size: 8
```

- `backend: pytorch` selects the TensorRT-LLM PyTorch/LLM API runtime used by
  the pinned image.
- `tensor_parallel_size: 8` shards tensor-parallel model work across all eight
  GPUs in a replica. This makes the large model fit and exposes aggregate GPU
  compute and memory bandwidth.
- `pipeline_parallel_size: 1` avoids pipeline stages and their bubbles. The
  configuration uses tensor/expert parallelism instead.
- `moe_expert_parallel_size: 8` distributes DeepSeek's MoE experts across the
  eight GPUs, reducing per-GPU expert storage and enabling parallel expert
  execution at the cost of expert-routing communication.

These degrees must match the eight ranks assigned to each server replica.

### Capacity limits

```yaml
max_batch_size: 512
max_num_tokens: 4608
max_seq_len: 23140
```

- `max_batch_size` caps the number of active sequences considered by the
  scheduler. A high cap exposes Offline batching opportunities.
- `max_num_tokens` limits the total tokens scheduled in an iteration. It
  balances GPU utilization against activation/KV memory and iteration time.
- `max_seq_len` permits the long combined prompt and generated response lengths
  required by DeepSeek-R1.

Increasing these values is not automatically faster. Larger batches/token
budgets consume more memory and may increase iteration latency or cause OOM.

### In-flight batching and scheduling

```yaml
enable_chunked_prefill: false
scheduler_config:
  capacity_scheduler_policy: MAX_UTILIZATION
  context_chunking_policy: FIRST_COME_FIRST_SERVED
```

TensorRT-LLM in-flight batching continuously combines prefill and decode work
from active requests. `MAX_UTILIZATION` packs work aggressively to keep the
GPUs occupied, which matches the throughput-oriented Offline scenario.
`FIRST_COME_FIRST_SERVED` gives deterministic ordering for context work when
chunking decisions are relevant.

Chunked prefill is disabled in this tuned profile. This avoids splitting
prefills into smaller scheduling units and their associated coordination
overhead. Enabling it can help other workloads, especially latency-sensitive or
mixed-length workloads, but must be benchmarked with the token budget and batch
shape together.

### KV cache

```yaml
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.9
  enable_block_reuse: false
```

- FP8 KV cache reduces memory footprint and memory traffic relative to wider
  cache formats, allowing more concurrent/long sequences.
- A 0.9 free-memory fraction assigns most available post-weight memory to KV
  cache while retaining headroom for runtime workspaces and transient buffers.
- Block reuse is disabled because MLPerf samples must not benefit from cached
  state left by earlier queries. This protects benchmark validity, even if
  reuse could improve repeated-prefix workloads outside MLPerf.

### Attention and MoE execution

```yaml
enable_attention_dp: true
moe_config:
  backend: CUTEDSL
attention_dp_config:
  enable_balance: true
  batching_wait_iters: 10
  timeout_iters: 500
```

Attention data parallelism allows ranks to process different attention work
while MoE experts remain distributed across the eight ranks. This topology is
suited to DeepSeek's mixture-of-experts structure and creates the `dep8` naming.

`CUTEDSL` selects the tuned MoE kernel backend available in this runtime.
Attention balancing attempts to distribute request work more evenly so one
rank does not become the long pole. The wait and timeout values trade a small
amount of scheduling delay for larger, better-balanced work groups while
preventing indefinite waiting.

### CUDA graphs

```yaml
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
```

CUDA graphs reduce repeated CPU/kernel-launch overhead by replaying captured
execution graphs. The listed batch sizes cover common powers of two plus large
Offline operating points. Padding maps nearby dynamic batches onto a captured
shape, improving graph reuse. More graph sizes consume capture time and memory;
too few sizes cause excessive padding.

### Observability and other runtime settings

```yaml
print_iter_log: true
enable_iter_perf_stats: true
return_perf_metrics: true
enable_layerwise_nvtx_marker: false
cache_transceiver_config:
  backend: DEFAULT
```

Iteration logs and performance metrics make throughput, batching, and stalls
observable during tuning. Layerwise NVTX markers are disabled in the normal run
to avoid unnecessary profiling instrumentation. The default cache transceiver
is retained; this is an IFB deployment rather than a disaggregated context/decode
deployment.

## Environment settings

`trtllm-serve-ifb-dep8-env.yaml` is loaded before `trtllm-serve` starts.

### `TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1`

Enables the MoE communication path that performs expert token exchange without
an additional all-gather. Avoiding redundant collection can reduce communication
volume and synchronization overhead for eight-way expert parallelism.

### `TRTLLM_SERVER_DISABLE_GC=1`

Disables server garbage collection during the performance run. This reduces the
risk of unpredictable GC pauses after the server has allocated its long-lived
model/runtime objects. Memory use must remain bounded because automatic cleanup
is intentionally suppressed.

### `OMPI_MCA_coll_ucc_enable=0`

Disables Open MPI's UCC collective component. The file notes this as required
on Lyris; it is primarily a compatibility/stability choice, not a universal
throughput recommendation. The a4x launch uses PMI-2 for process startup.

### `TRTLLM_ENABLE_PDL=0`

Disables TensorRT-LLM programmatic dependent launch for this validated profile.
This preserves the known execution path. PDL should only be enabled after
revalidating correctness, graph behavior, and performance with the exact image.

### `TLLM_PROFILE_START_STOP=12000-12100`

Defines an iteration window used when profiling capture is enabled. The normal
workflow has `COLLECT_NSYS=0`, so this range does not by itself enable profiling
or improve throughput.

## Why the configuration is optimized

The optimization comes from matching several layers at once:

1. Nine replicas expose full-rack data parallelism without coupling all 72 GPUs
   into one communication domain.
2. Each replica uses eight-way tensor and expert parallelism so the model fits
   and GB200/NVL communication can support the MoE execution pattern.
3. FP4 weights and FP8 KV cache reduce memory footprint and bandwidth demand.
4. In-flight batching, a high batch cap, a tuned token budget, and
   `MAX_UTILIZATION` keep Offline work resident on the GPUs.
5. CUDA graphs reduce launch overhead across expected dynamic batch sizes.
6. Attention balancing and the MoE all-to-all path reduce rank imbalance and
   avoid unnecessary communication.
7. LoadGen concurrency and expected QPS are scaled with the nine replicas while
   preserving MLPerf duration and query-count requirements.

Optimization is empirical. Preserve a known-good result, change one coupled set
of parameters at a time, and compare LoadGen throughput, server iteration
metrics, GPU memory headroom, failures, and result validity. Always run a
separate `AccuracyOnly` test after performance tuning.

## Logs to inspect

The generated sflow output directory contains:

- one `rank<N>.log` per TensorRT-LLM rank;
- `PerformanceOnly.log` for the harness;
- standard `mlperf_log_summary.txt`, `mlperf_log_detail.txt`, and related logs;
- `sflow.log` and task metadata;
- the composed workflow/config copies.

Use the LoadGen summary as the authoritative performance result. Server logs
explain how that result was produced: batch sizes, token scheduling, iteration
performance, memory pressure, readiness, and communication failures.

## Final result acceptance

For DeepSeek-R1 Offline, MLCommons v6.1 reports the official score as
`result_tokens_per_second` (`Tokens per second`). Use `Samples per second` as a
secondary operational comparison against the configured 108 samples/s sizing
target; 108 is not an MLPerf pass/fail floor.

| Check | Required interpretation |
| --- | --- |
| Result validity | `VALID` in `mlperf_log_summary.txt` |
| Minimum duration | At least 600,000 ms and summary reports `Min duration satisfied: Yes` |
| Offline samples | At least 561,664 completed samples for this config |
| Performance sample set | Effective performance sample count at least 4,388 |
| Seeds and errors | Fixed MLPerf seeds present; no LoadGen errors |
| Official score | Final `Tokens per second` / `result_tokens_per_second` |
| Optimization target | Final Samples/s divided by 108; report the ratio without treating it as validity |

PerformanceOnly establishes only the performance result. A complete closed
MLPerf Inference v6.1 DeepSeek-R1 datacenter submission requires Offline plus
at least one of Server or Interactive. The checker describes those two latency
scenarios as optional individually but rejects a submission containing neither.
The minimum run set is:

| Work item | Required result |
| --- | --- |
| Offline `PerformanceOnly` | `VALID`, minimum duration/query gates met, and the official tokens/s score recorded. Job 170528 satisfies this item. |
| Offline `AccuracyOnly` | Exact match at least 80.544618 and tokens/sample between 3497.60466 and 4274.85014. Job 170535 satisfies this item. |
| Offline `TEST06` | Compliance performance run and successful verification of first-token consistency where applicable, EOS behavior, and output-token counts. Job 170536 satisfies this item. |
| Server or Interactive `PerformanceOnly` | Choose at least one. Server is recommended for this colocated IFB workflow; its result must be `VALID` and satisfy 99p TTFT < 2 s and 99p TPOT < 80 ms. Server job 170539 satisfies this item. Interactive instead requires 99p TTFT < 1.5 s and 99p TPOT < 15 ms. |
| Chosen latency scenario `AccuracyOnly` | The same DeepSeek exact-match and tokens/sample bounds. Server job 170540 satisfies this item. |
| Chosen latency scenario `TEST06` | Compliance run and successful verification for that scenario. Server job 170541 satisfies this item. |
| Submission packaging/check | Performance, accuracy, TEST06, measurements, system description, and reproducible code arranged in the standard tree and accepted by the pinned v6.1 submission checker. |

Running both Server and Interactive is optional; including either means it needs
its own performance, accuracy, and TEST06 artifacts. Power is also optional;
claiming it adds separate ranging and power measurements plus their analyzer
logs and settings.

The recommended Server path now uses the same validated a4x environment choices:
shared `.sqsh`, partition `a4x`, PMI-2, exclusive allocation, and no
`--segment`. Preserve Server's scenario-specific serving and harness tuning, and
pass the complete three-path mount override when generating each job.

Run the checked-in comparison after LoadGen finishes:

```bash
python3 scripts/slurm_llm/deepseek_r1/check_offline_result.py \
  <workflow-output>/mlperf_harness/PerformanceOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Offline
```

The checker treats `Tokens per second` as the official result, reports the
Samples/s-to-108 ratio separately, and exits nonzero for missing or failed
validity gates. Exit code 2 means the LoadGen result is still incomplete.

Use the matching independent checker for each result:

```bash
python3 scripts/slurm_llm/deepseek_r1/check_accuracy_result.py \
  <workflow-output>/mlperf_harness/AccuracyOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Server

python3 scripts/slurm_llm/deepseek_r1/check_server_result.py \
  <workflow-output>/mlperf_harness/PerformanceOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Server

python3 scripts/slurm_llm/deepseek_r1/check_test06_result.py \
  <workflow-output>/mlperf_harness/PerformanceOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Server
```

The accuracy checker requires the path's scenario and `AccuracyOnly` mode to
match the LoadGen detail log. The TEST06 checker allows a skipped first-token
check only for Offline; Server and Interactive must report it as `True`.

## Validated a4x run: job 170528

The corrected 72-GPU PerformanceOnly run on 2026-07-10 completed successfully:

| Item | Result |
| --- | ---: |
| Parent Slurm job | `COMPLETED`, exit `0:0`, elapsed `01:22:00` |
| LoadGen validity | `VALID` |
| Official Offline score | `485843` tokens/s |
| Secondary throughput | `129.405` samples/s |
| Configured sizing target | `108` samples/s |
| Target comparison | `1.198194x` (19.82% above target) |
| Completed samples | `561664` |
| Performance sample set | `4388` |
| Minimum duration/queries | Both satisfied |
| LoadGen warnings/errors | None |

The result directory is:

```text
sflow_output/170528-trtllm_ifb-20260710-193617-0e83eb/mlperf_harness/PerformanceOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Offline
```

The complete raw run, final parent logs, exact generated `a4x_ready` sbatch/YAML
pair, generated LoadGen files, and runtime TRT-LLM YAMLs are archived at:

```text
gs://alisachen/mlperf-inference/deepseek-r1/offline/performance-only/2026-07-10/
```

The verified archive contains 127 objects and 1,171,518,006 logical bytes. It
contains only job 170528; failed attempts 170519, 170522, 170525, and 170526 and
their generated variants were explicitly removed.

Transport evidence from the measured workload showed NIXL UCX initialization
in all 72 rank logs and `MnnvlMemory` plus `NVLinkOneSided` selection in each of
the nine replica rank-0 logs. Post-run MNNVL smoke job `170532` completed `0:0`:
eight distinct GPUs across two nodes all reported fabric `Completed/Success`.

This is a valid Offline PerformanceOnly result.

## Validated a4x accuracy run: job 170535

The 72-GPU Offline AccuracyOnly run on 2026-07-11 completed successfully:

| Item | Result |
| --- | ---: |
| Parent Slurm job | `COMPLETED`, exit `0:0`, elapsed `00:24:48` |
| Exact match | `81.175934` (required `>= 80.544618`) |
| Tokens per sample | `3773.608250` (required `3497.60466` through `4274.85014`) |
| Evaluated samples | `4388` |
| Accuracy gates | All passed |

The result directory is:

```text
sflow_output/170535-trtllm_ifb-20260711-075046-cf4b93/mlperf_harness/AccuracyOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Offline
```

The complete raw run, authoritative final parent logs, exact generated wrapper
pair, generated LoadGen configuration, and runtime YAMLs are archived at:

```text
gs://alisachen/mlperf-inference/deepseek-r1/offline/accuracy-only/2026-07-11/
```

The verified archive contains 130 objects and 362,334,869 logical bytes and
contains only job 170535. The accuracy evaluator's temporary editable-install
files were removed afterward and the nested benchmark sources were restored.

## Validated a4x Offline TEST06 run: job 170536

The 72-GPU Offline TEST06 compliance run on 2026-07-11 completed successfully:

| Item | Result |
| --- | ---: |
| Parent Slurm job | `COMPLETED`, exit `0:0`, elapsed `00:13:35` |
| Harness step | `COMPLETED`, exit `0:0`, elapsed `00:05:10` |
| Audit result | `TEST06_PASS`; `audit_success: true` |
| First-token check | `Skipped` (expected for Offline) |
| EOS check | `True` |
| Sample-length check | `True` |
| Final verification marker | `TEST06 verification complete` |

The result directory is:

```text
sflow_output/170536-trtllm_ifb-20260711-200610-c678f5/mlperf_harness/PerformanceOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Offline
```

The independently run compliance validator passed all TEST06 checks. This
100-sample audit run's LoadGen summary reports `VALID`, with its audit-specific
minimum duration, minimum queries, and early-stopping checks satisfied. Its
mini-run throughput is not the scored Offline performance result, so this
artifact must not replace job 170528.

The complete compliance run and its exact generated/configuration artifacts are
archived at:

```text
gs://alisachen/mlperf-inference/deepseek-r1/offline/test06/2026-07-11/
```

The verified archive contains 112 objects and 95,746,790 logical bytes and
contains only job 170536.

## Validated a4x Server PerformanceOnly run: job 170539

The corrected 72-GPU Server PerformanceOnly run on 2026-07-11 completed
successfully:

| Item | Result |
| --- | ---: |
| Parent Slurm job | `COMPLETED`, exit `0:0`, elapsed `00:50:42` |
| Harness step | `COMPLETED`, exit `0:0`, elapsed `00:42:42` |
| LoadGen validity | `VALID` |
| Official Server score | `392959` completed tokens/s |
| p99 time to first token | `950.001124` ms (required `< 2000` ms) |
| p99 time per output token | `69.162691` ms (required `< 80` ms) |
| Completed queries | `210624` |
| Scheduled/completed samples per second | `110.19` / `104.73` |
| Minimum duration/queries | Both satisfied |
| Early stopping | Satisfied |
| LoadGen errors | None |

The result directory is:

```text
sflow_output/170539-trtllm_ifb-20260711-212202-b939e3/mlperf_harness/PerformanceOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Server
```

The independent Server result validator passed every validity and latency
gate. The full workflow, authoritative parent logs, exact generated wrapper
pair, generated LoadGen configuration, and runtime YAMLs are archived at:

```text
gs://alisachen/mlperf-inference/deepseek-r1/server/performance-only/2026-07-11/
```

The verified archive contains 128 objects and 467,850,135 logical bytes and
contains only job 170539. Failed infrastructure attempt 170537 was not
archived. After the Slurmd connection-capacity recovery, all 18 nodes returned
to `IDLE` without a new connection-ceiling warning.

## Validated a4x Server AccuracyOnly run: job 170540

The 72-GPU Server AccuracyOnly run on 2026-07-11 completed successfully:

| Item | Result |
| --- | ---: |
| Parent Slurm job | `COMPLETED`, exit `0:0`, elapsed `00:20:02` |
| Harness step | `COMPLETED`, exit `0:0`, elapsed `00:11:57` |
| Effective scenario/mode | `Server` / `AccuracyOnly` |
| Exact match | `81.175934` (required `>= 80.544618`) |
| Tokens per sample | `3725.837284` (required `3497.60466` through `4274.85014`) |
| Evaluated samples | `4388` |
| Accuracy gates | All passed |

The result directory is:

```text
sflow_output/170540-trtllm_ifb-20260711-221453-29169d/mlperf_harness/AccuracyOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Server
```

The scenario-aware independent accuracy validator passed every gate. The full
workflow, authoritative parent logs, exact generated wrapper pair, generated
LoadGen configuration, and runtime YAMLs are archived at:

```text
gs://alisachen/mlperf-inference/deepseek-r1/server/accuracy-only/2026-07-11/
```

The verified archive contains 130 objects and 229,094,564 logical bytes and
contains only job 170540. The evaluator's temporary editable-install files
were removed afterward and the nested benchmark sources were restored exactly.

## Validated a4x Server TEST06 run: job 170541

The 72-GPU Server TEST06 compliance run on 2026-07-11 completed successfully:

| Item | Result |
| --- | ---: |
| Parent Slurm job | `COMPLETED`, exit `0:0`, elapsed `00:13:13` |
| Harness step | `COMPLETED`, exit `0:0` |
| Audit result | `TEST06_PASS`; `audit_success: true` |
| First-token check | `True` (required for Server) |
| EOS check | `True` |
| Sample-length check | `True` |
| Final verification marker | `TEST06 verification complete` |
| p99 TTFT/TPOT | `1832.768547` ms / `14.005109` ms |

The result directory is:

```text
sflow_output/170541-trtllm_ifb-20260711-223711-723a4a/mlperf_harness/PerformanceOnly/GB200-NVL72_GB200-186GB_aarch64x72_TRT/deepseek-r1/Server
```

The scenario-aware independent TEST06 validator passed every metadata,
scenario, mode, first-token, EOS, sample-length, and completion gate. The raw
LoadGen summary reports `INVALID` only because the 100-query audit run did not
meet normal performance early-stopping sufficiency: it says 359 more queries
would be needed. Minimum duration, minimum queries, and the TTFT/TPOT
performance constraints all passed. LoadGen performance validity is not the
TEST06 compliance criterion, and this mini-run does not replace or invalidate
scored Server PerformanceOnly job 170539.

The full workflow, authoritative parent logs, exact generated wrapper pair,
generated LoadGen configuration, and runtime YAMLs are archived at:

```text
gs://alisachen/mlperf-inference/deepseek-r1/server/test06/2026-07-11/
```

The verified archive contains 112 objects and 29,339,932 logical bytes and
contains only job 170541. All 18 nodes returned to `IDLE`.

The minimum required Offline-plus-Server benchmark job matrix is now complete.
Submission staging, accuracy-log truncation, system/measurement review, and the
pinned v6.1 submission checker remain as non-benchmark packaging work.
