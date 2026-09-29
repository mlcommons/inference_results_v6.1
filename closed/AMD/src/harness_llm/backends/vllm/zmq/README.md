# Distributed SUT with ZMQ

## Usage

1) Start the SUT client
2) Start the workers on the nodes

### SUT client

First, get the IP for the workers
```bash
hostname -I
```

You need to start on one node. Make sure that device_count is the sum of all node's device_count.

NOTE: both port and port+1 will be used

#### Server
```bash
bash run_harness.sh --config-path llama2-70b-99/ --config-name server_mi355x --backend zmq test_mode=performance harness_config.output_log_dir=results/llama2_server_performance_zmq port=1234 harness_config.device_count=<SUM-OF-ALL-GPUS> harness_config.target_qps=<Node-count x single-node-qps>
```

#### Offline

```bash
bash run_harness.sh --config-path llama2-70b-99/ --config-name offline_mi355x --backend zmq test_mode=performance harness_config.output_log_dir=results/llama2_offline_performance_zmq port=12345 harness_config.device_count=<SUM-OF-ALL-GPUS> harness_config.target_qps=<Node-count x single-node-qps>
```

#### Debug

For debugging, append the following flags:
```bash
harness_config.target_qps=300 harness_config.duration_sec=30 harness_config.debug_record_sample_latencies=True harness_config.debug_print_finished=True harness_config.debug_dump_model_output=True
```

### Workers

You need to start on each node. Even on the headnode!

Use the IP of the headnode (see SUT Client)

#### Server

```bash
python harness_llm/backends/vllm/zmq/distributed_async_server.py --config-path llama2-70b-99/ --config-name server_mi355x node_id=`hostname` headnode_address=<IP>:12345
```

#### Offline

```bash
python harness_llm/backends/vllm/zmq/distributed_sync_offline.py --config-path llama2-70b-99/ --config-name offline_mi355x node_id=`hostname` headnode_address=<IP>:12345
```

# Disaggregated Prefill

Both the server and offline disagg paths now use the **same SUT-driven routing
model**: a single SUT class (`DistributedDisaggSUT`) and a single worker
entrypoint (`distributed_async_disagg.py`) serve both scenarios. The SUT owns
all routing end-to-end; workers are scenario-agnostic engine runners that only
talk to the SUT and never talk to each other.

The flow (identical for both scenarios) is:

1) Each worker registers with the SUT (announcing only its role).
2) The SUT assigns every worker a globally-unique `kv_rank`, splits them into a
   prefill pool and a decode pool (prefills get the low ranks), and waits until
   all workers have reported the KV listen address bound by their vLLM engine.
3) For each query the SUT picks **one prefill + one decode**. Selection is
   per-role and configurable via `harness_config.prefill_schedule_algo` /
   `harness_config.decode_schedule_algo`, each one of `round_robin` (default),
   `shortest_queue`, or `shortest_queue_with_tokens`.
4) Dispatch is **asymmetric**: the query is sent eagerly to the chosen prefill,
   and only released to the chosen decode after the prefill reports
   `PREFILL_DONE`. The KV cache itself is transferred by vLLM's
   `P2pNcclConnector`, keyed off the prefill/decode KV addresses that the SUT
   bakes into the `request_id`.

`harness_config.disagg: true` in the yaml selects the disagg SUT
(`DistributedDisaggSUT`) for both scenarios; the SUT reads `scenario` from the
config to specialize its behavior. Warmup (`harness_config.enable_warmup`) is
SUT-coordinated end-to-end: the SUT runs dummy samples through the real
prefill -> decode path before serving, then resets all load-balancing state.

The only differences between the two scenarios:

- **Server** streams tokens from the decode and the SUT reports first-token /
  TTFT plus the final completion per sample.
- **Offline** returns one full output per sample (no token streaming) and the
  SUT reports a single `QuerySamplesComplete` per sample (no first-token /
  TTFT).

## Server

The routing model, dispatch flow, `disagg` flag, and warmup are described in
the shared section above. Server-specific note: without `harness_config.disagg:
true` the plain `DistributedServerSUT` is used instead of the disagg SUT.

### Worker counts (single- or multi-node)

The disaggregated server supports `N_p` prefill instances and `N_d` decode
instances across one or more nodes. Counts live in `harness_config`:

```yaml
harness_config:
  prefills_count: 2          # this node's prefills
  decodes_count: 6           # this node's decodes
  # total_prefills_count / total_decodes_count: GLOBAL totals (N_p / N_d).
  # Omit for single-node (derived from the local counts above); set them for
  # multi-node -- the *_disagg_mn.yaml variants do this.
```

- **Single-node (default):** omit `total_*_count`. Both the SUT and the workers
  derive the global totals from this node's `prefills_count` / `decodes_count`.
- **Multi-node:** use the `*_disagg_mn.yaml` config, which sets the global
  `total_*_count` explicitly. As a guard, if the SUT sees workers registering
  from more than one host while `total_*_count` is unset, it aborts with an
  error instead of waiting for the wrong number of workers.

Constraints (asserted at startup):
- `(prefills_count + decodes_count) * data_parallel_size *
  tensor_parallel_size * pipeline_parallel_size == harness_config.device_count`
  (local check; `device_count` is per-node).
- `prefills_count <= total_prefills_count` and `decodes_count <= total_decodes_count`.

### Port layout

The server disagg processes use a deterministic port block starting at
`base_port` (the `port=` flag on the SUT client and the port half of
`headnode_address` on the worker side):

| Offset                          | Purpose                                                         |
|---------------------------------|-----------------------------------------------------------------|
| `base + 0`                      | SUT router (SUT -> workers)                                     |
| `base + 1`                      | SUT router (workers -> SUT)                                     |
| `base + 2 + kv_rank`            | KV cache port for the worker with `kv_rank`, in `[0, N_p + N_d)` (prefills first) |

Total ports used: `2 + N_p + N_d`.

### SUT client

First, get the head node's IP for the workers:
```bash
hostname -I
```

Start the SUT on the head node. `disagg: true` in the yaml selects the disagg
SUT, so the command is just:

```bash
bash run_harness.sh --config-path gpt-oss-120b/ --config-name server_mi355x_disagg \
  --backend zmq port=8888 \
  harness_config.output_log_dir=results/gpt-oss_server_disagg_zmq
```

For a multi-node run, use the `_disagg_mn` config instead
(`--config-name server_mi355x_disagg_mn`).

To debug, append `harness_config.debug_record_sample_latencies=True
harness_config.debug_print_finished=True harness_config.debug_dump_model_output=True`.
For a performance run instead of accuracy, append `test_mode=performance`.

### Workers

Start on EVERY node (including the head node). Because the SUT owns all
routing, the worker CLI is just:

| Flag                              | Meaning |
|-----------------------------------|---------|
| `node_id=<name>`                  | Unique label for this node (any string). |
| `headnode_address=<ip>:<port>`    | SUT host + base port. The base port + offsets are reused for the disagg port layout (see above). |

Workers auto-detect their reachable IP via vLLM's `get_ip()` (honoring
`VLLM_HOST_IP` for multi-NIC hosts) and ranks are SUT-assigned. The same yaml
is used on every node.

#### Single-node

```bash
python harness_llm/backends/vllm/zmq/distributed_async_disagg.py \
  --config-path gpt-oss-120b/ --config-name server_mi355x_disagg \
  node_id=$(hostname) headnode_address=<HEADNODE_IP>:8888
```

#### Multi-node (run the SAME command on EACH node)

The `_disagg_mn` yaml carries the global `total_*_count`:

```bash
python harness_llm/backends/vllm/zmq/distributed_async_disagg.py \
  --config-path gpt-oss-120b/ --config-name server_mi355x_disagg_mn \
  node_id=$(hostname) headnode_address=<HEADNODE_IP>:8888
```

If nodes are uneven, override that node's local counts with
`harness_config.prefills_count=<n> harness_config.decodes_count=<m>` (the global
`total_*_count` in the `_disagg_mn` yaml stays the same on every node).

## Offline

The offline disagg path now uses the **same SUT-driven routing as server**
(see the shared routing model above): the SUT assigns `kv_rank`s, splits
workers into prefill/decode pools, does per-pool load balancing via
`prefill_schedule_algo` / `decode_schedule_algo`, dispatches eagerly to the
prefill and gates the decode on `PREFILL_DONE`, and the KV cache is moved by
the `P2pNcclConnector`. The only offline-specific differences are:

- the decode returns **one full output per sample** (no token streaming), and
- the SUT reports a single `QuerySamplesComplete` per sample (no first-token /
  TTFT).

`harness_config.disagg: true` selects `DistributedDisaggSUT` for offline too,
and `enable_warmup` is SUT-coordinated exactly as for server.

### Worker counts and port layout

Identical to the server path: same `prefills_count` / `decodes_count` (local)
plus optional `total_*_count` (global, omit for single-node / set via
`*_disagg_mn.yaml` for multi-node) convention, the same startup constraints,
and the same port layout (`base+0` SUT->workers, `base+1` workers->SUT,
`base+2+kv_rank` per-worker KV port; total `2 + N_p + N_d`).

### SUT client

```bash
bash run_harness.sh --config-path gpt-oss-120b/ --config-name offline_mi355x_disagg \
  --backend zmq port=8888 \
  harness_config.output_log_dir=results/gpt-oss_offline_disagg_zmq
```

For multi-node, use `--config-name offline_mi355x_disagg_mn`.

### Workers

Start on EVERY node. Same `distributed_async_disagg.py` entrypoint as the
server scenario (the SUT reads `scenario: offline` from the config to
specialize behavior); the worker CLI is just `node_id` + `headnode_address`,
exactly as for server.

#### Single-node

```bash
python harness_llm/backends/vllm/zmq/distributed_async_disagg.py \
  --config-path gpt-oss-120b/ --config-name offline_mi355x_disagg \
  node_id=$(hostname) headnode_address=<HEADNODE_IP>:8888
```

#### Multi-node (run the SAME command on EACH node)

```bash
python harness_llm/backends/vllm/zmq/distributed_async_disagg.py \
  --config-path gpt-oss-120b/ --config-name offline_mi355x_disagg_mn \
  node_id=$(hostname) headnode_address=<HEADNODE_IP>:8888
```