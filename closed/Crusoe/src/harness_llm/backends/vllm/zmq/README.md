# Distributed SUT with ZMQ

## Usage

1) Start the SUT client
2) Start the workers on the nodes

# SGLang engine (deepseek-r1)

The ZMQ transport is engine-agnostic: the same head SUT / worker split drives
either the vllm or the sglang engine. The engine is picked automatically from
the model yaml — a config that declares `sglang_engine_config` (e.g.
`deepseek-r1/offline_mi355x_mn.yaml`) runs on sglang, one that declares
`vllm_engine_config` runs on vllm. Both the SUT client and the workers must be
started with `--backend zmq`.

sglang's data parallelism (`dp_size`, dp-attention) is intra-engine, so a
single worker instance spans `tp_size * pp_size` GPUs. For deepseek-r1 with
`tp_size: 8` on an 8-GPU node this is exactly one worker instance, i.e. the head
and its single worker both live on the same node.

## Single-node (head + worker on the same 8-GPU node)

Get the node IP (used for both, since the worker connects back to the head):
```bash
hostname -I
```

### SUT client (head)
```bash
bash run_harness.sh --config-path deepseek-r1/ --config-name offline_mi355x_mn \
  --backend zmq test_mode=performance port=12345 \
  harness_config.device_count=8 \
  harness_config.output_log_dir=results/deepseek_offline_zmq
```

### Worker (run on the SAME node; use the head IP)
```bash
python harness_llm/backends/vllm/zmq/distributed_sync_offline.py \
  --config-path deepseek-r1/ --config-name offline_mi355x_mn --backend zmq \
  node_id=$(hostname) headnode_address=<HEADNODE_IP>:12345
```

`port` and `port+1` are both used (SUT->worker and worker->SUT routers). Start
the SUT client first, then the worker.
