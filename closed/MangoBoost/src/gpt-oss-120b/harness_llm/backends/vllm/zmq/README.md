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
