To run this benchmark, first follow the setup steps in `closed/Nebius/README.md`.

# Server Performance
## LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
## Harness
When the server is ready run
```
make run_harness
```

# Server Accuracy
## LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint --test_mode=AccuracyOnly"
make run_llm_server
```
## Harness
When the server is ready run
```
make run_harness
```

# Server TEST07
## LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
## Harness
When the server is ready run
```
make run_audit_test07
```

# Server TEST09
## LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
## Harness
When the server is ready run
```
make run_audit_test09
```