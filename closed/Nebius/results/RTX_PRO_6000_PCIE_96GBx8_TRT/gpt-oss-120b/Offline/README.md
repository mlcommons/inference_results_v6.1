To run this benchmark, first follow the setup steps in `closed/Nebius/README.md`.

# Offline Performance
## LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
## Harness
When the server is ready run
```
make run_harness
```

# Offline Accuracy
## LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"
make run_llm_server
```
## Harness
When the server is ready run
```
make run_harness
```

# Offline TEST07
## LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
## Harness
When the server is ready run
```
make run_audit_test07
```

# Offline TEST09
## LLM server
Run the following commands in the docker container:
```
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
make run_llm_server
```
## Harness
When the server is ready run
```
make run_audit_test09
```