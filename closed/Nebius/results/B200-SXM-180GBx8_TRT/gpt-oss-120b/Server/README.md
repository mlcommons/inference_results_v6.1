To run this benchmark, first follow the setup steps in `closed/Nebius/README.md`, then the instructions in `closed/Nebius/src/nv_mlpinf/benchmarks/gpt_oss_120b/README.md`.

Afterwards run the following commands for benchmarking

# Server Performance
```
export SYSTEM_NAME=B200-SXM-180GBx8
export SUBMITTER=Nebius
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Server"
nv-mlpinf run_harness --benchmarks=gpt-oss-120b --scenarios=Server --test_mode=PerformanceOnly
```

# Server Accuracy
```
export SYSTEM_NAME=B200-SXM-180GBx8
export SUBMITTER=Nebius
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Server"
nv-mlpinf run_harness --benchmarks=gpt-oss-120b --scenarios=Server --test_mode=AccuracyOnly
```

# Server TEST07
```
export SYSTEM_NAME=B200-SXM-180GBx8
export SUBMITTER=Nebius
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Server"
make run_audit_test07 RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server --test_mode=PerformanceOnly"
```

# Server TEST09
```
export SYSTEM_NAME=B200-SXM-180GBx8
export SUBMITTER=Nebius
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Server"
make run_audit_test09 RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server --test_mode=PerformanceOnly"
```