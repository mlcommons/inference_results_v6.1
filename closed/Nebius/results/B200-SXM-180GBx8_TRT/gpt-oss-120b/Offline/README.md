To run this benchmark, first follow the setup steps in `closed/Nebius/README.md`, then the instructions in `closed/Nebius/src/nv_mlpinf/benchmarks/gpt_oss_120b/README.md`.

Afterwards run the following commands for benchmarking
# Offline Performance
```
export SYSTEM_NAME=B200-SXM-180GBx8
export SUBMITTER=Nebius
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Offline"
nv-mlpinf run_harness --benchmarks=gpt-oss-120b --scenarios=Offline --test_mode=PerformanceOnly
```

# Offline Accuracy
```
export SYSTEM_NAME=B200-SXM-180GBx8
export SUBMITTER=Nebius
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Offline"
nv-mlpinf run_harness --benchmarks=gpt-oss-120b --scenarios=Offline --test_mode=AccuracyOnly
```

# Offline TEST07
```
export SYSTEM_NAME=B200-SXM-180GBx8
export SUBMITTER=Nebius
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Offline"
make run_audit_test07 RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --test_mode=PerformanceOnly"
```

# Offline TEST09
```
export SYSTEM_NAME=B200-SXM-180GBx8
export SUBMITTER=Nebius
nv-mlpinf run_llm_server --benchmarks=gpt-oss-120b --scenarios=Offline"
make run_audit_test09 RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --test_mode=PerformanceOnly"
```