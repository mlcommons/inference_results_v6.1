# Server Scenario

This directory contains the MLPerf inference results for the Server scenario.

## Contents

- `accuracy/`: Accuracy test results
- `performance/`: Performance test results
- `measurements.json`: Measurement configuration
- `mlperf.conf`: MLPerf configuration file
- `user.conf`: User configuration overrides

## System

**System:** GB200-NVL4 (4x GB200-186GB, Red Hat OpenShift 4.21)
**Framework:** vLLM 0.24.0
**Scenario:** Server (QPS 39.14)

## Reproducing Results

See the full reproduction guide:
https://github.com/openshift-psap/mlperf-inference-6.1-redhat/blob/master/harness/README.md

### Quick Start

```bash
git clone --recurse-submodule https://github.com/openshift-psap/mlperf-inference-6.1-redhat.git
cd mlperf-inference-6.1-redhat

# Deploy model servers
cd setup/llm-d/GB200/
bash deploy_gptoss120b.sh server --standalone

# Run benchmark (inside client pod)
cd harness/
bash run_server.sh
```
