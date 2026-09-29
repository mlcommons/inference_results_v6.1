To run this benchmark, follow the setup steps in closed/NVIDIA/README.md.

System: NVIDIA GB200 NVL4 (4x GB200-186GB, RHEL 9.8)
Framework: vLLM v0.20.2rc1 + NVIDIA Dynamo
Scenario: Server (Poisson arrival, target QPS=46)

Steps:
1. source env-export.sh
2. Set SYSTEM to GB200-NVL72_GB200-186GB_aarch64x4 and scenario to Server
3. bash run-sflow-benchmark.sh
