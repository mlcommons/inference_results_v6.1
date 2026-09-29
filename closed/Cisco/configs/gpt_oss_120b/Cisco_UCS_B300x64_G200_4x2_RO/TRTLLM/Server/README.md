# GPT-OSS-120B x64 Server IFB

Cluster-specific roots, node names, accounts, partitions, and proxy settings
are supplied through environment variables in the launcher.

The selected full run uses 64 independent TP1/EP1 endpoints, QPS350,
600,000 ms minimum duration, 270,336 minimum queries, and QSL 6,396.
The submitted worker configuration is used without runtime source patches.

Run the checked-in profile after setting the required `MLPERF_*` environment
variables documented by the launcher:

```bash
export PROFILE="${CISCO_SUBMISSION_ROOT}/configs/gpt_oss_120b/Cisco_UCS_B300x64_G200_4x2_RO/TRTLLM/Server"
export MLPERF_OUTPUT_DIR=/path/to/output/gptoss-x64-server
export MLPERF_SERVER_HOSTS=host01,host01,host01,host01,host01,host01,host01,host01,host02,host02,host02,host02,host02,host02,host02,host02,host03,host03,host03,host03,host03,host03,host03,host03,host04,host04,host04,host04,host04,host04,host04,host04,host05,host05,host05,host05,host05,host05,host05,host05,host06,host06,host06,host06,host06,host06,host06,host06,host07,host07,host07,host07,host07,host07,host07,host07,host08,host08,host08,host08,host08,host08,host08,host08

MLPERF_TEST_MODE=PerformanceOnly sbatch \
  --account="${MLPERF_SLURM_ACCOUNT}" \
  --partition="${MLPERF_SLURM_PARTITION}" \
  "${PROFILE}/launchers/go_x64_server_ifb.sh"
```

Use `AccuracyOnly`, `TEST07`, or `TEST09` with `MLPERF_TEST_MODE`. The launcher
retains target QPS 350 and the unpatched worker path.
