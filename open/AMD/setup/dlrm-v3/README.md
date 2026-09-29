# DLRM-v3 MI355X C1-on setup and execution

Build and install LoadGen 6.0.17 from the packaged `src/dlrm-v3/loadgen`
source, then use `user_mi355x8_nve_b64_qps16200_PROD10min_C1on_v61.conf`. The wheel used for these runs had SHA256:

`bd72dcd50d1ab6f1bebf4b7cfe605aba9226ac03b2767c0300d77ee3b6e2a4d5`

The exact launch commands were:

```bash
WINDOW=1 MAX_ATTN_LEN=1024 BATCH=64 SCENARIO=Server MODE=performance \
CONF=user_mi355x8_nve_b64_qps16200_PROD10min_C1on_v61.conf \
TAG=mi355x8_C1on_v61seed_b64_q16200_SERVER10min WAIT=2400 \
bash scripts/run/run_gold.sh

WINDOW=1 MAX_ATTN_LEN=1024 BATCH=64 SCENARIO=Server \
CONF=user_mi355x8_nve_b64_qps16200_PROD10min_C1on_v61.conf \
TAG=mi355x8_C1on_v61seed_b64_SERVER_accuracy WAIT=7200 \
bash scripts/run/run_accuracy.sh

WINDOW=1 MAX_ATTN_LEN=1024 BATCH=64 SCENARIO=Offline MODE=performance \
CONF=user_mi355x8_nve_b64_qps16200_PROD10min_C1on_v61.conf \
TAG=mi355x8_C1on_v61seed_b64_q17000_OFFLINE10min WAIT=2400 \
bash scripts/run/run_gold.sh

WINDOW=1 MAX_ATTN_LEN=1024 BATCH=64 SCENARIO=Offline \
CONF=user_mi355x8_nve_b64_qps16200_PROD10min_C1on_v61.conf \
TAG=mi355x8_C1on_v61seed_b64_OFFLINE_accuracy WAIT=7200 \
bash scripts/run/run_accuracy.sh
```

For convenience, `scripts/run/run_open.sh` pins these required switches and
accepts `Server`, `ServerAccuracy`, `Offline`, or `OfflineAccuracy`.
