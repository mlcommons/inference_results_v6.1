# Running Llama2-70B-99

Run from `/lab-mlperf-inference/submission` inside the benchmark container.

## Offline

```bash
python3 submission.py --model llama2-70b-99 experiment \
  --scenario Offline \
  --model-conf ../code/llama2-70b-99/offline_mi355x.yaml \
  --user-conf ../code/llama2-70b-99/user_mi355x.conf
