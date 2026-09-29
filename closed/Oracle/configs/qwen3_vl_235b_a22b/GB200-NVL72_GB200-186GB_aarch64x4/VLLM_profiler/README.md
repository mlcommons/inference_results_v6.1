# Qwen3-VL — GB200-NVL4 Nsight Systems Profiling

A **developer profiling** deployment (not an MLPerf submission scenario). One
`vllm serve` under `nsys profile` (TP=4 for `Interactive/`; TP=1 for `Server/` and
`Offline/`, matching the per-worker engine shape dynamo replicates in the non-profiler
runs), driven by the inference-endpoint client with profiling enabled in `endpoint.yaml`
(`settings.profiling.engine: vllm` — POSTs `/start_profile` at the
performance-phase start, `/stop_profile` at the end). When the client task exits
(success OR failure) an EXIT trap drops a shared sentinel; the server then SIGINTs
nsys to flush the `.nsys-rep` and exits cleanly.

Uses the **2-task** template `_shared/templates/vllm_serve_endpoints_profile.yaml`
(vllm_server + endpoint_benchmark) — no etcd / NATS / dynamo.frontend / dynamo.vllm.

## Three modes (sub-folders)

| Folder | Load pattern | Dataset | TP | max-num-batched-tokens | DELAY_ITERATIONS |
|--------|--------------|---------|----|------------------------|------------------|
| `Server/`      | online / poisson streaming on              | full (48k)  | 1 | 8192 | 500 |
| `Interactive/` | online / poisson streaming on, low QPS     | 8k          | 4 | 4864 | 100 |
| `Offline/`     | `max_throughput` (saturating), no streaming | full (48k)  | 1 | 4864 | 500 |

Each mode's `VLLM_FLAGS` matches the corresponding non-profiler `VLLM/<mode>/` config
(GB200 values: max-num-batched-tokens 8192 for Server, 4864 for Interactive/Offline,
model V6.1-FP8-KV, vit fp8 scale path, priority scheduling) plus the cuda profiler +
NVTX flags; dynamo-only flags (`--enable-multimodal`, `--numa-bind*`, frontend
`--router-mode`) are dropped — the template NUMA-pins via `get_gpu_numa.py` instead.
`DELAY_ITERATIONS` is lower for `Interactive/` because its low QPS accrues engine
iterations more slowly.

Each sub-folder is a complete set: `qwen3vl_config.yaml` (per-mode TP + cuda profiler
flags), `endpoint.yaml` (5-min hard stop via `max_duration_ms`), `worker_env.yaml`. They
share the same template and the same `MODE`-driven nsys filename scheme.

nsys output basename: `q3vl_gb200x4_<MODE>_delay<DELAY_ITERATIONS>_max<MAX_ITERATIONS>.nsys-rep`
— e.g. `q3vl_gb200x4_server_delay500_max100`, `q3vl_gb200x4_interactive_delay100_max100`,
or `q3vl_gb200x4_offline_delay500_max100`.

## Launch (inside a 1-node salloc, from `closed/NVIDIA/`)

Pick the mode by swapping the third `-f`:

```bash
SROOT=configs/qwen3_vl_235b_a22b
MODE_DIR=Server        # or: Interactive, Offline
sflow run \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_serve_endpoints_profile.yaml \
  -f $SROOT/GB200-NVL72_GB200-186GB_aarch64x4/VLLM_profiler/$MODE_DIR/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HOME/.cache/huggingface \
  --set SLURM_ACCOUNT=<account> \
  --set SLURM_PARTITION=<gb200 partition>
```

The endpoint image is the `_shared/slurm_env.yaml` default (same as the GB300 profile;
it supports the `settings.profiling` client-side trigger).

For batch submission, swap `sflow run` → `sflow batch ... --partition <gb200 partition>
--account <account> --time 02:00:00 --nodes 1 --sbatch-path <dir>/sbatch.sh --submit`.

## Output

All run artifacts land under
`closed/NVIDIA/profile_out/<jobid>-vllm_serve_endpoints_profile-<timestamp>-<hash>/`
(sflow's per-run id, from `SFLOW_WORKFLOW_OUTPUT_DIR`; inside the `/work` mount — no
dependence on sflow output-dir mounts):

- Trace: `.../vllm_server/q3vl_gb200x4_<mode>_..._.nsys-rep` (open in Nsight Systems).
- Server log: `.../vllm_server/vllm_serve.log`.
- Client log + report: `.../endpoint_benchmark/inference_endpoint.log`,
  `results/qwen3_vl_235b_a22b_profile_<mode>_gb200x4/`.

## Knobs (per-mode, in each `qwen3vl_config.yaml`)

`MODE`, `DELAY_ITERATIONS`, `MAX_ITERATIONS` are the tunable knobs (also encoded in the
nsys filename). Override at launch, e.g. `--set MAX_ITERATIONS=200`.
