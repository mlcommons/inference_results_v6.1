# Qwen3-VL — G894 B300x8 (8×B300 SXM6) Nsight Systems Profiling

A **developer profiling** deployment (not an MLPerf submission scenario). One
`vllm serve` under `nsys profile` (TP=1, matching the per-worker engine shape dynamo
replicates in the non-profiler runs), driven by the inference-endpoint client with
profiling enabled in `endpoint.yaml` (`settings.profiling.engine: vllm` — POSTs
`/start_profile` at the performance-phase start, `/stop_profile` at the end). When the
client task exits (success OR failure) an EXIT trap drops a shared sentinel; the server
then SIGINTs nsys to flush the `.nsys-rep` and exits cleanly.

Uses the **2-task** template `_shared/templates/vllm_serve_endpoints_profile.yaml`
(vllm_server + endpoint_benchmark) — no etcd / NATS / dynamo.frontend / dynamo.vllm.

## Two modes (sub-folders — G894 B300x8 has no Interactive scenario)

| Folder | Load pattern | Dataset | TP | max-num-batched-tokens | DELAY_ITERATIONS |
|--------|--------------|---------|----|------------------------|------------------|
| `Server/`  | online / poisson streaming on               | full (48k) | 1 | 13824 | 500 |
| `Offline/` | `max_throughput` (saturating), no streaming | full (48k) | 1 | 13824 | 500 |

Each mode's `VLLM_FLAGS` matches the corresponding non-profiler `VLLM/<mode>/` config
(B300 values: max-num-batched-tokens 13824, model V6.1-FP8-KV, vit fp8 scale path;
Server adds priority scheduling, Offline adds mm-encoder-tp-mode data) plus the cuda
profiler + NVTX flags; dynamo-only flags (`--enable-multimodal`, frontend
`--router-mode`) are dropped.

**amd64 system**: both configs override the arm64 `CONTAINER_IMAGE` /
`ENDPOINT_CONTAINER_IMAGE` defaults from `_shared/slurm_env.yaml` — the server image
is the amd64 build used by the non-profiler B300 runs, and the endpoint image is
`inference-endpoint-dev-amd64:01c8a37e`, the amd64 build of the same profiling-capable
(`settings.profiling` / `/start_profile` trigger) endpoint image the GB200/GB300
profiles use.

**Backend image tag ends in `-nsys`**: the upstream `amd64_..._gb300deps` image was
built before the Dockerfile's nsight-systems install step (`dpkg` shows only
nsight-compute; the arm64 image ships `nsight-systems-cli-2025.4.1`), so these configs
pin `..._gb300deps-nsys` — the same image with exactly that Dockerfile step applied on
top (nsys 2025.4.1.172, matching arm64). Once the upstream amd64 image is rebuilt from
the current `mpi-dynamo-vllm.Dockerfile`, the suffix tag can be retired.

Registry pulls can also be slow from B300 clusters (enroot's 300 s per-attempt curl
timeout truncates the 31 GB image layer): pre-import both images once with
`enroot import` and pass the `.sqsh` paths via `--set CONTAINER_IMAGE=...
--set ENDPOINT_CONTAINER_IMAGE=...`.

Each sub-folder is a complete set: `qwen3vl_config.yaml` (cuda profiler flags +
amd64 images), `endpoint.yaml` (5-min hard stop via `max_duration_ms`),
`worker_env.yaml`. They share the same template and the same `MODE`-driven nsys
filename scheme.

nsys output basename: `q3vl_g894_b300x8_<MODE>_delay<DELAY_ITERATIONS>_max<MAX_ITERATIONS>.nsys-rep`
— e.g. `q3vl_g894_b300x8_server_delay500_max100` or `q3vl_g894_b300x8_offline_delay500_max100`.

## Launch (inside a 1-node salloc, from `closed/NVIDIA/`)

Pick the mode by swapping the third `-f`:

```bash
SROOT=configs/qwen3_vl_235b_a22b
MODE_DIR=Server        # or: Offline
sflow run \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_serve_endpoints_profile.yaml \
  -f $SROOT/G894-ZD3-AAX7_B300-SXM-270GBx8/VLLM_profiler/$MODE_DIR/qwen3vl_config.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HOME/.cache/huggingface \
  --set SLURM_ACCOUNT=<account> \
  --set SLURM_PARTITION=<b300 partition>
```

For batch submission, swap `sflow run` → `sflow batch ... --partition <b300 partition>
--account <account> --time 02:00:00 --nodes 1 --sbatch-path <dir>/sbatch.sh --submit`.

## Output

All run artifacts land under
`closed/NVIDIA/profile_out/<jobid>-vllm_serve_endpoints_profile-<timestamp>-<hash>/`
(sflow's per-run id, from `SFLOW_WORKFLOW_OUTPUT_DIR`; inside the `/work` mount — no
dependence on sflow output-dir mounts):

- Trace: `.../vllm_server/q3vl_g894_b300x8_<mode>_..._.nsys-rep` (open in Nsight Systems).
- Server log: `.../vllm_server/vllm_serve.log`.
- Client log + report: `.../endpoint_benchmark/inference_endpoint.log`,
  `results/qwen3_vl_235b_a22b_profile_<mode>_g894_b300x8/`.

## Knobs (per-mode, in each `qwen3vl_config.yaml`)

`MODE`, `DELAY_ITERATIONS`, `MAX_ITERATIONS` are the tunable knobs (also encoded in the
nsys filename). Override at launch, e.g. `--set MAX_ITERATIONS=200`.
