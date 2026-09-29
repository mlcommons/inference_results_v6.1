# Source

Code that builds the engine and drives the harness for the two submitted llama3.1-8b results.

## llama3.1-8b/  (TensorRT-LLM 1.0.0 FP8; MLCommons endpoints harness)

- `build_official_calib.py` -- materializes the calibration `train.jsonl` from the official
  id list. **Must be run before the pipeline**, which exits `CALIB_MISSING` (rc 5) without it.
  It reads `/cal/calibration-list.txt` and writes `/work/mlperf_l31_calib/train.jsonl`, so both
  paths must be bound and `/work` must be the same `$WORK` the pipeline uses. Neither sbatch
  invokes it or binds `/cal`. It also needs **network access** (`load_dataset`), unlike the
  benchmark stages, which run offline-pinned. Invocation and rationale are in
  `documentation/calibration.md`.
- `l31_8b_fp8_pipeline.sbatch` -- Offline: ModelOpt FP8 (W8A8) post-training quantization
  against the official MLPerf calibration set, `trtllm-build`, `trtllm-serve`, then the
  benchmark. Also writes the `kv_cache_config` YAML that disables KV-cache block reuse.
  Stages 1 and 2 are skipped when their artifacts exist, so a from-scratch reproduction needs an
  empty `$WORK`. `EPOCHS=3` is the setting used for the submitted Offline result.
- `l31_8b_server_acc.sbatch` -- Server: serves the cached engine and runs one online pass at
  `target_qps = 58` carrying both the accuracy and performance datasets. This produced the
  submitted Server result.
- `l31_8b_server_sweep.sbatch` -- performance-only `target_qps` sweep. Exploratory; it did not
  produce a submitted result.
- `config/staged/` -- the two endpoints benchmark configs the scripts read, as inputs. Locally
  authored, not upstream MLCommons examples. Neither script reads them from here: both read
  `$BASE/endpoints/examples/05_Llama3.1-8B_Example/`, so they must be copied there first. See
  `config/README.md` for the staging step and for the fields the scripts patch at launch.

The LoadGen-facing client is the unmodified MLCommons `inference-endpoints` package
(github.com/mlcommons/endpoints @ `e060c0e`); we supply configuration, not client code. The
effective configuration for each result is that row's `<scenario>/config.yaml`.

Each submitted row maps to one script and one config:

| Row | script | staged config | effective config |
|---|---|---|---|
| Offline | `l31_8b_fp8_pipeline.sbatch` (`EPOCHS=3`) | `config/staged/submission_llama3_8b_offline.yaml` | `Offline/config.yaml` |
| Server | `l31_8b_server_acc.sbatch` | `config/staged/submission_llama3_8b_server_q58.yaml` | `Server/config.yaml` |

## Paths

The sbatch scripts carry absolute paths for this cluster (including SLURM account, partition
and QoS) and set `BASE`/`WORK` at the top; a reviewer reproducing them must repoint those. The
scripts reference the harness files from `$WORK`, which is where they are staged at run time.

These are sanitized copies: internal comparison constants used during tuning were removed from the
console output, and explanatory comments were added. No change alters the measured configuration.

## Compliance note on KV-cache block reuse

TensorRT-LLM 1.0.0 enables prefix-KV sharing across requests by default
(`enable_block_reuse=True`, `enable_partial_reuse=True`). That is cross-query caching, which the
inference rules prohibit. **Both submitted results pin all three reuse flags false** via an
`--extra_llm_api_options` YAML written by the job:

```yaml
kv_cache_config:
  enable_block_reuse: false
  enable_partial_reuse: false
  copy_on_partial_reuse: false
  free_gpu_memory_fraction: 0.9
```

Note that `--extra_llm_api_options` **replaces** the whole `kv_cache_config` object rather than
merging into it, so `free_gpu_memory_fraction` lives in the same YAML. Setting it on the command
line instead would silently restore the reuse defaults.

Measured cost of disabling reuse, from a Server A/B at the same operating point (q58):
7,379.17 -> 7,373.00 tokens/s, **-0.08%**.

## Operational constraints when reproducing these runs

Three environment constraints matter, and the scripts handle all three:

1. **Two `trtllm-serve` processes cannot share a node.** TensorRT-LLM binds a fixed MPI-proxy port
   (`127.0.0.1:10012`), so concurrent servers on one host collide. Run the two scenarios on
   different nodes, or sequentially. The cleanup trap is scoped to the job's own port so it cannot
   terminate a co-located job belonging to the same user.
2. **The server port must sit below the kernel ephemeral range.** On this cluster
   `ip_local_port_range` is 32768-60999; a listener placed inside that window can lose its port to
   another process's outbound connection. The scripts choose `20000 + JOBID % 12000` and probe for
   a free port.
3. **The client connection pool must be bounded.** With `max_connections: -1` the endpoints client
   sets its pool ceiling to the entire ephemeral-port budget and pre-establishes a quarter of it,
   which exhausts the range on multi-epoch runs and surfaces as
   `OSError: [Errno 99] Cannot assign requested address`. The scripts set `max_connections: 2048`
   and `warmup_connections: 256` -- four times the server's 512 `max_batch_size`, so the bound
   cannot throttle the measurement.
