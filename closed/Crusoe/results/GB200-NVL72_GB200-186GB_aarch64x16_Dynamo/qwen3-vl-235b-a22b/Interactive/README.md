# Qwen3-VL — GB200x16 Interactive, P/D-disaggregated (perf)

4-node × 4 GPUs (16 total) interactive-scenario perf run using vLLM+Dynamo
prefill/decode disaggregation: 4 prefill workers (TP2, 8 GPUs) + 8 decode
workers (TP1, 8 GPUs), a 50:50 prefill:decode GPU split (all 16 GPUs used).
Client drives the 8k shopify dataset (`shopify_product_catalogue_8k::q3vl`)
at `target_qps=40`, Poisson load.

**Note:** this submission was revised from an earlier `target_qps=43`
candidate after discovering the pass/fail bar is the **early-stopping**
p99 latency (must be < 1.5s), not the regular p99 shown throughout the
tuning history below. Re-checked against early-stopping p99, qps43
(1.516s) and every ratio5050 point at qps42 and above fail; qps40
(early-stopping p99 = 1.487s) is the highest-qps point across the full
autotune sweep that passes.

This topology and target_qps were found via a tuning campaign, not the
initial guess. Progression:

1. First validated config (6 prefill-TP2 + 4 decode-TP1, 75:25 split,
   adapted directly from NVIDIA's only published disaggregated Qwen3-VL
   recipe for GB300-NVL72 aarch64x72) passed at target_qps=16 (15.85 QPS
   achieved).
2. NVIDIA separately suggested a decode-heavier topology (5 prefill-TP2 +
   6 decode-TP1) estimated at ~50-60 qps; that exact topology needs 5
   Slurm nodes under this project's whole-node-per-role packing and
   doesn't fit our 4-node cluster, so 4 prefill-TP2 + 6 decode-TP1 (14/16
   GPUs) was used instead. A binary search on target_qps for this
   topology (qps50 FAIL -> qps45 FAIL -> qps40 FAIL -> qps20 PASS ->
   qps30 PASS -> qps35 PASS -> qps37 PASS -> qps38 PASS -> qps39 FAIL)
   converged on target_qps=38 (37.85 QPS achieved) as the max passing
   point for that topology.
3. A 50:50 prefill:decode ratio (4 prefill-TP2 + 8 decode-TP1, using the
   2 GPUs the previous topology left idle) was tested at target_qps=40 -
   the exact point where the 4:6 topology failed (p99=1507.49ms) - and
   **passed** (p99=1457.07ms, 2.9% margin), confirming the 50:50 ratio
   has more headroom than the decode-heavier one.
4. A binary search on target_qps for the 50:50 topology followed, tracked
   using regular (non-early-stopping) p99: qps40 (job 755) PASS
   (p99=1457.07ms) -> qps45 (job 757) FAIL (p99=1520.91ms) -> qps42
   (job 758) PASS (p99=1472.27ms) -> qps43 (job 759) PASS (p99=1481.04ms,
   1.26% margin). This qps43 result (job 759) was the originally
   submitted config, but it fails the actual pass bar once checked
   against **early-stopping** p99 (must be < 1.5s): qps43's
   early-stopping p99 is 1.516ms over. Re-checking early-stopping p99
   across the whole sweep (both topologies, both `nvidia_recipe` and
   `ratio5050` configs) shows qps40 (job 755, this config) is the highest
   passing point: early-stopping p99 = 1.487s, vs. 1.504s at qps42,
   1.516s at qps43, and 1.536s+ at qps44/45. Current submitted result:
   **39.85 QPS** (job 755), a ~2.5x throughput improvement over the
   original 75:25 topology's 15.85 QPS, at the same accuracy floor.

## Launch

From `closed/NVIDIA/`:

```bash
SROOT=configs/qwen3_vl_235b_a22b
RUN=q3vl_gb200x16_interactive_pd_disagg_$(date +%Y%m%d-%H%M%S)
OUT=scaleout/sflow_output/$RUN
ACCT=<your-slurm-account>

sflow batch \
  -f $SROOT/_shared/slurm_env.yaml \
  -f $SROOT/_shared/templates/vllm_dynamo_pd_disagg_serve_endpoints.yaml \
  -f $SROOT/GB200-NVL72_GB200-186GB_aarch64x16/VLLM/Interactive/qwen3vl_pd_disagg_config_ratio5050_qps40.yaml \
  --set WORK_DIR=$PWD \
  --set HF_CACHE_HOST_DIR=$HOME/.cache/huggingface \
  --set SLURM_ACCOUNT=$ACCT \
  --partition gb200 --account $ACCT --time 04:00:00 --nodes 4 \
  --output-dir $OUT --sbatch-path $OUT/sbatch.sh --submit
```

## Results

- Per-task logs: `$OUT/<jobid>-<workflow>-<ts>/{sflow.log,<task>/...}`
- Final QPS / TPS: `grep "QPS:" <jobid-dir>/sflow.log`
- Loadgen percentiles (TTFT, TPOT, end-to-end latency, both regular and
  early-stopping): `results_experiments/qwen3vl_interactive_autotune/ratio5050_qps40/report.txt`
  (job 755; full tuning history across both topologies in
  `/home/ubuntu/qwen3vl_tuning/nvidia_recipe_binary_search.log` and
  `/home/ubuntu/qwen3vl_tuning/followup_ratio_and_getzcopy_qps40.log`).
  Check `early_stopping_percentiles` in `performance/result_summary.json`
  against the pass bar, not `percentiles` (regular) - see note at top.

## Per-recipe overrides

`NUM_PREFILL_WORKERS`/`NUM_DECODE_WORKERS`/`PREFILL_TENSOR_PARALLEL_SIZE`/
`DECODE_TENSOR_PARALLEL_SIZE` in `qwen3vl_pd_disagg_config_ratio5050_qps40.yaml`
control the disaggregation topology (must resolve to whole Slurm nodes per
role - see that file's own header comment for the node-packing formula).
`target_qps` lives in `endpoint_qps40.yaml`.
