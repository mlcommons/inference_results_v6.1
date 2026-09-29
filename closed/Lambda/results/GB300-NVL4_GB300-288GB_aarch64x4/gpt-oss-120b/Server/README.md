# GPT-OSS-120B Server (Lambda 4xGB300)

TensorRT-LLM endpoint + MLPerf LoadGen harness. 4 data-parallel replicas, one GB300 per replica.

Result: 54,323.41 completed tokens/s (41.50 completed samples/s at target_qps=42), VALID,
performance constraints satisfied. TTFT p99 = 260ms (limit 3s), TPOT p99 = 42.8ms (limit 80ms).
Accuracy: exact_match = 83.491 (threshold 82.2987 = 99% of 83.13). PASS.
Compliance: TEST07 PASS (61.111 vs threshold 60.698), TEST09 PASS (mean output tokens 1308.88,
within 1150.38-1406.02).

Reproduce:

Please review closed/Lambda/src/nv_mlpinf/benchmarks/gpt_oss_120b/README.md.

The runs were orchestrated by nv-sflow on a single-node Slurm cluster with Pyxis/Enroot
containers, which required these deltas to the stock GB300 4-GPU recipe:
`mpi: pmi2` instead of `pmix` (the container's OpenMPI 4.1.9a1 ships an internal PMIx 3.x client
that cannot reach the site PMIx 5.0.1 server), `container_writable: true`, an explicit
`#SBATCH --gres=gpu:4` (this sflow version does not emit one from `--gpus-per-node`), and a
per-replica `CUDA_VISIBLE_DEVICES` pin so the four `trtllm-serve` replicas do not all land on GPU 0.
