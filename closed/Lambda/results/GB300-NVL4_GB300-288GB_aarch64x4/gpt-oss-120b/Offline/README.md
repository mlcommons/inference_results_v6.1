# GPT-OSS-120B Offline (Lambda 4xGB300)

TensorRT-LLM endpoint + MLPerf LoadGen harness. 4 data-parallel replicas, one GB300 per replica.

Result: 59,737.9 tokens/s (45.69 samples/s), VALID.
Accuracy: exact_match = 82.842 (threshold 82.2987 = 99% of 83.13). PASS.
Compliance: TEST07 PASS (61.515 vs threshold 60.698), TEST09 PASS (mean output tokens 1308.03,
within 1150.38-1406.02).

Reproduce:

Please review closed/Lambda/src/nv_mlpinf/benchmarks/gpt-oss-120b/README.md.

The runs were orchestrated by nv-sflow on a single-node Slurm cluster with Pyxis/Enroot
containers, which required these deltas to the stock GB300 4-GPU recipe:
`mpi: pmi2` instead of `pmix` (the container's OpenMPI 4.1.9a1 ships an internal PMIx 3.x client
that cannot reach the site PMIx 5.0.1 server), `container_writable: true`, an explicit
`#SBATCH --gres=gpu:4` (this sflow version does not emit one from `--gpus-per-node`), and a
per-replica `CUDA_VISIBLE_DEVICES` pin so the four `trtllm-serve` replicas do not all land on GPU 0.
