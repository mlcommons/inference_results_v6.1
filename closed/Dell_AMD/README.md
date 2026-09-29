# MLPerf Inference v6.1 AMD Optimized Implementations

This is a repository for joint submission of Dell and AMD using optimized implementations for the MLPerf Inference Benchmark. Please refer to the [AMD submission](https://github.com/mlcommons/inference_results_v6.1/tree/main/closed/AMD) for the scripts to reproduce this submission using the MI350P specific configs from the `src` directory from here. 

## Hardware

- **GPU:** AMD Instinct MI350P
- **System:** Dell PowerEdge XE7745

## Notes

> **gpt-oss-120b on MI350P:** Export the following variable before running the benchmark:
> ```bash
> export CU_NUM=256
> ```

> Follow DLRMv3 from this setup for MI350P
