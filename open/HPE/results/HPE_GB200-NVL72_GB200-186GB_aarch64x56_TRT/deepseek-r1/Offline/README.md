Run DeepSeek-R1 with the HPE nv-sflow configuration for the 56-GPU system:

```bash
sbatch scaleout/sflow/tools/run_deepseek_centml_performance_15nodes.sbatch
sbatch scaleout/sflow/tools/run_deepseek_centml_accuracy_15nodes.sbatch
sbatch scaleout/sflow/tools/run_deepseek_centml_test06_15nodes.sbatch
```

The SUT uses seven `dep8` replicas across fourteen four-GPU nodes.
