# Custom System Names

If your MLPerf system name is different from a listed config name, for example
the listed config is `B300-SXM-270GBx8`, but your system has the same number of
GPUs and can use the same benchmark config, rename the config folder to your
system name.

For single-node Docker runs, pass the same system name to both the server and
harness launches:

```bash
SYSTEM_NAME=<your-system-name> make run_llm_server
SYSTEM_NAME=<your-system-name> make run_harness
```

For multi-node SLURM runs with nv-sflow, edit the `SYSTEM_NAME` variable in the
benchmark's `*_config_sflow.yaml` file. For example:

```yaml
variables:
  SYSTEM_NAME:
    value: <your-system-name>
```

Example config file:

```text
configs/deepseek_r1/B300-SXM-270GBx8/TRTLLM/Offline/deepseek_config_sflow.yaml
```

After this change, nv-sflow automatically passes the selected system name to the
harness during the run.
