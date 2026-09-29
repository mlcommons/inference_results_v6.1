# DLRM-v3 Submission Tools Payload

This folder contains runner-side helper tools for packaging and future final-tree
assembly/checking. The upstream MLCommons `tools/submission/` scripts are not
vendored here; run them from a separate `mlcommons/inference` checkout against the
generated final submission tree.

Current helper:

```bash
python submission/tools/dlrm-v3/assemble_submission.py \
  --output /path/to/submission-output \
  --clean
```

This generates `/path/to/submission-output/closed/AMD/...` from the runner-owned
`submission/` staging tree. Add `--skip-results` for a structure-only dry run.
When an `mlcommons/inference` checkout is available, pass
`--mlcommons-inference-root /path/to/mlcommons-inference --run-truncate
--run-checker` to run official submission tooling after assembly.
