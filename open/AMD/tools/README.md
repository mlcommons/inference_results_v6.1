# Submission Tools Staging

This runner-owned folder stages AMD-side helper tools intended for the final
`closed/AMD/tools/` submission area.

Use this for lightweight AMD-side helper scripts that assemble, filter, or
validate the q12,200 DLRM-v3 submission payload before running the official
MLCommons submission tooling from a separate `mlcommons/inference` checkout.

Examples:

- copy/prune staged `src/`, `setup/`, `systems/`, and `results/` into a final
  `closed/AMD/...` output tree
- generate or validate checksums
- prepare lists of result files to keep/drop
- invoke MLCommons submission checker commands with the right arguments

Do not vendor the upstream MLCommons `tools/submission/` implementation here.
Reference it as an external tooling dependency instead.
