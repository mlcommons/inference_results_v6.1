# Submission Systems Staging

This runner-owned folder stages system descriptions intended for the final
`closed/AMD/systems/` submission area.

For the q12,200 MI355X GOLD run, the draft system entry is:

```text
8xMI355X_2xEPYC_9575F.json
```

Before final submission, confirm or update:

- system name
- submitter/division/status fields required by the MLCommons checker
- 8x MI355X GPU configuration
- CPU/socket/core/SMT information
- memory/storage
- host OS/kernel/amdgpu/KMD
- container image digest and software stack

Do not leave required fields as `TBD` in the final generated submission tree.
