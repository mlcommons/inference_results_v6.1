# AMD MI355X DLRM-v3 open-division submission

This package contains AMD's MLPerf Inference v6.1 open-division DLRM-v3
results for one eight-GPU MI355X system. The implementation enables C1-on
HSTU sliding-window attention with a 1024-position causal window.

Submitted operating points:

- Server: target 16,200 queries/s, batch size 64.
- Offline: target 17,000 queries/s, batch size 64.
- Attention: `WINDOW=1`, selecting `DLRM_HSTU_MAX_ATTN_LEN=1024`.
- LoadGen: 6.0.17 at upstream revision `a468775ea2`, using v6.1 seeds.
- Gate: degree-5 polynomial SiLU approximation.

The canonical LoadGen user configuration is
`src/dlrm-v3/harness/benchmarks/user_mi355x8_nve_b64_qps16200_PROD10min_C1on_v61.conf`.
Byte-identical copies are included as each scenario's `user.conf`.

See `documentation/dlrm-v3/README.md` for result and C1-on details and
`setup/dlrm-v3/README.md` for exact reproduction commands. Compliance tests
such as TEST08 are not required for this open submission and are not included.
