# AMD DLRM-v3 MI355X C1-on open submission

This MLPerf Inference v6.1 open result uses C1-on sliding-window HSTU
attention on eight AMD Instinct MI355X accelerators.

## Sliding-window attention

The required runner switch is `WINDOW=1`. It selects the C1-on branch in
`run_gold.sh`, which forwards `DLRM_HSTU_MAX_ATTN_LEN=1024` to the HSTU
module. Each causal query is therefore bounded to the most recent 1024
positions. `WINDOW=0` selects full-causal C1-off and was not used here.

The submitted operating stack also uses batch 64, `FUSE_EPILOGUE=1`,
`INFLIGHT=32`, `LASTLAYER_TARGETS_ONLY=0`, and a degree-5 polynomial SiLU
gate. Some retained historical launcher comments describe C1-on as unsuitable
for a closed submission; that restriction does not apply to this open result.

## LoadGen and seeds

All four result sets were produced with LoadGen 6.0.17 built from upstream
revision `a468775ea2dd421e12fa0770ec70b27d823f8c96`. The embedded v6.1
seed set is:

- QSL seed: `2085463073848966840`
- Sample-index seed: `2747215439041700203`
- Schedule seed: `16159082839903944936`

The exact LoadGen source used to build the installed wheel is included under
`src/dlrm-v3/loadgen`.

## Results

- Server PerformanceOnly: 16,194.39 queries/s; p99 47.163671 ms.
- Server AccuracyOnly: GAUC 0.7861629372250531.
- Offline PerformanceOnly: 16,363 samples/s.
- Offline AccuracyOnly: GAUC 0.7861666437936649.

The canonical `user.conf` sets Server target QPS to 16,200, Offline target QPS
to 17,000, and a 600,000 ms minimum performance duration for both scenarios.
