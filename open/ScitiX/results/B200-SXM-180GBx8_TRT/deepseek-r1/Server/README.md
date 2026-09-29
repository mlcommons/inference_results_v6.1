Please review open/ScitiX/src/README.md.

For more details, please refer to `closed/NVIDIA/README.md`.
## B200 提交配置 override (相对默认)
- server_target_qps: 16
- MTP speculative_decoding: decoding_type=MTP, num_nextn_predict_layers=1, max_total_draft_tokens=1 (解TPOT墙, 差异化杠杆)
- warmup_iterations: 2, enable_chunked_context: True
- 结果: 59451 tok/s (VALID, TTFT/TPOT均过门槛)

## B200 提交配置 override (P1, 相对默认) — 61634 tok/s VALID, em 81.61
- server_target_qps: 16.75
- max_concurrency: 10240, kvcache_free_gpu_mem_frac: 0.95, cuda_graph_batch_sizes 扩到 1024
- adp_balancing wait/timeout: 2/6 ; OMPI_MCA_coll_ucc_enable: 0
- MTP: decoding_type=MTP, num_nextn_predict_layers=1, max_total_draft_tokens=1
- warmup_iterations: 2, enable_chunked_context: True, moe_backend: CUTEDSL
- 突破: conc10240顶服务率(q17不发散)+ adp2/6在q16.75压TTFT至1616ms(≤2000); 聚合部署无需PD
