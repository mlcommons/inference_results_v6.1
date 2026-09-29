# e2e-rag-qna (Offline) -- Intel 1-node-2S-Xeon6787P

E2E-RAG multi-hop question answering over the frozen FRAMES Wikipedia corpus:
iterative retrieve -> grade -> check-sufficiency -> answer, with query
decomposition, against a prebuilt FAISS-HNSW vector database.

## Reproduction

```bash
# servers (CPU container): embedding + reranker + 20B grader
bash scripts/servers/launch_servers.sh embed rerank 20b
# 120B query/sufficiency/answer model (GPU container)
bash scripts/servers/launch_server_120b.sh

bash scripts/run_qna_perf.sh          # performance run
bash scripts/run_qna_accuracy.sh      # accuracy run
bash scripts/run_compliance_test09.sh # TEST09 output-length compliance

# accuracy scoring (Llama-3.1-8B judge on its own server)
bash scripts/servers/launch_servers.sh judge
python3 -m evaluation.accuracy_eval_qna --log_dir <acc_run_dir> \
    --results_file <acc_run_dir>/results/results.json \
    --dataset_path ${RUN_DATASET}
```

## Accuracy

Reported accuracy is LLM-judge answer accuracy over the 824 FRAMES queries: a
Llama-3.1-8B judge scores each generated answer against the ground truth.
Gate: >= 33.95%.
