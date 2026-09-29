# e2e-rag-db (Offline) -- Intel 1-node-2S-Xeon6787P

E2E-RAG data-setup (vector database build) workload: parse the frozen Wikipedia
HTML corpus, chunk it, embed the passages with e5-base-v2, and build a FAISS
HNSW index.

## Reproduction

```bash
bash scripts/run_ingestion_perf.sh       # performance run
bash scripts/run_ingestion_accuracy.sh   # accuracy run + DB manifest check
```

## Accuracy

The reported accuracy is the DB manifest probe-query retrieval accuracy: the
mean top-K document-URL set overlap against the reference database over 50
fixed probe queries (see `ingestion/db_manifest.py`). Gate: >= 0.95.

The accuracy run additionally verifies file-processing success rate, the
database MD5 reported by the SUT, vector/docstore consistency, index
dimension, and the FAISS index parameters.
