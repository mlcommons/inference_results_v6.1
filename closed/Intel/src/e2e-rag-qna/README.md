# E2E: RAG Benchmark

End-to-end retrieval-augmented generation benchmark for multi-hop question answering on the [FRAMES](https://huggingface.co/datasets/google/frames-benchmark) Wikipedia dataset.

**Key features:**
- MLPerf Loadgen integration for standardized benchmarking
- Dense vector retrieval with FAISS (HNSW/IVF/Flat indexing)
- ColBERTv2 reranking
- Iterative multi-shot retrieval with LLM-driven query decomposition
- Cross-vendor hardware support (Intel XPU, AMD GPU/ROCm, NVIDIA GPU/CUDA, CPU)
- Separate performance and accuracy testing modes

---

## Table of Contents

- [Deployment Topology](#deployment-topology)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Typical Workflow](#typical-workflow)
- [Step 1: Download Models and Data](#step-1-download-models-and-data-one-time)
- [Step 2: Build Vector Database](#step-2-build-vector-database-one-time-measured-operation)
- [Verify Vector Database](#verify-vector-database)
- [Step 3: Run Question-Answering Workload](#step-3-run-question-answering-workload)
- [Async Pipeline](#async-pipeline)
- [LLM & Retrieval Servers](#llm--retrieval-servers)
- [Prerequisites](#prerequisites)
- [License](#license)

---

## Deployment Topology

The workload is split across **two containers** so the large model runs on the
accelerator while everything else runs on the CPU host:

| Container | Base image | Runs |
|---|---|---|
| **GPU container** (`e2e-rag-gpu`) | `vllm_xpu:gpt-oss-ww25-rc0` | GPT-OSS-**120B** on Intel XPU |
| **CPU container** (`e2e-rag-cpu`) | `tiyengar:vllm_cpu` | GPT-OSS-**20B**, embedding, reranker, and the judge |

> Container **names** above are examples — pick your own. The **image** tags are
> the ones this project is validated against (`docker images` on the host).

**Server layout** (defaults from `config.default.sh`):

| Service | Model | Port | Container | Role |
|---|---|---|---|---|
| LLM (20B) | `gpt-oss-20b-mxfp4` | 8192 | CPU | Answer generation + document-relevance grading |
| LLM (120B) | `gpt-oss-120b-mxfp4` | 8123 | GPU (XPU, TP=4) | Query decomposition + sufficiency check |
| Judge | `Llama-3.1-8B-Instruct` | 8125 | CPU | Answer scoring (accuracy mode only) |
| Embedding | `e5-base-v2` (+ FAISS) | 8100 | CPU | Query embedding + vector search (async pipeline) |
| Rerank | `ColBERTv2.0` | 8101 | CPU | Passage reranking (async pipeline) |

On the CPU container the 20B / embedding / rerank services are NUMA-pinned to
disjoint core ranges (see `SERVER_*_CORES` in `config.default.sh`). The **judge
runs after the benchmark completes**, so it can safely reuse cores that the other
services are pinned to — those services are idle during scoring. Give the judge
its own cores only if you have spare RAM and want to overlap it with a run.

---

## Environment Setup

The two vLLM images above already contain vLLM and the accelerator runtime. Start
one container from each image, mounting this repo and your model/data directories.
Both use host networking (`--net=host`), so the service ports are reachable
directly on the host — no `-p` mapping needed.

```bash
# Common mounts (adjust to your host):
#   $DATA_DIR   → /data      models + datasets live here
#   $CODE_DIR   → /workspace  this repository
DATA_DIR=/data
CODE_DIR=$(pwd)

# GPU container — GPT-OSS-120B on Intel XPU
docker run --privileged -itd --name e2e-rag-gpu \
    -u root --ipc=host --net=host --cap-add=ALL \
    --device /dev/dri:/dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v /lib/modules:/lib/modules \
    -v $DATA_DIR:/data \
    -v $CODE_DIR:/workspace \
    -v $HOME:/host \
    --workdir /workspace --entrypoint /bin/bash \
    vllm_xpu:gpt-oss-ww25-rc0

# CPU container — GPT-OSS-20B + embedding + reranker + judge
docker run --privileged -itd --name e2e-rag-cpu \
    -u root --ipc=host --net=host --cap-add=ALL \
    -v $DATA_DIR:/data \
    -v $CODE_DIR:/workspace \
    -v $HOME:/host \
    --workdir /workspace \
    tiyengar:vllm_cpu

# Inside each container: install this repo's Python dependencies once
docker exec -it e2e-rag-gpu bash -c "cd /workspace && bash scripts/setup/setup.sh"
docker exec -it e2e-rag-cpu bash -c "cd /workspace && bash scripts/setup/setup.sh"
```

> These flags mirror the validated gpt-oss launch (`--privileged`, `--net=host`,
> `--ipc=host`, `/dev/dri` passthrough for the XPU). Container **names** are
> examples; the **image** tags are what this project is validated against. `/data`
> is where models and datasets are expected to live (see
> [Configuration](#configuration)).

---

## Configuration

All scripts are driven by two shell config files. Every value is guarded with
`${VAR:-default}`, and scripts source `config.sh` **before** `config.default.sh`,
so the resolution order per variable is (**first wins**):

1. **Exported shell env var** — highest priority.
2. **`config.sh`** — your machine-specific overrides (gitignored). Absolute
   `/data` model paths, server ports/cores, device settings.
3. **`config.default.sh`** — generic committed defaults (localhost endpoints,
   relative paths). You normally don't edit this. **Do not delete it** — every
   script sources it for baseline defaults.

**To override a value**, either copy the single line you care about from
`config.default.sh` into `config.sh` and edit it (keep the `${VAR:-...}` guard),
or export it in your shell:

```bash
# one-off override — beats both config files
RUN_PERF_COUNT=100 bash scripts/run_qna_accuracy.sh
```

A ready-to-edit `config.sh` is committed as `config.sh.example`; copy it and fill
in your host's absolute paths:

```bash
cp config.sh.example config.sh   # then edit for your host
```

Key variables (grouped by prefix in `config.default.sh`):

- **`INGESTION_*`** — database build: `INGESTION_DOC_DIR`, `INGESTION_DB`,
  `INGESTION_CHUNK_LEN` (768), `INGESTION_CHUNK_OVERLAP` (32),
  `INGESTION_DEVICE`, `INGESTION_MAX_WORKERS`.
- **`RUN_*`** — QA run: `RUN_DATA_DIR`, `RUN_DB_NAME`, `RUN_PERF_COUNT` (824),
  `RUN_MAX_WORKERS` (10), `RUN_PERF_CACHE_FILE`.
- **`INFERENCE_*`** — endpoints/models used by a run: `INFERENCE_LLM_URL` (20B),
  `INFERENCE_QUERY_URL` / `INFERENCE_SUFFICIENCY_URL` (120B),
  `INFERENCE_JUDGE_URL`, `INFERENCE_MAX_ITERATIONS` (5).
- **`SERVER_*`** — how each vLLM/retrieval server launches: model path, port,
  tensor-parallel size, GPU memory fraction, and core pinning.

---

## Typical Workflow

```
1. Download models + frozen corpus → 2. Build vector database → 3. Run QA workload
   (setup/download_dataset_and_models.sh)   (scripts/run_ingestion_perf.sh)   (scripts/run_qna_accuracy.sh)
```

### Step 1: Download models and data (one-time)

```bash
bash scripts/setup/download_dataset_and_models.sh
```

This downloads all required models and datasets from MLCommons storage (~283GB total)
and extracts the frozen document corpus (`docs.tar.gz`) into `doc_html/`:
- **FRAMES Dataset** (~674KB)
- **Frozen document corpus** `docs.tar.gz` → `doc_html/` (2515 HTML pages)
- **Embedding Model** e5-base-v2 (~2.2GB)
- **Reranker Model** ColBERTv2.0 (~1.4GB)
- **GPT-OSS-120B Model** (~196GB)
- **GPT-OSS-20B Model** (~83GB)

See [docs/MLCOMMONS_ASSETS.md](docs/MLCOMMONS_ASSETS.md) for individual download
commands and detailed model information. Set the resulting paths in `config.sh`
(`SERVER_*_MODEL`) — the defaults assume they live under `/data`.

> **Important — use the frozen corpus.** The benchmark ships a fixed Wikipedia
> snapshot as `docs.tar.gz`; the download script extracts it to `doc_html/`.
> Build your vector database from this corpus. Do **not** re-scrape Wikipedia
> with `ingestion.download_docs` — that fetches whatever revision is live today,
> which differs from the reference corpus in bytes, passage counts, and
> retrieval results, and will fail cross-system DB-manifest verification.
> `ingestion.download_docs` remains only for regenerating the snapshot itself
> (benchmark maintenance):
>
> ```bash
> # Snapshot regeneration only — NOT for submissions:
> python3 -m ingestion.download_docs --output_dir doc_html --format html --processes 30
> ```

### Step 2: Build vector database (one-time measured operation)

**Performance mode:**
```bash
bash scripts/run_ingestion_perf.sh
```

**Accuracy mode** (with verification):
```bash
bash scripts/run_ingestion_accuracy.sh
```

Override defaults via `config.sh` or inline env vars, e.g.:
```bash
INGESTION_DOC_DIR=doc_html \
INGESTION_DB=data/vector_html_hnsw_len768_ov32_word.db \
INGESTION_DEVICE=auto \
bash scripts/run_ingestion_perf.sh
```

**Output:** Creates `${INGESTION_DB}` and its `_data/` directory.

### Verify Vector Database

Because vendors build their own vector DB, we verify **behavioral equivalence**
to a reference DB rather than byte-identity. A DB is considered equivalent when
it uses the same embedding **model**, the same corpus + chunking + parsing, and
the same FAISS index parameters — even if the HTML/passages are in a different
order and the embeddings are numerically different.

The check validates:

- **passage count** and **embedding dimension**
- **FAISS index params/algorithm** (index type, metric, HNSW `M` / `efConstruction` / `efSearch`)
- **corpus set** — an order-independent hash of the passage texts (same HTML +
  chunking + parsing produce the same set; catches parser/chunking drift)
- **top-K retrieval overlap** vs reference queries — mean set-overlap of the
  top-K retrieved URLs must be ≥ threshold (default `0.90`); tolerant of rank
  shuffling from numerically different embeddings

**Compare two DBs directly on disk** (no manifest; reference from
`REFERENCE_DATABASE` in `config.sh`):
```bash
bash scripts/db/compare_dbs.sh CANDIDATE_DB [REFERENCE_DB] [OVERLAP_THRESHOLD]
```

**Manifest workflow** (portable — ship a small `.json.gz`, no vectors inside).
Reference builder writes a manifest; any vendor verifies against it:
```bash
# Reference side — writes DB_MANIFEST_V2 (from config) by default:
bash scripts/db/write_db_manifest_v2.sh [OUTPUT]

# Vendor side — verifies RUN_DATABASE against the manifest:
bash scripts/db/verify_db_manifest_v2.sh [MANIFEST] [OVERLAP_THRESHOLD]
```

Both scripts read `RUN_DATABASE`, `EMBEDDING_MODEL`, `RUN_DATASET`,
and the manifest path from `config.sh`, and print the manifest's SHA-256 so it's
provable which reference was used. Exit code `0` = equivalent, `1` = not.

### Step 3: Run question-answering workload

Start the servers first (see [LLM & Retrieval Servers](#llm--retrieval-servers)).

> Run the `scripts/run_*.sh` driver scripts (ingestion and QA) from **inside the
> CPU container** — they call the local embedding/rerank/judge stack that lives
> there. Only the 120B server runs on the GPU container.

**Performance mode** (replays cached LLM responses, no live inference):
```bash
bash scripts/run_qna_perf.sh
```
If `RUN_PERF_CACHE_FILE` exists it is replayed automatically.

**Accuracy mode** (live LLM inference + judge scoring):
```bash
bash scripts/run_qna_accuracy.sh
```

Endpoints and models come from the `INFERENCE_*` variables in `config.sh`.
**Output:** results under `${RUN_OUTPUT_DIR}` (default `output/`).

Both scripts use the **async pipelined SUT by default**, which needs the embed +
rerank services running (see [LLM & Retrieval Servers](#llm--retrieval-servers)).
Set `ASYNC_PIPELINE=0` to fall back to the sequential SUT.

---

## Async Pipeline

By default the QA workload uses the **sequential** SUT (one query at a time). The
**async pipeline** SUT instead drives many queries concurrently on a single event
loop, overlapping embedding, retrieval, reranking, and the multiple LLM calls each
query issues per iteration. On a busy server this keeps the 20B/120B vLLM batches
full and cuts wall-clock substantially.

Enable it by setting `ASYNC_PIPELINE=1` for the accuracy run:

```bash
ASYNC_PIPELINE=1 bash scripts/run_qna_accuracy.sh
```

This requires the **embedding + rerank HTTP services** to be up (they are only used
by the async path — the sequential SUT embeds/reranks in-process):

```bash
bash scripts/servers/start_pipeline_servers.sh   # embedding (8100) + rerank (8101)
```

**Async-only knobs** (env vars, read by `run_qna_accuracy.sh` when `ASYNC_PIPELINE=1`):

| Variable | Default | Meaning |
|---|---|---|
| `MAX_CONCURRENT_QUERIES` | `128` | Cap on queries in flight on the host. This is the **only** host-side throttle; it does **not** limit LLM requests to the server — each query issues several LLM calls per iteration, all uncapped and batched by `vllm serve`. |
| `EMBED_URL` | `http://127.0.0.1:8100` | Embedding + FAISS-search service. |
| `RERANK_URL` | `http://127.0.0.1:8101` | ColBERT rerank service. |
| `TRACE` | `${RUN_OUTPUT_DIR}/results/pipeline_trace.json.gz` | Timeline trace output path (see below). |

### Timeline trace

The async run writes a **Chrome Trace / Perfetto** timeline to the `TRACE` path
(`.json.gz`); open it in [ui.perfetto.dev](https://ui.perfetto.dev) to inspect
per-query spans, stage flows, and server in-flight counters. Async-only — a
sequential run ignores `TRACE`.

---

## LLM & Retrieval Servers

Launch each server with the scripts under `scripts/servers/`. They read config
(`config.sh` then `config.default.sh`), self-background under `nohup`, and poll
`/v1/models` (or `/health`) until ready. Each writes its log to the **repo root**:

| Script | Log file (repo root) |
|---|---|
| `launch_server_120b.sh` | `log-server-120b.log` |
| `launch_server_20b.sh` | `log-server-20b.log` |
| `launch_server_8b_judge.sh` / `_xpu.sh` | `log-server-8b-judge.log` / `-xpu.log` |
| `launch_server_embedding.sh` | `log-server-embedding.log` |
| `launch_server_rerank.sh` | `log-server-rerank.log` |

**On the GPU container** — GPT-OSS-120B (Intel XPU, TP=4, port 8123):
```bash
bash scripts/servers/launch_server_120b.sh
```

**On the CPU container** — 20B + embed + rerank + judge, all at once:
```bash
bash scripts/servers/launch_servers.sh cpu           # 20b(8192) + judge(8125) + embed(8100) + rerank(8101)
```

Or launch them individually:
```bash
bash scripts/servers/launch_server_20b.sh            # gpt-oss-20b, port 8192 (TP=4)
bash scripts/servers/launch_server_embedding.sh      # e5 embedder + FAISS index, port 8100
bash scripts/servers/launch_server_rerank.sh         # ColBERT rerank pool, port 8101
bash scripts/servers/launch_server_8b_judge.sh       # Llama-3.1-8B judge, port 8125
```

`launch_servers.sh` delegates to the individual `launch_server_*.sh` scripts
(same target names as `stop_servers.sh`): `20b 120b judge embed rerank cpu all`.
`cpu` = the CPU stack, `all` adds 120b. Both scripts print usage if run with no
target.

**Alternatives / helpers:**
- `launch_server_8b_judge_xpu.sh` — run the judge on an Intel XPU instead of CPU.
- `health_check.sh` — shared readiness poller sourced by the launchers.
- `stop_servers.sh` — stop servers 

To change a model path, port, tensor-parallel size, memory fraction, or core
pinning, edit the corresponding `SERVER_*` variables in `config.sh` rather than
the scripts.

---

## Prerequisites

### Required Models and Data

All required models and datasets are hosted on MLCommons storage. Download them
in one shot with `scripts/setup/download_dataset_and_models.sh`
([Step 1](#step-1-download-models-and-data-one-time)), or individually per
[docs/MLCOMMONS_ASSETS.md](docs/MLCOMMONS_ASSETS.md). Ensure sufficient disk space
(~283GB for the full model set).

### System Requirements

- **Disk**: ~283GB for the full model set; ~50GB for documents + vector DB
- **CPU-host RAM**: the CPU container serves the 20B (TP=4), embedding, reranker,
  and judge. With all four loaded this uses **~200GB** of host RAM (measured:
  ~205GB resident; the 20B's four tensor-parallel workers dominate). Plan for a
  large-memory host — a few hundred GB. The GPU container adds only ~30GB of host
  RAM (the 120B weights live in accelerator memory).

---

## License

Licensed under the Apache License, Version 2.0. See the per-file copyright headers
for details.
