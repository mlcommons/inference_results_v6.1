# Benchmarking Infra Architecture

The benchmarking infrastructure is a single MPI world where worker ranks host
inference servers and the last rank hosts the LoadGen runner. LoadGen issues
timestamp-based query batches over sharded ZMQ, workers run inference, then
return results to LoadGen for reporting and MLPerf bookkeeping.

### Benchmarking Server-Client Architecture

```
                        MPI COMM_WORLD (world_size = N)
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Rank 0 .. Rank N-2 (Workers)                       Rank N-1 (LoadGen)       │
│                                                                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐      │
│  │ DLRMInferenceServer│  │ DLRMInferenceServer│  │ DLRMInferenceServer│      │
│  │     Rank 0         │  │     Rank 1         │  │     Rank N-2       │      │
│  │                    │  │                    │  │                    │      │
│  │ • ZMQ listener     │  │ • ZMQ listener     │  │ • ZMQ listener     │      │
│  │ • batch prep       │  │ • batch prep       │  │ • batch prep       │      │
│  │ • GPU inference    │  │ • GPU inference    │  │ • GPU inference    │      │
│  │ • send results     │  │ • send results     │  │ • send results     │      │
│  └─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘      │
│            │                       │                       │                 │
│            └───────────────────────┼───────────────────────┘                 │
│                                    │                                         │
│                              ◄─────┼─────►                                   │
│                                    │                                         │
│                    ┌───────────────▼───────────────┐                         │
│                    │    TestRunner (Rank N-1)      │                         │
│                    │                               │                         │
│                    │  • issue queries              │                         │
│                    │  • receive results            │                         │
│                    │  • LoadGen callbacks          │                         │
│                    └───────────────────────────────┘                         │
│                                                                              │
│                    (sharded ZMQ sockets per node)                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Inference Backend Architecture

```
                              MPI COMM_WORLD (72 Workers + 1 LoadGen)
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                    │
│   ╔═══════════════════════════════════════════════════════════════════════════════════════════╗    │
│   ║                    SHARDED EMBEDDING TABLE (Total M rows)                                 ║    │
│   ║   [Shard 0: 0..M/72] [Shard 1: M/72..2M/72] ... [Shard 71: 71M/72..M]                     ║    │
│   ╚═══════════════════════════════════════════════════════════════════════════════════════════╝    │
│          │                     │                              │                                    │
│          ▼                     ▼                              ▼                                    │
│   ┌─────────────────┐   ┌─────────────────┐           ┌─────────────────┐                          │
│   │ DLRMInfServer 0 │   │ DLRMInfServer 1 │    ...    │ DLRMInfServer 71│    Rank 72 (LoadGen)     │
│   │                 │   │                 │           │                 │                          │
│   │ ┌─────────────┐ │   │ ┌─────────────┐ │           │ ┌─────────────┐ │   ┌──────────────────┐   │
│   │ │Dense Weights│ │   │ │Dense Weights│ │           │ │Dense Weights│ │   │   TestRunner     │   │
│   │ │ (full copy) │ │   │ │ (full copy) │ │           │ │ (full copy) │ │   │                  │   │
│   │ └─────────────┘ │   │ └─────────────┘ │           │ └─────────────┘ │   │ • issue queries  │   │
│   │ ┌─────────────┐ │   │ ┌─────────────┐ │           │ ┌─────────────┐ │   │ • recv results   │   │
│   │ │ Emb Shard 0 │ │   │ │ Emb Shard 1 │ │           │ │ Emb Shard 71│ │   │ • LoadGen cb     │   │
│   │ │ (1/72 table)│ │   │ │ (1/72 table)│ │           │ │ (1/72 table)│ │   └────────┬─────────┘   │
│   │ └─────────────┘ │   │ └─────────────┘ │           │ └─────────────┘ │            │             │
│   │                 │   │                 │           │                 │            │             │
│   │   KJT Batch     │   │   KJT Batch     │           │   KJT Batch     │            │             │
│   │       │         │   │       │         │           │       │         │            │             │
│   │       ▼         │   │       ▼         │           │       ▼         │            │             │
│   │ ┌───────────┐   │   │ ┌───────────┐   │           │ ┌───────────┐   │            │             │
│   │ │ Lookup    │   │   │ │ Lookup    │   │           │ │ Lookup    │   │            │             │
│   │ │ local?────┼───┼───┼─┼───────────┼───┼───────────┼─┼───────────┼───┼────────────┘             │
│   │ │  yes→use  │◄──┼───┼─┼──►fetch───┼◄──┼───────────┼─┼──►fetch───┼◄──┘  (async, no sync)        │
│   │ │  no→fetch │   │   │ │  (async)  │   │           │ │  (async)  │   |                          │
│   │ └─────┬─────┘   │   │ └─────┬─────┘   │           │ └─────┬─────┘   |                          │
│   │       │         │   │       │         │           │       │         |                          │
│   │       ▼         │   │       ▼         │           │       ▼         |                          │
│   │  GPU Inference  │   │  GPU Inference  │           │  GPU Inference  |                          │
│   │       │         │   │       │         │           │       │         |                          │
│   └───────┼─────────┘   └───────┼─────────┘           └───────┼─────────┘                          │
│           │                     │                             │                                    │
│           └─────────────────────┴─────────────────────────────┘                                    │
│                                 │                                                                  │
│                                 ▼                                                                  │
│                    ┌────────────────────────┐                                                      │
│                    │   Results via ZMQ      │                                                      │
│                    │   (sharded sockets)    │                                                      │
│                    └────────────────────────┘                                                      │
│                                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

```

Key components:
- `benchmarks/run_benchmark.py` orchestrates ranks, model/dataset init, and LoadGen.
- `inference_harness/inference_server.py` implements worker-side request ingest and inference.
- `inference_harness/test_runner.py` implements LoadGen callbacks and result collection.

### Overall Execution Flow

The benchmarking process follows a three-stage execution flow synchronized via MPI barriers:

**Stage 1: Worker Initialization & Warmup**
- All worker ranks (ranks 0 to N-2) initialize `DLRMInferenceServer` instances
- Each worker loads the preprocessed dataset and model weights
- Workers perform warmup inference steps to initialize CUDA kernels and stabilize performance
- MPI barrier ensures all workers complete initialization before proceeding

**Stage 2: Communication Socket Warmup**
- LoadGen rank (rank N-1) initializes `TestRunner` and waits for workers to complete Stage 1
- LoadGen sends **50,000 test batches** to workers via sharded ZMQ sockets
- This warms up the multi-node communication infrastructure and measures baseline latency
- MPI barrier ensures communication warmup completes before starting MLPerf benchmark

**Stage 3: MLPerf LoadGen Benchmark**
- LoadGen starts the official MLPerf LoadGen test (`lg.StartTestWithLogSettings`)
- LoadGen issues queries according to the configured scenario (Server/Offline)
- Workers process inference requests and return results
- LoadGen collects results, calculates accuracy metrics, and generates MLPerf-compliant logs
- Final MPI barrier ensures clean shutdown of all ranks

This staged approach ensures consistent performance measurements by eliminating cold-start effects from model loading, CUDA initialization, and network communication setup.
