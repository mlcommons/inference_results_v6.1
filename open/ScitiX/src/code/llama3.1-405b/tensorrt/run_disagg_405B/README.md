# LLAMA3.1-405B Disaggregated Multi-Node SLURM Script

This SLURM script launches a disaggregated TensorRT-LLM inference setup for Llama3.1-405B across multiple nodes, separating context processing and generation workloads for optimal performance and resource utilization.

## Overview

The script implements a **disaggregated serving benchmarking** that contains the following component:
- **Context Servers**: Handle prompt processing and KV-cache generation
- **Generation Servers**: Handle token generation and sampling
- **Orchestrator**: Coordinates between context and generation workers
- **MLPerf Loadgen Measuring Client**: Using MLPerf loadgen to measure the backend throughput at latency


## Architecture

```
┌─ Node 0 ────────────────────┐  ┌─ Node 1 ────────────────────┐  ┌─ Node ... ────────────────────┐  ┌─ Node 13 ────────────────────┐
│ Context Server 0             │  │ Context Server 1             │  │ Context Server 2             │  │ Context Server 3             │
│ • TP=4, PP=1 (1 node)       │  │ • TP=4, PP=1 (1 node)       │  │ • TP=4, PP=1 (1 node)       │  │ • TP=4, PP=1 (1 node)       │
│ • Batch Size: 128           │  │ • Batch Size: 128           │  │ • Batch Size: 128           │  │ • Batch Size: 128           │
│ • Max Tokens: 4096          │  │ • Max Tokens: 4096          │  │ • Max Tokens: 4096          │  │ • Max Tokens: 4096          │
└─────────────────────────────┘  └─────────────────────────────┘  └─────────────────────────────┘  └─────────────────────────────┘
                    ↓                            ↓                            ↓                            ↓
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           Generation Server (Node 0)                                                              │
│                                       • TP=8, PP=1 (8 GPUs shared across 2 nodes nodes)                                            │
│                                       • Batch Size: 1024                                                                          │
│                                       • Max Tokens: 1024                                                                          │
│                                       • GPU Memory Fraction: 0.95                                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                              ↑
                                                    MLPerf Benchmark Client
                                                    (connects to server endpoint)
```

## Prerequisites

### 1. Hardware Requirements
- **18 nodes** with **4 GB200 GPUs each** (72 total GPUs)
- **SLURM cluster** with Enroot support
- **High-speed interconnect** for multi-node communication
- **Large shared storage** for model weights (~230GB for FP4 quantized model)

### 2. Software Requirements

#### Build Enroot Image
First, build the required enroot image using the pyxis instructions:

```bash
cd mlperf-inference/closed/NVIDIA/pyxis/
make -f Makefile.pyxis build_base_sqsh \
    ARCH=aarch64 \
    SLURM_MODE="srun" \
    SBATCH_PARTITION=<your slurm partition> \
    SBATCH_CONTAINER_SAVE=<path_to_save_sqsh_image> \
    INSTALL_TRTLLM=1 \
    INSTALL_LOADGEN=1 \
    SBATCH_ACCOUNT=<your slurm account> \
    MLPERF_SCRATCH_PATH=<your scratch path that stores models and datasets>
```

#### Download Model Checkpoint
Download the FP4 quantized model from HuggingFace:

1. **Install required tools**:
```bash
pip install huggingface_hub
```

2. **Download the model**:
```bash
huggingface-cli download nvidia/Llama-3.1-405B-Instruct-FP4 \
    --local-dir <MLPERF_SCRATCH_PATH>/models/Llama3.1-405B/fp4-quantized-modelopt/llama3.1-405b-instruct-hf-torch-fp4
```

**Required directory structure**:
```
<MLPERF_SCRATCH_PATH>/
└── models/
    └── Llama3.1-405B/
        └── fp4-quantized-modelopt/
            └── llama3.1-405b-instruct-hf-torch-fp4/
                ├── config.json
                ├── model-*.safetensors
                ├── tokenizer.json
                └── ...
```

### 3. Model Information
Based on the [NVIDIA Llama-3.1-405B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP4) model:
- **Parameters**: 405 billion parameters
- **Quantization**: FP4


## Configuration

### Step 1:
The script includes several environment variables that are set automatically, typically you don't need to change.

```bash
# Build environment configuration
export ENV="dev"
export BUILD_CONTEXT="aarch64-Grace"
export CUDA_VER="12.9"
export TRT_VER="10.11.0.33"
export MITTEN_HASH="82a930d962ce6bd8aed38cf185a2acfbbfd6b84b"
export INSTALL_RGAT_DEPS="0"
export INSTALL_TRTLLM=""
export INSTALL_LOADGEN=""

# Python path configuration
export PYTHONPATH=/work/.llm_aarch64/lib/python3.12/site-packages/:${PYTHONPATH}
```

### Step 2: Update Script Paths
Edit the script variables to match your environment:

```bash
# Step 1: Base image (automatically set)
export BASE_IMAGE="nvcr.io/nvidia/mlperf/mlperf-inference:mlpinf-v5.1-cuda12.9-pytorch25.05-ubuntu24.04-aarch64-Grace"

# Step 2: Scratch path for models and datasets
export MLPERF_SCRATCH_PATH="/your/path/to/mlperf_inference_storage"

# Step 3: Container image path (built via pyxis instructions)
CONTAINER_IMAGE=/your/path/to/trtllm_sqsh_image.sqsh

# Step 4: Container mounts (modify paths as needed)
CONTAINER_MOUNTS="/your/path/to/mlperf-inference/:/your/path/to/mlperf-inference/,/your/path/to/mlperf-inference/closed/NVIDIA/:/work,/your/path/to/mlperf_inference_storage_clone:/your/path/to/mlperf_inference_storage_clone"

# Step 5: Working directory
WORK_DIR=/your/path/to/mlperf-inference/closed/NVIDIA/code/llama3_1-405b/tensorrt/run_disagg_405B

# Step 6: Model directory (HF quantized FP4 model)
MODEL_DIR=/your/path/to/llama3.1-405b-instruct-hf-torch-fp4
```

### Step 3: Configure Disaggregated Setup
Current **optimal configuration** (best tokens/s/gpu):

```bash
# Context servers configuration
num_ctx_servers=14          # 14 context servers
ctx_tp_size=4              # Tensor parallelism = 4 GPUs per server
ctx_pp_size=1              # Pipeline parallelism = 1
ctx_batch_size=128         # Batch size for context processing
ctx_max_num_tokens=4096    # Max tokens per context batch

# Generation server configuration  
num_gen_servers=2          # 2 generation servers
gen_tp_size=8             # Tensor parallelism = 8 GPUs per server
gen_batch_size=1024        # Batch size for generation
gen_max_num_tokens=1024    # Max tokens per generation batch
gen_gpu_memory_fraction=0.95  # GPU memory allocation
```

**Total GPU Usage**: 72 GPUs (14 context servers × 4 GPUs + 2 generation servers × 8 GPUs)

### Step 4: SLURM Configuration
```bash
#SBATCH --job-name=mlperf_405B_test
#SBATCH --nodes=18              # 18 nodes
#SBATCH --ntasks=72            # 72 total tasks
#SBATCH --ntasks-per-node=4    # 4 tasks per node
#SBATCH --partition=<your partition>  # Update to your partition
#SBATCH --account=<your account> # Update to your account
#SBATCH --time=4:00:00         # 4 hour time limit
#SBATCH --container-remap-root
#SBATCH --container-mount-home
```

### Step 5: Server Configuration
The script includes specific server timeout settings:
- **Request timeout**: 7200 seconds
- **Response timeout**: 7200 seconds
- **Warm-up time**: 120 seconds

## How to Run

### 1. Submit the Job
```bash
sbatch disaggr_torch_405B_ptyche.slurm
```

### 2. Monitor Progress
```bash
# Check job status
squeue -u $USER

# Monitor main job log
tail -f disagg_405B_slurm_log.txt

# Monitor server and workers
tail -f ${WORK_DIR}/llama3_405B_torch_backend_disagg/output_server.log
tail -f ${WORK_DIR}/llama3_405B_torch_backend_disagg/output_workers.log
```

### 3. Check Benchmark Results
```bash
# Performance results
cat ${WORK_DIR}/llama3_405B_torch_backend_disagg/mlperf_benchmark_performance.log

# Accuracy results  
cat ${WORK_DIR}/llama3_405B_torch_backend_disagg/mlperf_benchmark_accuracy.log
```

## Performance Tuning

### Server Target QPS Configuration
For optimal performance, tune the server target QPS in:
```
closed/NVIDIA/configs/GB200-NVL72_GB200-NVL-186GB_aarch64x72/Interactive/llama3_1-405b.py
```

**Recommended scaling**:
- **72 GPUs**: `server_target_qps = 15.2`

## Script Execution Flow

1. **Container Initialization**: Start the enroot container across all nodes
2. **Configuration Generation**: Generate YAML configuration file using `gen_yaml.py`
3. **Worker Startup**: Launch context and generation workers using `start_worker.sh`
4. **Server Startup**: Start the disaggregated TensorRT-LLM server
5. **Warm-up Period**: Wait 120 seconds for workers and server to initialize
6. **Benchmark Execution**: Run both performance and accuracy benchmarks
7. **Result Collection**: Save results to log files

## Output Files

### Generated Configuration
- `${LOG_DIR}/config.yaml` - Auto-generated disaggregated server configuration

### Logs
- `disagg_405B_slurm_log.txt` - Main SLURM job output
- `${LOG_DIR}/output_server.log` - TensorRT-LLM server logs
- `${LOG_DIR}/output_workers.log` - Worker process logs
- `${LOG_DIR}/mlperf_benchmark_performance.log` - Performance benchmark results
- `${LOG_DIR}/mlperf_benchmark_accuracy.log` - Accuracy benchmark results
