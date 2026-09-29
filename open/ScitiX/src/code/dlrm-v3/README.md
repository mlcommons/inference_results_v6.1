# Generative Recommender Benchmarking Harness

This harness is designed for **MLPerf Inference 6.0 DLRM v3 Blackwell (B200, GB200) submission**.

## Overview

This project supports flexible GPU benchmarking configurations:
- **Single-node + multi-GPU**: B200 systems
- **Multi-node + multi-GPU**: GB200 systems

We use the same distributed benchmarking architecture in both cases: **worker ranks** run `DLRMInferenceServer`, and **the last rank** runs the `TestRunner` (LoadGen).

**Configuration Examples:**
- **B200x8**: 9 MPI ranks total (8 workers + 1 LoadGen as the last rank)
- **GB200 NVL72**: 73 MPI ranks total (72 workers + 1 LoadGen as the last rank)


## Getting Started

### Step 1: Download Model and Dataset

Download the dataset and model following [MLCommons' reference implementation](https://inference.mlcommons-storage.org/index.html).

**Model Locations (NVIDIA Internal):**
- **B200 (computelab):** `/home/mlperf_inference_storage_01/models/dlrmv3/trained_checkpoint`
  - *Recommended:* Copy the 1TB model to `/raid/data` for faster loading
- **GB200 (ptyche):** `/lustre/fsw/coreai_mlperf_inference/mlperf_inference_storage/models/dlrmv3/trained_checkpoint/`

**Dataset Locations (NVIDIA Internal):**
- **Raw Dataset:** `/home/mlperf_inference_storage_01/data/dlrmv3/data/streaming-100b/sampled_data`
- **Preprocessed Dataset:** `/home/mlperf_inference_storage_01/preprocessed_data/dlrmv3/test_export/preprocess_final_1`

### Step 2: Preprocess the Dataset

Run the preprocessing script to prepare the dataset for inference:

```bash
python tools/preprocess_data.py \
   --dataset-path $DATASET_PATH \
   --output-dir $OUTPUT_DIR \
   --dataset-percentage 1 \
   --use-multiprocessing
```

### Step 3: Build and Run Container

#### B200: Build Docker for NVIDIA Internal (x86_64)

```bash
# Initialize SSH agent and upload your private key so Docker can clone private repos
cd mlperf-inference/closed/NVIDIA
make prebuild BENCHMARK=dlrm USE_PREBUILT_IMAGES=1
```

#### B200: Build Docker for NVIDIA OEM Partner in MLPerf (x86_64)

```bash
# Initialize SSH agent and upload your private key so Docker can clone private repos
cd mlperf-inference/closed/NVIDIA

docker pull registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/dlrm_images:dlrmv3-release-x86_64
docker tag registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/dlrm_images:dlrmv3-release-x86_64 gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/dlrm_images:dlrmv3-release-x86_64
make prebuild BENCHMARK=dlrm USE_PREBUILT_IMAGES=1
```

#### GB200: Export Enroot for NVIDIA Internal or NVIDIA OEM Partner  (aarch64)

```bash
# Allocate a GB200 node
srun -A coreai_mlperf_inference -p batch -N1 --time=30 --pty bash
# For Nvidia OEM partner, your account and partition might be different, change accordingly


# Export the enroot sqsh image from GitLab registry
# Make sure to enable GitLab access for enroot with a credential file under ~/.config/enroot/.credentials
# .credentials file example line: 
# machine gitlab-master.nvidia.com login <user_id> password <access_token>

enroot import 'docker://gitlab-master.nvidia.com/mlpinf/mlperf-inference/dlrm_images:dlrmv3-release-aarch64-Grace'
enroot import 'docker://registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/dlrm_images:dlrmv3-release-aarch64-Grace'

# Test the sqsh initialization with a srun command
srun --mpi=pmix \
  --container-image=<Absolute path to mlpinf+mlperf-inference+dlrm_images+dlrmv3-release-aarch64-Grace.sqsh> \
  --container-mounts=<Absolute path to mlperf-inference repo>/mlperf-inference/closed/NVIDIA/code/dlrm-v3:/work \
  --container-mount-home \
  --container-workdir=/work \
  --no-container-remap-root \
  --nodes=1 \
  --ntasks=1 \
  --pty bash
```


### Step 4: Run Benchmarks

#### Step 4.1: B200 Single-Node Benchmarking

Inside the container, edit the environment variables in `B200_run_performance_harness.sh`:

```bash
DATASET_PATH="/home/mlperf_inference_storage_01/preprocessed_data/dlrmv3/preprocess_final_1/"
CHECKPOINT_PATH="/raid/data/zihaok_1/89/"
```

Then run the benchmark:

```bash
cd code/dlrm-v3

# Offline accuracy + performance run
bash benchmarks/B200_run_performance_harness.sh SCENARIO=Offline MODE=performance OUTPUT_DIR=<SystemName>/Offline/performance
bash benchmarks/B200_run_performance_harness.sh SCENARIO=Offline MODE=accuracy OUTPUT_DIR=<SystemName>/Offline/accuracy

# Server accuracy + performance run
bash benchmarks/B200_run_performance_harness.sh SCENARIO=Server MODE=performance OUTPUT_DIR=<SystemName>/Server/performance
bash benchmarks/B200_run_performance_harness.sh SCENARIO=Server MODE=accuracy OUTPUT_DIR=<SystemName>/Server/accuracy
```


#### Step 4.2: GB200 Multi-Node Benchmarking (via Slurm)

Once you successfully import the enroot image from GitLab, you can run multi-node experiments using the script located at `mlperf-inference/closed/NVIDIA/code/dlrm-v3/benchmarks/cluster_commands/GB200_run_performance_harness.sh`.

Before running, edit the following variables in the script for your environment:

```bash
# Slurm configuration
#SBATCH --account=coreai_mlperf_inference
#SBATCH --partition=36x2-a01r

# Dataset and checkpoint paths
DATASET_PATH="/lustre/fsw/coreai_mlperf_inference/mlperf_inference_storage/preprocessed_data/dlrmv3/preprocess_final_1"
CHECKPOINT_PATH="/lustre/fsw/coreai_mlperf_inference/mlperf_inference_storage/models/dlrmv3/trained_checkpoint"

# Container configuration
CONTAINER_IMAGE="/path/to/your/mlpinf+mlperf-inference+dlrm_images+dlrmv3-release-aarch64-Grace.sqsh"
CONTAINER_MOUNTS="/path/to/mounts"
```


#### Running the Benchmarks

**Example: 4-Node GB200 Configuration**

```bash
sbatch --nodes=4 --segment=4 \
  benchmarks/cluster_commands/GB200_run_performance_harness.sh \
  SCENARIO=Server \
  MODE=performance \
  OUTPUT_DIR=results/GB200x4/Server/performance \
  USER_CONF=./benchmarks/user_gb200x4.conf
```

**Example: NVL72 (18-Node) Result Collection**

Notice: due to NVL72's embedding table sharding, we can run set batch size to 128 in offline scenario for a little bit extra perf, for server scenario, we still stick with batch size 64

```bash
# Server scenario - Performance
sbatch --nodes=18 --segment=18 \
  benchmarks/cluster_commands/GB200_run_performance_harness.sh \
  SCENARIO=Server \
  MODE=performance \
  OUTPUT_DIR=GB200NVL72/Server/performance \
  USER_CONF=./benchmarks/user_gb200x18.conf

# Server scenario - Accuracy
sbatch --nodes=18 --segment=18 \
  benchmarks/cluster_commands/GB200_run_performance_harness.sh \
  SCENARIO=Server \
  MODE=accuracy \
  OUTPUT_DIR=GB200NVL72/Server/accuracy \
  USER_CONF=./benchmarks/user_gb200x18.conf

# Offline scenario - Performance
sbatch --nodes=18 --segment=18 \
  benchmarks/cluster_commands/GB200_run_performance_harness.sh \
  SCENARIO=Offline \
  MODE=performance \
  OUTPUT_DIR=GB200NVL72/Offline/performance \
  USER_CONF=./benchmarks/user_gb200x18.conf

# Offline scenario - Accuracy
sbatch --nodes=18 --segment=18 \
  benchmarks/cluster_commands/GB200_run_performance_harness.sh \
  SCENARIO=Offline \
  MODE=accuracy \
  OUTPUT_DIR=GB200NVL72/Offline/accuracy \
  USER_CONF=./benchmarks/user_gb200x18.conf
```

## Compliance Test Integration and Result Folder Preparation

Following the guideline of [TEST08 compliance test](https://github.com/mlcommons/inference/tree/master/compliance/TEST08).

### Setup

Inside the container, copy the audit config to the current directory:

⚠️ **Warning:** After the audit test is run, if you want to rerun performance or accuracy mode, please remove the `audit.config` file, otherwise you will always run in audit mode.
```bash
cp /opt/inference/compliance/TEST08/dlrm-v3/audit.config mlperf-inference/closed/NVIDIA/code/dlrm-v3
```

### Offline Compliance

1. Run Offline compliance and dump result to it's own folder:

```bash
bash benchmarks/B200_run_performance_harness.sh SCENARIO=Offline MODE=performance OUTPUT_DIR=<SystemName>/Offline/audit
```

2. Run the verification script:

```bash
python /opt/inference/compliance/TEST08/run_verification.py \
    --reference_accuracy <SystemName>/Offline/accuracy/mlperf_log_accuracy.json \
    --test_accuracy=<SystemName>/Offline/audit/mlperf_log_accuracy.json &> <SystemName>/Offline/audit/verify_accuracy.txt
```

Expected output:

```
Verifying accuracy. This might take a while...
Reading accuracy mode results...
Reading performance mode results...

num_acc_log_entries = 349823
num_perf_log_entries = 4039
num_matched = 4039
num_unmatched = 0
num_ne_mismatch = 0
tolerance = 0.10%

TEST PASS
TEST08 verification complete
```

### Server Compliance

1. Run Server compliance:

```bash
bash benchmarks/B200_run_performance_harness.sh SCENARIO=Server MODE=performance OUTPUT_DIR=<SystemName>/Server/audit
```

2. Run the verification script:

```bash
python /opt/inference/compliance/TEST08/run_verification.py \
    --reference_accuracy <SystemName>/Server/accuracy/mlperf_log_accuracy.json \
    --test_accuracy=<SystemName>/Server/audit/mlperf_log_accuracy.json &> <SystemName>/Server/audit/verify_accuracy.txt
```

## Result Folder Preparation For Submission:

please follow the guideline in README_submitting.md



## Tips for Development and Testing

1. **Faster testing with smaller datasets**: During the bring-up phase, you can generate a smaller version of the dataset by controlling `--dataset-percentage` in `preprocess_data.py`. Then in the run script, specify `DATASET_PERCENTAGE` along with `DATASET_PATH` to load the smaller dataset, saving significant time during pipeline validation.

2. **Skip model loading for initial development**: Since the model size is 1TB, loading time can be a significant overhead during pipeline cleaning. For initial code bring-up, you can temporarily comment out `load_model_dense()` and `load_model_sparse()` functions in `inference_server.py`. Once the code bring-up is complete, add these back for official submission.


