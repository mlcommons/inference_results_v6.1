## MangoBoost MLPerf Inference v6.1 Submission - GPT-OSS-120B


**Prefill Decode (P/D) disaggregated 2-node MI355X**: For the interactive scenario, we deploy a P/D setup and transfer the KV cache using MoriIO. The result meets the GPT-OSS interactive requirement of P99 TTFT < 2s, and P99 TPOT < 15ms.

--- 

## Guideline to Reproduce

### 1. Setup
Pull our mlperf docker from Docker Hub:
```bash
docker pull llmboost/mb-llmboost:mlperf-6.1
```

Run the docker on each node and go inside the bash of the docker container (*all the after commands are assumed to be executed inside the docker container*): 

```bash
docker run --rm -it --name llmboost-pd \
  --network host --ipc host --privileged --group-add video \
  --cap-add=SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --shm-size=192g \
  --ulimit nofile=1048576:1048576 --ulimit memlock=-1:-1 \
  --device=/dev/kfd --device=/dev/dri -v /dev/infiniband:/dev/infiniband \
  -v <path to gpt-oss-120b model>:/models/models/mlperf_inference/gpt-oss-120b  \
  -v <path to dataset>:/models/models/mlperf_inference/data/gpt-oss-120b \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --entrypoint bash \
  llmboost/mb-llmboost:mlperf-6.1
```

---
## 2. Interactive Scenario - Prefill Decode Disaggregation

Two nodes of MI355X needs to be prepared and connected by RDMA nics. For instance, 8x 400GB/s NICs per node in this submission. Suppose you have two nodes called NODE-A and NODE-B (doesn't matter which one is NODE-A or NODE-B).

### 2.1. Performance Run

On NODE-A, please run the below command to start the P/D proxy and also the laodgen benchmarking:
```bash
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
python harness.py --model_name gpt-oss-120b --test_mode Server --pd --interactive \
    --pd_bind <puclic ip addr of NODE-A>:36000 --pd_prefill 8 --pd_decode 8 \
    --warmup_requests 100 \
    --user_conf conf/user_gpt-oss-120b_16x_mi355x-interactive.conf \
    --result_dir $SUBMISSION_DIR/results/gpt-oss-120b/Interactive/performance/run_1
```

Start another terminal on NODE-A, and run the same docker container. Inside the docker container, please start the Prefiller using the below command:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role prefill --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --nics rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 --nics_per_worker 0 \
    --temperature 1.0 \
    --ib_gid_index 1 2>&1 | tee /workspace/apps/mlperf/engine_logs/pd_prefill.log
```
> **NOTE:** Please replace the rdma0,...,rdma7 according to your own machine.

Next, on NODE-B, please run the below command inside the docker container:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role decode --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --nics rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 --nics_per_worker 0 \
    --temperature 1.0 \
    --ib_gid_index 1 2>&1 | tee /workspace/apps/mlperf/engine_logs/pd_decoder.log
```
> **NOTE:** Please replace the rdma0,...,rdma7 according to your own machine.

Then, the Interactive performance benchmark will start automatically.

### 2.2. Accuracy Run

On NODE-A, run the below commands to start the P/D proxy, and then start the loadgen for accuracy run.:
```bash
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
export ACC_DIR=$SUBMISSION_DIR/results/gpt-oss-120b/Interactive/accuracy
cd /workspace/apps/mlperf
python harness.py --model_name gpt-oss-120b --test_mode Server --pd --interactive \
    --accuracy_test \
    --dataset_path /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_ref.parquet \
    --pd_bind <puclic ip addr of  NODE-A>:36000 --pd_prefill 8 --pd_decode 8 \
    --warmup_requests 8 \
    --user_conf conf/user_gpt-oss-120b_16x_mi355x-interactive.conf \
    --result_dir $ACC_DIR

# Score
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
export ACC_DIR=$SUBMISSION_DIR/results/gpt-oss-120b/Interactive/accuracy
MLPERF_INFERENCE_ROOT=/workspace/apps/mlperf/inference \
PYTHON3_PATH=$(which python) \
TOKENIZER=/models/models/mlperf_inference/gpt-oss-120b \
  tools/eval_gpt_oss_accuracy.sh \
    $ACC_DIR/mlperf_log_accuracy.json \
    /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_ref.parquet \
    | tee $ACC_DIR/accuracy.txt
```

Meanwhile, create a separate terminal on NODE-A, and start prefiller inside the docker container:
```bash
cd /workspace/apps/mlperf
python pd_worker.py --role prefill --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --nics rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 --nics_per_worker 0 \
    --temperature 1.0 --max_tokens 32768 \
    --ib_gid_index 1 2>&1 | tee /workspace/apps/mlperf/engine_logs/pd_prefill_acc.log
```

On NODE-B, start the decoder using the below command:
```bash
cd /workspace/apps/mlperf
python pd_worker.py --role decode --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --nics rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 --nics_per_worker 0 \
    --temperature 1.0 --max_tokens 32768 \
    --ib_gid_index 1 2>&1 | tee /workspace/apps/mlperf/engine_logs/pd_decode_acc.log
```

**Expected Result**: The expected accuracy should be greater than 82.30%.

### 2.3. Compliance Test07

On NODE-A, please run the commands:
```bash
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
export AUDIT_OUT=$SUBMISSION_DIR/results/gpt-oss-120b/Interactive/audit/compliance
export RUN=/workspace/apps/mlperf/compliance_runs/Interactive/TEST07

cp inference/compliance/TEST07/gpt-oss-120b/audit.config .

python harness.py --model_name gpt-oss-120b --test_mode Server --pd --interactive \
    --dataset_path /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet \
    --pd_bind <puclic ip addr of  NODE-A>:36000 --pd_prefill 8 --pd_decode 8 \
    --warmup_requests 8 \
    --user_conf conf/user_gpt-oss-120b_16x_mi355x-interactive.conf \
    --result_dir $RUN

grep -c "Found Audit Config file" $RUN/mlperf_log_detail.txt
rm audit.config

# Score
mkdir -p /workspace/apps/mlperf/compliance_runs/verify && cd $_
python /workspace/apps/mlperf/inference/compliance/TEST07/run_verification.py \
    -c $RUN \
    -o $AUDIT_OUT \
    --audit-config /workspace/apps/mlperf/inference/compliance/TEST07/gpt-oss-120b/audit.config \
    --accuracy-script "python3 /workspace/apps/mlperf/inference/language/gpt-oss-120b/eval_mlperf_accuracy.py \
        --mlperf-log {accuracy_log} \
        --reference-data /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet \
        --tokenizer /models/models/mlperf_inference/gpt-oss-120b"
```

Meanwhile, create a separate terminal on NODE-A, and start prefiller inside the docker container:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role prefill --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --nics rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 --nics_per_worker 0 \
    --temperature 1.0 \
    --ib_gid_index 1 2>&1 | tee /workspace/apps/mlperf/engine_logs/pd_prefill_compl.log
```

On the NODE-B, start decoder inside the docker container by running the following commands:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role decode --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --nics rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 --nics_per_worker 0 \
    --temperature 1.0 \
    --ib_gid_index 1 2>&1 | tee /workspace/apps/mlperf/engine_logs/pd_decode_compl.log
```

### 2.3. Compliance Test09


On NODE-A, please run the commands:
```bash
cd /workspace/apps/mlperf
export RUN9=/workspace/apps/mlperf/compliance_runs/Interactive/TEST09

cp inference/compliance/TEST09/gpt-oss-120b/audit.config .

python harness.py --model_name gpt-oss-120b --test_mode Server --pd --interactive \
    --pd_bind <puclic ip addr of  NODE-A>:36000 --pd_prefill 8 --pd_decode 8 \
    --warmup_requests 8 \
    --user_conf conf/user_gpt-oss-120b_16x_mi355x-interactive.conf \
    --result_dir $RUN9       

grep -c "Found Audit Config file" $RUN9/mlperf_log_detail.txt
rm audit.config

export SUBMISSION_DIR=/workspace/apps/mlperf/submission
export AUDIT_OUT=$SUBMISSION_DIR/results/gpt-oss-120b/Interactive/audit/compliance
export RUN9=/workspace/apps/mlperf/compliance_runs/Interactive/TEST09

mkdir -p /workspace/apps/mlperf/compliance_runs/verify
cd /workspace/apps/mlperf/compliance_runs/verify
python /workspace/apps/mlperf/inference/compliance/TEST09/run_verification.py \
    -c $RUN9 -o $AUDIT_OUT \
    --audit-config /workspace/apps/mlperf/inference/compliance/TEST09/gpt-oss-120b/audit.config
```

Meanwhile, create a separate terminal on NODE-A, and start prefiller inside the docker container:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role prefill --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --nics rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 --nics_per_worker 0 \
    --temperature 1.0 \
    --ib_gid_index 1 2>&1 | tee /workspace/apps/mlperf/engine_logs/pd_prefill_compl.log
```

On the NODE-B, start decoder inside the docker container by running the following commands:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role decode --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --nics rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 --nics_per_worker 0 \
    --temperature 1.0 \
    --ib_gid_index 1 2>&1 | tee /workspace/apps/mlperf/engine_logs/pd_decode_compl.log
```

---

## 3. Offline Scenario

Two nodes of MI355X needs to be prepared. Suppose you have two nodes called NODE-A and NODE-B (doesn't matter which one is NODE-A or NODE-B).

### 3.1. Performance Run

On NODE-A, please run the following commands inside the docker container:

```bash
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
python harness.py --model_name gpt-oss-120b --test_mode Offline \
    --remote_engines 16 --pd_bind <puclic ip addr of  NODE-A>:36000 \
    --warmup_requests 3 \
    --user_conf conf/user_gpt-oss-120b_16x_mi355x.conf \
    --result_dir $SUBMISSION_DIR/results/gpt-oss-120b/Offline/performance/run_1
```

Create another terminal on NODE-A. Then, run the following commands inside the docker container:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role aggregated --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --scenario Offline --accelerator_name mi355x --model_name gpt-oss-120b \
    --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 \
    2>&1 | tee engine_logs/agg_n101.log
```

Meanwhile, on NODE-B, please run the below commands inside the docker container:

```bash
python pd_worker.py --role aggregated --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --engine_base 8 --scenario Offline --accelerator_name mi355x --model_name gpt-oss-120b \
    --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 \
    2>&1 | tee engine_logs/agg_n102.log
```

### 3.2. Accuracy Run

On NODE-A, please run the below commands inside the docker container:

```bash
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
export ACC_DIR=$SUBMISSION_DIR/results/gpt-oss-120b/Offline/accuracy
python harness.py --model_name gpt-oss-120b --test_mode Offline \
    --accuracy_test \
    --dataset_path /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_ref.parquet \
    --remote_engines 16 --pd_bind <puclic ip addr of  NODE-A>:36000 \
    --warmup_requests 1 \
    --user_conf conf/user_gpt-oss-120b_16x_mi355x.conf \
    --result_dir $ACC_DIR

# Score
mkdir -p /workspace/apps/mlperf/compliance_runs/verify
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
export ACC_DIR=$SUBMISSION_DIR/results/gpt-oss-120b/Offline/accuracy
MLPERF_INFERENCE_ROOT=/workspace/apps/mlperf/inference \
PYTHON3_PATH=$(which python) \
TOKENIZER=/models/models/mlperf_inference/gpt-oss-120b \
  tools/eval_gpt_oss_accuracy.sh \
    $ACC_DIR/mlperf_log_accuracy.json \
    /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_ref.parquet \
    2> $SUBMISSION_DIR/results/gpt-oss-120b/Offline/eval_stderr.log \
    | tee $ACC_DIR/accuracy.txt
```

Meanwhile, on another terminal on NODE-A, run the following commands inside docker container:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role aggregated --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --scenario Offline --accelerator_name mi355x --model_name gpt-oss-120b \
    --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 --max_tokens 32768 \
    2>&1 | tee engine_logs/agg_acc_n101.log
```

On NODE-B, please meanwhile run the following commands:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role aggregated --sut <puclic ip addr of  NODE-A>:36000 --devices 8 \
    --engine_base 8 --scenario Offline --accelerator_name mi355x --model_name gpt-oss-120b \
    --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 --max_tokens 32768 \
    2>&1 | tee engine_logs/agg_acc_n102.log
```

### 3.3. Compliance Test07

On NODE-A, please run:
```bash
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
export AUDIT_OUT=$SUBMISSION_DIR/results/gpt-oss-120b/Offline/audit/compliance
export RUN7=/workspace/apps/mlperf/compliance_runs/Offline/TEST07

cp inference/compliance/TEST07/gpt-oss-120b/audit.config .
python harness.py --model_name gpt-oss-120b --test_mode Offline \
    --dataset_path /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet \
    --remote_engines 16 --pd_bind <puclic ip addr of  NODE-A>:20000 \
    --warmup_requests 3 \
    --user_conf conf/user_gpt-oss-120b_16x_mi355x-test07.conf \
    --result_dir $RUN7
rm audit.config

# Score
mkdir -p /workspace/apps/mlperf/compliance_runs/verify
cd /workspace/apps/mlperf/compliance_runs/verify

python /workspace/apps/mlperf/inference/compliance/TEST07/run_verification.py \
    -c $RUN7 -o $AUDIT_OUT \
    --audit-config /workspace/apps/mlperf/inference/compliance/TEST07/gpt-oss-120b/audit.config \
    --accuracy-script "python3 /workspace/apps/mlperf/inference/language/gpt-oss-120b/eval_mlperf_accuracy.py \
        --mlperf-log {accuracy_log} \
        --reference-data /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet \
        --tokenizer /models/models/mlperf_inference/gpt-oss-120b"
```

Meanwhile, in another terminal of NODE-A, please run the below commands inside the docker container:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role aggregated --sut <puclic ip addr of  NODE-A>:20000 --devices 8 \
    --scenario Offline --accelerator_name mi355x --model_name gpt-oss-120b \
    --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 \
    2>&1 | tee engine_logs/agg_compl_n101.log
```

On NODE-B, run the below commands:
```bash
cd /workspace/apps/mlperf
python pd_worker.py --role aggregated --sut <puclic ip addr of  NODE-A>:20000 --devices 8 \
    --engine_base 8 --scenario Offline --accelerator_name mi355x --model_name gpt-oss-120b \
    --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 \
    2>&1 | tee engine_logs/agg_compl_n101.log
```

### 3.3. Compliance Test09

On NODE-A, please run:
```bash
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
export AUDIT_OUT=$SUBMISSION_DIR/results/gpt-oss-120b/Offline/audit/compliance
export RUN7=/workspace/apps/mlperf/compliance_runs/Offline/TEST07
export RUN9=/workspace/apps/mlperf/compliance_runs/Offline/TEST09

cp inference/compliance/TEST09/gpt-oss-120b/audit.config .

python harness.py --model_name gpt-oss-120b --test_mode Offline \
    --remote_engines 16 --pd_bind 10.24.144.101:20000 \
    --warmup_requests 3 \
    --user_conf conf/user_gpt-oss-120b_16x_mi355x.conf \
    --result_dir $RUN9

grep -c "Found Audit Config file" $RUN9/mlperf_log_detail.txt
rm audit.config

python /workspace/apps/mlperf/inference/compliance/TEST09/run_verification.py \
    -c $RUN9 -o $AUDIT_OUT \
    --audit-config /workspace/apps/mlperf/inference/compliance/TEST09/gpt-oss-120b/audit.config
```

Meanwhile, in another terminal of NODE-A, please run the below commands inside the docker container:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role aggregated --sut <puclic ip addr of  NODE-A>:20000 --devices 8 \
    --scenario Offline --accelerator_name mi355x --model_name gpt-oss-120b \
    --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 \
    2>&1 | tee engine_logs/agg_compl_n101.log
```

On NODE-B, run the below commands:
```bash
cd /workspace/apps/mlperf
python pd_worker.py --role aggregated --sut <puclic ip addr of  NODE-A>:20000 --devices 8 \
    --engine_base 8 --scenario Offline --accelerator_name mi355x --model_name gpt-oss-120b \
    --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 \
    2>&1 | tee engine_logs/agg_compl_n101.log
```