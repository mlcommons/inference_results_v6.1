## Dell_MangoBoost MLPerf Inference v6.1 Submission

**Multi Region 4-Cluster Inference**: We connect four clusters in different regions to make a scalable inference, which is comprised of MI300X from MangoBoost cluster (in Korea), MI355X from Dell cluster, MI355X from TensorWave cluster and MI300X Azure cluster. The scaling achieves **97%**. 

**Performance requirement** All the result meets the GPT-OSS Offline and Server requirement of P99 TTFT < 3s, and P99 TPOT < 80ms.

**Accuracy**: All the Acuccracy also pass the GPT-OSS-120B threshold > 82.30% (99% of 83.13%). Compliance TEST07 and TEST09 also pass.


## Interactive Scenario - Intra-Node P/D Disaggregation

### 1. Setup

Pull our mlperf docker from Docker Hub:

```bash
docker pull llmboost/mb-llmboost:mlperf-6.1
```

Run the docker on each node and go inside the bash of the docker container (all the after commands are assumed to be executed inside the docker container):

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

### 2. Performance Run

Please run the below command to start the P/D proxy and also the laodgen benchmarking:

```bash
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
python harness.py --model_name gpt-oss-120b --test_mode Server --pd --interactive \
    --pd_bind 127.0.0.1:20002 --pd_prefill 4 --pd_decode 4 \
    --warmup_requests 100 \
    --user_conf conf/user_gpt-oss-120b_4p4d_mi355x-interactive.conf \
    --result_dir $SUBMISSION_DIR/results/gpt-oss-120b/Interactive/performance/run_1
```

Start another terminal, and run the same docker container. Inside the docker container, please start the Prefiller using the below command:

```bash
# prefill — engines 0..3 on GPU 0-3
cd /workspace/apps/mlperf
python pd_worker.py --role prefill --sut 127.0.0.1:20002 --devices 4 \
    --kv_backend xgmi \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 2>&1 | tee engine_logs/pd_prefill_intra.log
```

Next, run the below command inside the docker container in another terminal inside the container:

```bash
# decode — engines 0..3 on GPU 4-7
cd /workspace/apps/mlperf
python pd_worker.py --role decode --sut 127.0.0.1:20002 --devices 4 --device_base 4 \
    --kv_backend xgmi \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 2>&1 | tee engine_logs/pd_decode_intra.log
```

### 2. Accuracy Run

Inside the container, run the below commands to start the P/D proxy, and then start the loadgen for accuracy run:

```bash
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYS=8xMI355X_intra_pd
export ACC_DIR=$SUBMISSION_DIR/results/gpt-oss-120b/$SYS/Interactive/accuracy
CONF=conf/user_gpt-oss-120b_4p4d_mi355x-interactive.conf

python harness.py --model_name gpt-oss-120b --test_mode Server --pd --interactive \
    --accuracy_test \
    --dataset_path /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_ref.parquet \
    --pd_bind 127.0.0.1:22000 --pd_prefill 4 --pd_decode 4 \
    --warmup_requests 8 \
    --user_conf $CONF \
    --result_dir $ACC_DIR

# Score
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYS=8xMI355X_intra_pd
export ACC_DIR=$SUBMISSION_DIR/results/gpt-oss-120b/$SYS/Interactive/accuracy
CONF=conf/user_gpt-oss-120b_4p4d_mi355x-interactive.conf
MLPERF_INFERENCE_ROOT=/workspace/apps/mlperf/inference \
PYTHON3_PATH=$(which python) \
TOKENIZER=/models/models/mlperf_inference/gpt-oss-120b \
  tools/eval_gpt_oss_accuracy.sh \
    $ACC_DIR/mlperf_log_accuracy.json \
    /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_ref.parquet \
    2> $(dirname $ACC_DIR)/eval_stderr.log \
    | tee $ACC_DIR/accuracy.txt
```

Meanwhile, create a separate terminal, and start prefiller inside the docker container:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role prefill --sut 127.0.0.1:22000 --devices 4 \
    --kv_backend xgmi \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 --max_tokens 32768 \
    2>&1 | tee engine_logs/pd_prefill_intra_acc.l
```

Then, in another container, start the decoder:
```bash
cd /workspace/apps/mlperf
python pd_worker.py --role decode --sut 127.0.0.1:22000 --devices 4 --device_base 4 \
    --kv_backend xgmi \
    --accelerator_name mi355x --model_path /models/models/mlperf_inference/gpt-oss-120b \
    --temperature 1.0 --max_tokens 32768 \
    2>&1 | tee engine_logs/pd_decode_intra_acc.lo
```

**Expected Result**: The expected accuracy should be greater than 82.30%.

### 3. Compliance Test07

Please in the docker container run the below commands:
```bash
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYS=8xMI355X_intra_pd
export AUDIT_OUT=$SUBMISSION_DIR/results/gpt-oss-120b/$SYS/Interactive/audit/compliance
export RUN7=/workspace/apps/mlperf/compliance_runs/Interactive/TEST07
CONF=conf/user_gpt-oss-120b_4p4d_mi355x-interactive.conf

cp inference/compliance/TEST07/gpt-oss-120b/audit.config .
python harness.py --model_name gpt-oss-120b --test_mode Server --pd --interactive \
    --dataset_path /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet \
    --pd_bind 127.0.0.1:36000 --pd_prefill 4 --pd_decode 4 \
    --warmup_requests 8 --user_conf $CONF --result_dir $RUN7

grep -c "Found Audit Config file" $RUN7/mlperf_log_detail.txt   # >= 1
rm audit.config

mkdir -p /workspace/apps/mlperf/compliance_runs/verify && cd $_
python /workspace/apps/mlperf/inference/compliance/TEST07/run_verification.py \
    -c $RUN7 -o $AUDIT_OUT \
    --audit-config /workspace/apps/mlperf/inference/compliance/TEST07/gpt-oss-120b/audit.config \
    --accuracy-script "python3 /workspace/apps/mlperf/inference/language/gpt-oss-120b/eval_mlperf_accuracy.py \
        --mlperf-log {accuracy_log} \
        --reference-data /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet \
        --tokenizer /models/models/mlperf_inference/gpt-oss-120b"
```

Meanwhile, create a separate terminal and start prefiller inside the docker container:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role prefill --sut 127.0.0.1:36000 --devices 4 \
    --kv_backend xgmi --accelerator_name mi355x \
    --model_path /models/models/mlperf_inference/gpt-oss-120b --temperature 1.0 \
    2>&1 | tee engine_logs/pd_prefill_intra_compl.log
```

Inside the other terminal, start the decoder with the following commands:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role decode --sut 127.0.0.1:36000 --devices 4 --device_base 4 \
    --kv_backend xgmi --accelerator_name mi355x \
    --model_path /models/models/mlperf_inference/gpt-oss-120b --temperature 1.0 \
    2>&1 | tee engine_logs/pd_decode_intra_compl.log
```

### 4. Compliance Test09

Please in the docker container run the below commands:
```bash
cd /workspace/apps/mlperf
export SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYS=8xMI355X_intra_pd
export AUDIT_OUT=$SUBMISSION_DIR/results/gpt-oss-120b/$SYS/Interactive/audit/compliance
CONF=conf/user_gpt-oss-120b_4p4d_mi355x-interactive.conf
export RUN9=/workspace/apps/mlperf/compliance_runs/Interactive/TEST09

cp inference/compliance/TEST09/gpt-oss-120b/audit.config .
python harness.py --model_name gpt-oss-120b --test_mode Server --pd --interactive \
    --pd_bind 127.0.0.1:36000 --pd_prefill 4 --pd_decode 4 \
    --warmup_requests 8 --user_conf $CONF --result_dir $RUN9

grep -c "Found Audit Config file" $RUN9/mlperf_log_detail.txt
rm audit.config

mkdir -p /workspace/apps/mlperf/compliance_runs/verify && cd $_
python /workspace/apps/mlperf/inference/compliance/TEST09/run_verification.py \
    -c $RUN9 -o $AUDIT_OUT \
    --audit-config /workspace/apps/mlperf/inference/compliance/TEST09/gpt-oss-120b/audit.config
```

Meanwhile, create a separate terminal and start prefiller inside the docker container:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role prefill --sut 127.0.0.1:36000 --devices 4 \
    --kv_backend xgmi --accelerator_name mi355x \
    --model_path /models/models/mlperf_inference/gpt-oss-120b --temperature 1.0 \
    2>&1 | tee engine_logs/pd_prefill_intra_compl.log
```

Inside the other terminal, start the decoder with the following commands:

```bash
cd /workspace/apps/mlperf
python pd_worker.py --role decode --sut 127.0.0.1:36000 --devices 4 --device_base 4 \
    --kv_backend xgmi --accelerator_name mi355x \
    --model_path /models/models/mlperf_inference/gpt-oss-120b --temperature 1.0 \
    2>&1 | tee engine_logs/pd_decode_intra_compl.log
```