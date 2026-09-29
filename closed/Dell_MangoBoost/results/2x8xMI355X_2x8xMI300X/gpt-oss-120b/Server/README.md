## Dell_MangoBoost MLPerf Inference v6.1 Submission

**Multi Region 4-Cluster Inference**: We connect four clusters in different regions to make a scalable inference, which is comprised of MI300X from MangoBoost cluster (in Korea), MI355X from Dell cluster, MI355X from TensorWave cluster and MI300X Azure cluster. The scaling achieves **97%**. 

**Performance requirement** All the result meets the GPT-OSS Offline and Server requirement of P99 TTFT < 3s, and P99 TPOT < 80ms.

**Accuracy**: All the Acuccracy also pass the GPT-OSS-120B threshold > 82.30% (99% of 83.13%). Compliance TEST07 and TEST09 also pass.

## Heterogenous 4-cluster Inference

In this submission, we connect 4 different clusters in different regions and make a joint inference. This cross-region cluster is comprised of a MI300X node from MangoBoost cluster in Korea, a MI355X node from Dell cluster, a MI355X node from TensorWave cluster, and a MI300X node from Azure cluster.

### 1. Setup

Please pull the Docker from our Docker Hub:
```bash
# on MI300X node
docker pull llmboost/mb-llmboost:mlperf-6.1-multi-cluster-mi300x

# on MI355X node
docker pull llmboost/mb-llmboost:mlperf-6.1-multi-cluster-mi355x
```

Then, run the docker container with the following commands:
```bash
# on MI300X node
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
  llmboost/mb-llmboost:mlperf-6.1-multi-cluster-mi300x

# on MI355X node
docker run --rm -it --name llmboost-multi-cluster \
  --network host --ipc host --privileged --group-add video \
  --cap-add=SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --shm-size=192g \
  --ulimit nofile=1048576:1048576 --ulimit memlock=-1:-1 \
  --device=/dev/kfd --device=/dev/dri -v /dev/infiniband:/dev/infiniband \
  -v <path to gpt-oss-120b model>:/models/models/mlperf_inference/gpt-oss-120b  \
  -v <path to dataset>:/models/models/mlperf_inference/data/gpt-oss-120b \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --entrypoint bash \
  llmboost/mb-llmboost:mlperf-6.1-multi-cluster-mi355x
```

### 2. Heterogeneous Cluster Server Scenario Benchmarking

Please on MI300X node, launch the docker and start the LLMBoost service by runming the command:
```bash
cd /workspace/apps/mlperf
python3 server.py --test_mode Server \
    --model_path "/models/models/mlperf_inference/gpt-oss-120b" \
    --accelerator_name mi300x \
    --load_balancing_mode auto
```
> **NOTE:** Both MI300X nodes share the same launch command.

Then, please on MI355X node, launch the docker and start the LLMBoost service by running the below command:
```bash
cd /workspace/apps/mlperf
python3 server.py --test_mode Server \
    --model_path "/models/models/mlperf_inference/gpt-oss-120b" \
    --accelerator_name mi355x \
    --load_balancing_mode auto
```
> **NOTE:** Both MI300X nodes share the same launch command.

Then, wait until all the nodes finish the intialization and listening on the port `0.0.0.0:8000` and `0.0.0.0:8001`.

### 2.1. Heterogeneous Cluster Server Performance Run
With the LLMBoost service listening on every nodes, you can start a separate terminal on MI355X node, and run the performance benchmark:

```bash
cd /workspace/apps/mlperf
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"Dell_MangoBoost_Multi_Clusters"}
python3 client.py \
    --model_name gpt-oss-120b \
    --test_mode Server \
    --user_conf conf/user_gpt-oss-120b_16x_mi355x_16x_mi300x.conf \
    --sut_server_addr "http://<node-mi355x-1 ip addr>:8000,http://<node-mi355x-2 ip addr>:8000,http://<node-mi300x-1 ip addr>:8000,http://<node-mi300x-2 ip addr>:8000" \
    --scheduler weighted_random \
    --scheduler_weights "81,82,18,16" \
    --parallel_requests 200 \
    --result_dir $SUBMISSION_DIR/results/gpt-oss-120b/$SYSTEM_NAME/Server/performance/run_1
```
> **NOTE:** We schedule the requests to each node based on pre-defined weights. Because of the hardware difference, the final weight we used is "81,82,18,16".


### 2.2. Heterogeneous Cluster Server Accuracy Run

First, please restart the inference service using commands in **Sec 2.**, with adding `--max_tokens 32768`. 

Then, run the accuracy test with the following commands:
```bash
cd /workspace/apps/mlperf
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"Dell_MangoBoost_Multi_Clusters"}
ACC_DIR=$SUBMISSION_DIR/results/gpt-oss-120b/$SYSTEM_NAME/Server/accuracy
python3 client.py \
    --model_name gpt-oss-120b \
    --test_mode Server \
    --accuracy_test \
    --user_conf conf/user_gpt-oss-120b_16x_mi355x_16x_mi300x.conf \
    --dataset_path /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_ref.parquet \
    --sut_server_addr "http://<node-mi355x-1 ip addr>:8000,http://<node-mi355x-2 ip addr>:8000,http://<node-mi300x-1 ip addr>:8000,http://<node-mi300x-2 ip addr>:8000" \
    --scheduler weighted_random \
    --scheduler_weights "81,82,18,16" \
    --parallel_requests 200 \
    --result_dir $ACC_DIR

# Score
cd /workspace/apps/mlperf
tools/eval_gpt_oss_accuracy.sh $ACC_DIR/mlperf_log_accuracy.json 2>&1 | tee $ACC_DIR/accuracy.txt
```

This command will output the accuracy in the end, and please make sure the score is above the constraint (> 82.30%) so that it can pass the submission checker.

### 2.3. Heterogeneous Cluster Serveer Compliance Tests

First, please restart the inference service using commands in **Sec 2**. 

Then, using the following command to run the audit test on multi-cluster:

```bash
# Test 07
cd /workspace/apps/mlperf               
REF=/lab-mlperf-inference/mlperf_inference
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=Dell_MangoBoost_Multi_Clusters
RES=$SUBMISSION_DIR/results/gpt-oss-120b/$SYSTEM_NAME
T7=$RES/Server/compliance_runs/TEST07
mkdir -p $T7

cp $REF/compliance/TEST07/gpt-oss-120b/audit.config .

python3 client.py \
    --model_name gpt-oss-120b \
    --test_mode Server \
    --user_conf conf/user_gpt-oss-120b_TEST07.conf \
    --dataset_path /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet \
    --sut_server_addr "http://<node-mi355x-1 ip addr>:8000,http://<node-mi355x-2 ip addr>:8000,http://<node-mi300x-1 ip addr>:8000,http://<node-mi300x-2 ip addr>:8000" \
    --scheduler weighted_random --scheduler_weights "81,82,18,16" \
    --parallel_requests 200 \
    --result_dir $T7

rm audit.config

# Score Test 07
cd $T7
python3 $REF/compliance/TEST07/run_verification.py \
    -c $T7 \
    -o $RES/Server/audit/compliance \
    --audit-config $REF/compliance/TEST07/gpt-oss-120b/audit.config \
    --accuracy-script "python3 $REF/language/gpt-oss-120b/eval_mlperf_accuracy.py \
        --mlperf-log {accuracy_log} \
        --reference-data /models/models/mlperf_inference/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet \
        --tokenizer /models/models/mlperf_inference/gpt-oss-120b"

# Test 09
cd /workspace/apps/mlperf
T9=$RES/Server/compliance_runs/TEST09
mkdir -p $T9
cp $REF/compliance/TEST09/gpt-oss-120b/audit.config .

python3 client.py \
    --model_name gpt-oss-120b \
    --test_mode Server \
    --user_conf conf/user_gpt-oss-120b_16x_mi355x_16x_mi300x.conf \
    --sut_server_addr "http://<node-mi355x-1 ip addr>:8000,http://<node-mi355x-2 ip addr>:8000,http://<node-mi300x-1 ip addr>:8000,http://<node-mi300x-2 ip addr>:8000" \
    --scheduler weighted_random --scheduler_weights "81,82,18,16" \
    --parallel_requests 200 \
    --result_dir $T9

rm audit.config

cd $T9
python3 $REF/compliance/TEST09/run_verification.py \
    -c $T9 \
    -o $RES/Server/audit/compliance \
    --audit-config $REF/compliance/TEST09/gpt-oss-120b/audit.config
```