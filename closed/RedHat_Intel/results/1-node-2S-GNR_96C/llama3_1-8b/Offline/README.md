# Llama3.1-8B Inference on Intel(R) Xeon(R) CPU

## LEGAL DISCLAIMER
To the extent that any data, datasets, or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality. By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license. 

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets, or models. 

## Supported Hardware
This workload supports the following hardware configurations:
- 2x    Intel(R) Xeon(R) 6980P   (128 cores/socket)
- 2x/4x Intel(R) Xeon(R) 6787P   ( 86 cores/socket)
- 1x/2x Intel(R) Xeon(R) 6987P-C (120 cores/socket)

## Launch the Docker Image
Set the directories on the host system where model, dataset, and log files will reside. These locations will retain model and data content between Docker sessions.
```
export DATA_DIR="${DATA_DIR:-${PWD}/data}"
export MODEL_DIR="${MODEL_DIR:-${PWD}/model}"
export LOG_DIR="${LOG_DIR:-${PWD}/logs}"
```

In the Host OS environment, run the following after setting the proper Docker image name. If the Docker image is not on the system already, it will be retrieved from the registry.

If retrieving the model or dataset, ensure any necessary proxy settings are run inside the container.
```
export DOCKER_IMAGE=intel/mlperf:mlperf-inference-6.1-llama3.1-8b_cpu
MOUNT_ARGS="-v ${MODEL_DIR}:/model -v ${DATA_DIR}:/data -v ${LOG_DIR}:/logs"
ENV_VARS="-e http_proxy=${http_proxy} -e https_proxy=${https_proxy} -e no_proxy=${no_proxy}"

docker run --privileged -it --rm --ipc=host --net=host --cap-add=ALL \
        ${ENV_VARS} ${MOUNT_ARGS} \
        --workdir /workspace \
        ${DOCKER_IMAGE} /bin/bash
```

## Download Resources
Calibrated models and datasets directly from MLCommons using the following command.
NOTE: For large resources, MLCommons may implement a verification service such as Cloudflare before the download begins. Follow the unique browser link displayed in the CLI output for further instructions.
```
bash download_resources.sh
```

## Run Benchmark
Run these steps inside the Docker container. For an MLPerf submission run - including log structure, compliance runs, and submission checking - see the next section: 'MLPerf Submission Run.' The results of these benchmark runs should match the MLPerf results, but with less structural overhead. After a successful run, results may be found at: '/workspace/run_output'.

Performance
```
SCENARIO=Offline MODE=Performance bash run_benchmark.sh
SCENARIO=Server  MODE=Performance bash run_benchmark.sh
```
Accuracy
```
SCENARIO=Offline MODE=Accuracy    bash run_benchmark.sh
SCENARIO=Server  MODE=Accuracy    bash run_benchmark.sh
```

## MLPerf Submission Run
Run these steps inside the Docker container. Select the appropriate scenario.

Performance
```
SCENARIO=Offline MODE=Performance MLPERF_STAGE=True bash run_benchmark.sh
SCENARIO=Server  MODE=Performance MLPERF_STAGE=True bash run_benchmark.sh
```
Accuracy
```
SCENARIO=Offline MODE=Accuracy    MLPERF_STAGE=True bash run_benchmark.sh
SCENARIO=Server  MODE=Accuracy    MLPERF_STAGE=True bash run_benchmark.sh
```
Compliance
Note: Run this step inside the Docker container. After the benchmark scenarios have been run and results exist in {LOG_DIR}/results, run this step to complete compliance runs.
```
SCENARIO=Offline MODE=Compliance  MLPERF_STAGE=True bash run_benchmark.sh
SCENARIO=Server  MODE=Compliance  MLPERF_STAGE=True bash run_benchmark.sh
```

## MLPerf Submission Checker
Run this step inside the Docker container. The following script will perform accuracy log truncation and run the submission checker on the contents of {LOG_DIR}. Ensure the submission content has been populated before running. The original content of ${LOG_DIR} is not modified. Ensure the VENDOR and SYSTEM tags reflect those used when preparing the previous results (see ${LOG_DIR}/systems).
```
VENDOR=OEM SYSTEM=1-node-2S-GNR_96C bash code/prepare_mlperf_submission.sh
```
