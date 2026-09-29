# R-GAT Inference on CPU

## LEGAL DISCLAIMER
To the extent that any data, datasets, or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality. By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license. 

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets, or models. 

## Launch the Docker Image
Set the directories on the host system where dataset and log files will reside. These locations will retain model and data content between Docker sessions. Unlike similar containers, the model files are embedded, so no additional directories should be set for them.
```
export DATA_DIR="${DATA_DIR:-${PWD}/data}"
export LOG_DIR="${LOG_DIR:-${PWD}/logs}"
```

In the Host OS environment, run the following after setting the proper Docker image name. If the Docker image is not on the system already, it will be retrieved from the registry.

If retrieving the model or dataset, ensure any necessary proxy settings are run inside the container.
```
export DOCKER_IMAGE=keithachornintel/mlperf:mlperf-inference-6.1-rgat_cpu-r1
MOUNT_ARGS="-v ${DATA_DIR}:/data -v ${LOG_DIR}:/logs"
ENV_VARS="-e http_proxy=${http_proxy} -e https_proxy=${https_proxy} -e no_proxy=${no_proxy}"

docker run --privileged -it --rm --ipc=host --net=host --cap-add=ALL \
        ${ENV_VARS} ${MOUNT_ARGS} \
        --workdir /workspace \
        ${DOCKER_IMAGE} /bin/bash
```

## Prepare workload resources [one-time operations]
Download the dataset: Run this step inside the Docker container.  This operation will preserve the dataset on the host system using the volume mapping above.
NOTE: This is a very time-intensive and storage-intensive process (over 12h runtime and 2TB of storage temporarily needed).  Once completed, be sure to preserve the dataset to avoid repeating this step.
```
bash download_resources.sh
```

## Run Benchmark
Run these steps inside the Docker container. For an MLPerf submission run - including log structure, compliance runs, and submission checking - see the next section: 'MLPerf Submission Run.'
NOTE: This workload performs best if the system can have 1.5TB Memory. In cases where there isn't enough memory, please expect up to 4% run-to-run variation.

Performance:
```
SCENARIO=Offline MODE=Performance bash run_benchmark.sh
```
Accuracy:
```
SCENARIO=Offline MODE=Accuracy    bash run_benchmark.sh
```

## MLPerf Submission Run
Run these steps inside the Docker container. Select the appropriate scenario.

Performance:
```
SCENARIO=Offline MODE=Performance MLPERF_STAGE=True bash run_benchmark.sh
```
Accuracy:
```
SCENARIO=Offline MODE=Accuracy    MLPERF_STAGE=True bash run_benchmark.sh
```
Compliance:
Note: Run this step inside the Docker container. After the benchmark scenarios have been run and results exist in {LOG_DIR}/results, run this step to complete compliance runs.
```
SCENARIO=Offline MODE=Compliance  MLPERF_STAGE=True bash run_benchmark.sh
```

## MLPerf Submission Checker
Run this step inside the Docker container. The following script will perform accuracy log truncation and run the submission checker on the contents of {LOG_DIR}. Ensure the submission content has been populated before running. The original content of ${LOG_DIR} is not modified. Ensure the VENDOR and SYSTEM tags reflect those used when preparing the previous results (see ${LOG_DIR}/systems).
```
VENDOR=Dell SYSTEM=1-node-2S-GNR_86C-1DPC bash code/prepare_mlperf_submission.sh
```
