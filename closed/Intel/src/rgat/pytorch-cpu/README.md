# R-GAT Inference on CPU

## LEGAL DISCLAIMER
To the extent that any data, datasets, or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality. By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license. 

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets, or models. 

## Launch the Docker Image
Set the directories on the host system where model, dataset, and log files will reside. These locations will retain model and data content between Docker sessions.
```
export MODEL_DIR="${MODEL_DIR:-${PWD}/model}"
export DATA_DIR="${DATA_DIR:-${PWD}/data}"
export LOG_DIR="${LOG_DIR:-${PWD}/logs}"
```

In the Host OS environment, run the following after setting the proper Docker image name. If the Docker image is not on the system already, it will be retrieved from the registry.

If retrieving the model or dataset, ensure any necessary proxy settings are run inside the container.
```
export DOCKER_IMAGE=intel/mlperf:mlperf-inference-6.1-rgat_cpu
MOUNT_ARGS="-v ${MODEL_DIR}:/model -v ${DATA_DIR}:/data -v ${LOG_DIR}:/logs"

docker run --privileged -it --rm \
        --ipc=host --net=host --cap-add=ALL \
        -e http_proxy=${http_proxy} \
        -e https_proxy=${https_proxy} \
        -e HTTP_PROXY=${http_proxy} \
        -e HTTPS_PROXY=${https_proxy} \
        -e no_proxy=${no_proxy} \
        ${MOUNT_ARGS} \
        --workdir /workspace \
        ${DOCKER_IMAGE} /bin/bash
```

## Prepare workload resources [one-time operations]
Download the model: Run this step inside the Docker container.  This operation will preserve the model on the host system using the volume mapping above.
```
bash scripts/download_model.sh
```
Download the dataset: Run this step inside the Docker container.  This operation will preserve the dataset on the host system using the volume mapping above.
NOTE: This is a very time-intensive and storage-intensive process (over 12h runtime and 2TB of storage temporarily needed).  Once completed, be sure to preserve the dataset to avoid repeating this step.
```
bash scripts/download_dataset.sh
```

## Run Benchmark
Run this step inside the Docker container.  Select the appropriate scenario.
NOTE: This workload performs best if the system can have 1.5TB Memory. In cases where there isn't enough memory, please expect up to 4% run-to-run variation.

Performance:
```
SCENARIO=Offline MODE=Performance bash run_mlperf.sh
```
Accuracy:
```
SCENARIO=Offline MODE=Accuracy    bash run_mlperf.sh
```

## Run Compliance Tests
Run this step inside the Docker container.  After the benchmark scenarios have been run and results exist in {LOG_DIR}/results, run this step to complete compliance runs. Compliance output will be found in the {LOG_DIR}/results directory.
```
SCENARIO=Offline MODE=Compliance  bash run_mlperf.sh
```

## Validate Submission Checker
Run this step inside the Docker container.  The following script will perform accuracy log truncation and run the submission checker on the contents of {LOG_DIR}. The source scripts are distributed as MLPerf Inference reference tools. Ensure the submission content has been populated before running.  The script output is transient and destroyed after running.  The original content of ${LOG_DIR} is not modified. Update values for VENDOR and SYSTEM to match the intended system json file.
```
VENDOR=OEM SYSTEM=1-node-2S-GNR_86C bash scripts/prepare_submission.sh
```
