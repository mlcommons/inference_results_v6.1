# MLPerf Inference v6.1 Nebius submission
Based on the MLPerf v6.0 NVIDIA-Optimized Implementation.

This README covers the submission of the `gpt-oss-120b` benchmark on 1 node `RTX_PRO_6000_PCIE_96GB x8` using the Docker workflow.

The details related to other benchmarks and other workflows can be found at https://github.com/mlcommons/inference_results_v6.0/blob/main/closed/NVIDIA/README.md

The following instructions require the content of closed/Nebius expect the `results` and `systems` directories to be replaced with the content of closed/Nebius/R6000.

### Setting Up 3rdparty Dependencies

The submission export does not include the `3rdparty/` directory. You must clone the required repositories before building or running any benchmarks:

```bash
cd closed/Nebius
mkdir -p 3rdparty
git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git 3rdparty/trtllm
git clone --depth 1 https://github.com/NVIDIA/mitten.git 3rdparty/mitten
git clone --depth 1 https://github.com/mlcommons/inference.git 3rdparty/mlc-inference
```

### Setting up the Scratch Spaces

NVIDIA's MLPerf Inference submission stores the models, datasets, and preprocessed datasets in a central location we refer to as a "Scratch Space".

Because of the large amount of data that needs to be stored in the scratch space, we recommend that the scratch be at least **10 TB**. This size is recommended if you wish to obtain every dataset in order to run each benchmark and have extra room to store logs, engines, etc. If you do not need to run every single benchmark, it is possible to use a smaller scratch space.

**Note that once the scratch space is setup and all the data, models, and preprocessed datasets are set up, you do not have to re-run this step.** You will only need to revisit this step if:

- You accidentally corrupted or deleted your scratch space
- You need to redo the steps for a benchmark you previously did not need to set up
- You, NVIDIA, or MLCommons has decided that something in the preprocessing step needed to be altered

Once you have obtained a scratch space, set the `MLPERF_SCRATCH_PATH` environment variable. This is how our code tracks where the data is stored. By default, if this environment variable is not set, we assume the scratch space is located at `/home/mlperf_inference_storage`. Because of this, it is highly recommended to mount your scratch space at this location.

**If you export MLPERF_SCRATCH_PATH, scratch space will mount automatically when you launch container.**

```
$ export MLPERF_SCRATCH_PATH=/path/to/scratch/space
```
This `MLPERF_SCRATCH_PATH` will also be mounted inside the docker container at the same path (i.e. if your scratch space is located at `/mnt/some_ssd`, it will be mounted in the container at `/mnt/some_ssd` as well.)

Then create empty directories in your scratch space to house the data:

```
$ mkdir $MLPERF_SCRATCH_PATH/data $MLPERF_SCRATCH_PATH/models $MLPERF_SCRATCH_PATH/preprocessed_data
```
After you have done so, you will need to download the models and datasets, and run the preprocessing scripts on the datasets. **If you are submitting MLPerf Inference with a low-power machine, such as a mobile platform, it is recommended to do these steps on a desktop or server environment with better CPU and memory capacity.**

Next, follow the model-specific README at `closed/Nebius/src/gpt-oss-120b/tensorrt/README.md`

### N.B. Running INT8 calibration

**Note this is not needed for external users**

For legacy benchmarks using implicit quantization (e.g. RetinaNet, 3d-unet etc.), the calibration caches generated from the default calibration sets (set by MLPerf Inference committee) are already provided in each benchmark directory. If you would like to regenerate the calibration cache for a specific benchmark, run:

```
$ make calibrate RUN_ARGS="--benchmarks=[benchmark]"
```
See documentation/calibration.md for an explanation on how calibration is used for NVIDIA's submission.
