# MLPerf Inference v6.0 Azure-Optimized Implementations

This is a repository of Azure-optimized implementations for the [MLPerf](https://mlcommons.org/en/) Inference Benchmark.
This README is a quickstart tutorial on how to use our code as a public / external user.

## Table of Contents

- [NVIDIA's Submission](#nvidias-submission)
- [MLPerf Inference Policies and Terminology](#mlperf-inference-policies-and-terminology)
- [Setting Up 3rdparty Dependencies](#setting-up-3rdparty-dependencies)
- [Quick Start: Running Your First Benchmark](#quick-start-running-your-first-benchmark)

---

### Azure's Submission

Our submission includes support for deepseek-r1 on GB200 and GB300. The instructions to reproduce are stored in `closed/Azure/src/nv_mlpinf/benchmarks/`. **Each benchmark contains a** `README.md` **with detailed setup, data preparation, and execution instructions.** Please refer to the benchmark-specific README for accurate reproduction steps.

---

### MLPerf Inference Policies and Terminology

This is a new-user guide to learn how to use NVIDIA's MLPerf Inference submission repo. **To get started with MLPerf Inference, first familiarize yourself with the [MLPerf Inference Policies, Rules, and Terminology](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)**. This is a document from the MLCommons committee that runs the MLPerf benchmarks, and the rest of all MLPerf Inference guides will assume that you have read and familiarized yourself with its contents. The most important sections of the document to know are:

- [Key terms and definitions](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#11-definitions-read-this-section-carefully)
- [Scenarios](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#3-scenarios)
- [Benchmarks and constraints for the Closed Division](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#411-constraints-for-the-closed-division)
- [LoadGen Operation](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#51-loadgen-operation)

### Setting Up 3rdparty Dependencies

The submission export does not include the `3rdparty/` directory. You must clone the required
repositories before building or running any benchmarks:

```bash
cd closed/Azure
git submodule update --init
```

### Setting up the Scratch Space

The scratch space stores models, datasets, and preprocessed data. **Recommended size: ≥10 TB** (for all benchmarks; smaller if running subset).

The canonical in-container scratch root is `/home/mlperf_inference_storage`.
Keep that directory layout as the source of truth for model weights, datasets,
and preprocessed data:

```
/home/mlperf_inference_storage/
|-- data/
|-- models/
`-- preprocessed_data/
```

Use `/home/mlperf_inference_storage` as the shared recipe layout whenever
possible, so Docker and Enroot/Pyxis runs can look for weights and datasets in
the same logical place:

- Model weights: `/home/mlperf_inference_storage/models/...`
- Raw or tokenized datasets: `/home/mlperf_inference_storage/data/...`
- Preprocessed benchmark artifacts: `/home/mlperf_inference_storage/preprocessed_data/...`


**Setup (choose one):**

1. Mount or bind your scratch space at `/home/mlperf_inference_storage`
  before launching Docker, OR
2. Export `MLPERF_SCRATCH_PATH` to your scratch location before launching the
  Docker container. Docker mounts that path and passes the environment variable
   into the container.

For nv-sflow SLURM runs, update the `SCRATCH_DIR` variable in the
`slurm_env_sflow.yaml` for each run config to the host path where you store
weights and data. The environment file mounts that host path at
`/home/mlperf_inference_storage` inside the Enroot container.

**Note:** Setup is one-time only. Re-run only if data is corrupted, new benchmarks are added, or preprocessing changes.

**If you export MLPERF_SCRATCH_PATH, scratch space will mount automatically when you launch container.**

```
$ export MLPERF_SCRATCH_PATH=/path/to/scratch/space
```

This `MLPERF_SCRATCH_PATH` will also be mounted inside the docker container at the same path (i.e. if your scratch space is located at `/mnt/some_ssd`, it will be mounted in the container at `/mnt/some_ssd` as well.)

Then create empty directories in your scratch space to house the data:

```
$ mkdir $MLPERF_SCRATCH_PATH/data $MLPERF_SCRATCH_PATH/models $MLPERF_SCRATCH_PATH/preprocessed_data
```

After you have done so, you will need to download the models and datasets, and run the preprocessing scripts on the datasets. 

Enter a container (see Container Setup above), then verify the scratch space:

```
$ echo $MLPERF_SCRATCH_PATH  # Make sure that the container has the MLPERF_SCRATCH_PATH set correctly
$ ls -al $MLPERF_SCRATCH_PATH  # Make sure that the container mounted the scratch space correctly
$ make clean  # Make sure that the build/ directory isn't dirty
$ make link_dirs  # Link the build/ directory to the scratch space
$ ls -al build/  # You should see output like the following:
total 8
drwxrwxr-x  2 user group 4096 Jun 24 18:49 .
drwxrwxr-x 15 user group 4096 Jun 24 18:49 ..
lrwxrwxrwx  1 user group   35 Jun 24 18:49 data -> $MLPERF_SCRATCH_PATH/data
lrwxrwxrwx  1 user group   37 Jun 24 18:49 models -> $MLPERF_SCRATCH_PATH/models
lrwxrwxrwx  1 user group   48 Jun 24 18:49 preprocessed_data -> $MLPERF_SCRATCH_PATH/preprocessed_data
```

Once you have verified that the `build/data`, `build/models/`, and `build/preprocessed_data` point to the correct directories in your scratch space, you can continue.

### Quick Start: Running Your First Benchmark

**Submission Matrix:** We formally support and fully test configuration files for specific systems. See [Docker support](configs/DOCKER_SUPPORT.md) and [SLURM support](configs/SLURM_SUPPORT.md) for the complete system support matrix.

For systems not on the list, manual tuning or config changes may be required to achieve optimal performance.

**Launch Modes:** Benchmarks support two launch modes:

- **Docker (single-node)** - Simplest option for running on a single machine (e.g., B200x8, B300x8)
- **Enroot + SLURM (multi-node)** - For scale-out across multiple nodes (e.g., GB200x72)

For a quick example using Docker mode, see the [GPT-OSS-120B README](src/nv_mlpinf/benchmarks/gpt_oss_120b/README.md). For detailed launch instructions for both modes, refer to [AGENTS.md](AGENTS.md#run-benchmark-step-2-launch-instructions).


#### Use a Non-Root User

**We highly recommend to run MLPerf as a sudo user (i.e. a user in the sudo group), and avoid using sudo command in the container. Some functionality might be broken without sudo privileges.**

If you're already a non-root user, simply don't use sudo for any command that is not a package install or a command that specifically has 'sudo' contained in it. Otherwise, create a new user. It is advisable to make this new user a sudoer, but as said before, do not invoke sudo unless necessary.

Make sure that your user is in docker group already. If you get permission issue when running docker commands, please add the user to docker group with `sudo usermod -a -G docker $USER`.

#### Software Dependencies

**Datacenter systems:**

Starting v6.0, we recommend using SLURM with Pyxis/Enroot workflow to run multi-node scale-out experiments.

Some of our submissions use Docker to set up the environment. Requirements are:

- [Docker CE](https://docs.docker.com/engine/install/)
  - If you have issues with running Docker without sudo, follow this [Docker guide from DigitalOcean](https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket) on how to enable Docker for your new non-root user. Namely, add your new user to the Docker usergroup, and remove ~/.docker or chown it to your new user.
  - Install Docker buildx plugin: `apt-get install docker-buildx-plugin`
  - You may also have to restart the docker daemon for the changes to take effect:

```
$ sudo systemctl restart docker
```

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
  - libnvidia-container >= 1.4.0
- NVIDIA Driver Version 550.xx or greater
  - For v6.0 submission, we recommend driver version 550.xx or greater

### Update the results directory for submission

See [docs/SUBMISSION.md](docs/SUBMISSION.md).

### Run compliance tests and update the compliance test logs

See [docs/SUBMISSION.md](docs/SUBMISSION.md).


