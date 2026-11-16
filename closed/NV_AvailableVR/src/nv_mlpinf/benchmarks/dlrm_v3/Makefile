.PHONY: build_docker run_docker



HOST_VOL ?= ${PWD}
CONTAINER_VOL ?= /work
DOCKER_NAME := generative-recommender-harness-docker
ARCH ?= blackwell

# User info (needed for run_docker and graceblackwell build)
GROUPID := $(shell id -g `whoami`)
UID := $(shell id -u `whoami`)
GROUPNAME := $(shell id -g -n `whoami`)
UNAME := $(shell whoami)


build_docker_blackwell:
	docker buildx build -t $(DLRMV3_BASE_IMAGE) -f Dockerfile.x86 .

build_docker_graceblackwell:
	docker buildx build -t $(DLRMV3_BASE_IMAGE) -f Dockerfile.aarch64 .



run_docker:
	docker run --rm -it --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 \
	-v $(realpath $(HOST_VOL)):$(CONTAINER_VOL) -v $(realpath ${HOME}):/mnt/${HOME} -w /work  \
	--cap-add=ALL \
	-v /home/scratch.zihaok_gpu:/home/scratch.zihaok_gpu \
	-v /home/mlperf_inference_storage_01:/home/mlperf_inference_storage_01 \
	-v /raid/data/zihaok:/raid/data/zihaok \
	-v /raid/data/zihaok_1:/raid/data/zihaok_1 \
	--name $(DOCKER_NAME)docker --user $(UID) \
	$(DLRMV3_BASE_IMAGE)

