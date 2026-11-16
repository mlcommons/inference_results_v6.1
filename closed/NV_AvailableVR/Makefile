# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is the main entrypoint of NVIDIA's MLPerf Inference codebase, and includes all of the other Makefiles used
# in the project.
# This file contains the targets used to run the actual MLPerf Inference workloads, as well as some basic utility
# commands.

MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))  # Must be declared before includes
include $(CURDIR)/Makefile.const
include $(CURDIR)/Makefile.docker
include $(CURDIR)/Makefile.data
include $(CURDIR)/Makefile.tests
include $(CURDIR)/Makefile.submission

# Auto-install nv-mlpinf with the right extras based on --benchmarks in RUN_ARGS.
# Runs every time (pip is fast when deps are already satisfied).

# Generate TensorRT engines (plan files) and run the harness.
.PHONY: run
run:
	@$(MAKE) -f $(MAKEFILE_NAME) generate_engines
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness

# Generate TensorRT engines (plan files).
.PHONY: generate_engines
generate_engines: link_dirs
	@$(PYTHON3_CMD) -m nv_mlpinf.main $(RUN_ARGS) --action="generate_engines"

.PHONY: run_llm_server
run_llm_server: link_dirs
	@$(PYTHON3_CMD) -m nv_mlpinf.main $(RUN_ARGS) --action="run_llm_server"

# Run the harness and check accuracy if in AccuracyOnly mode.
# Add "set -o pipefail" so that "<command> | tee output.txt" will fail when "<command>" fails because otherwise tee
# would return status 0 and clear the failure status.
NO_DISPLAY_RESULTS ?= 0
IGNORE_ACCURACY_FAILURE ?= 0
MPI_CMD ?= mpirun
MPI_EXPORTS ?= -x LD_LIBRARY_PATH
MPI_ARGS ?= --allow-run-as-root
.PHONY: run_harness
run_harness: link_dirs
	@mkdir -p $(LOG_DIR)
	@set -o pipefail && LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m nv_mlpinf.main $(RUN_ARGS) --action="run_harness" 2>&1 | tee $(LOG_DIR)/stdout.txt
ifeq ($(NO_DISPLAY_RESULTS), 0)
	@$(MAKE) -f $(MAKEFILE_NAME) display_results
endif

.PHONY: display_results
display_results:
	@set -o pipefail && $(PYTHON3_CMD) -m scripts.print_harness_result --log_dir=$(LOG_DIR) --ignore-accuracy-failure=$(IGNORE_ACCURACY_FAILURE) 2>&1 | tee -a $(LOG_DIR)/stdout.txt

.PHONY: run_audit_harness
run_audit_harness: link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test01
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test04
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test06
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test07
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test09

.PHONY: run_audit_test01_once
run_audit_test01_once: link_dirs
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m nv_mlpinf.main $(RUN_ARGS) --audit_test=TEST01 --server_target_qps_adj_factor=0.92 --action="run_harness"

# TEST01 is sometimes unstable. Try up to two times before failing.
.PHONY: run_audit_test01
run_audit_test01:
	@for i in 1 2; do echo "TEST01 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test01_once && break; done

.PHONY: run_audit_test04_once
run_audit_test04_once: link_dirs
	@echo "Sleep to reset thermal state before TEST04..." && sleep 20
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m nv_mlpinf.main $(RUN_ARGS) --audit_test=TEST04 --server_target_qps_adj_factor=0.92 --action="run_harness"

# TEST04 is so short that it is sometimes unstable. Try up to three times before failing.
.PHONY: run_audit_test04
run_audit_test04:
	@for i in 1 2 3; do echo "TEST04 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test04_once && break; done

.PHONY: run_audit_test06
run_audit_test06: link_dirs
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m nv_mlpinf.main $(RUN_ARGS) --audit_test=TEST06 --server_target_qps_adj_factor=0.92 --action="run_harness"

.PHONY: run_audit_test07
run_audit_test07: link_dirs
	@echo "Running TEST07..."
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m nv_mlpinf.main $(RUN_ARGS) --audit_test=TEST07 --server_target_qps_adj_factor=0.92 --action="run_harness"

.PHONY: run_audit_test09
run_audit_test09: link_dirs
	@echo "Running TEST09..."
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m nv_mlpinf.main $(RUN_ARGS) --audit_test=TEST09 --server_target_qps_adj_factor=0.92 --action="run_harness"


############################## UTILITY ##############################

# Add symbolic links to scratch path if it exists.
.PHONY: link_dirs
link_dirs:
	@mkdir -p build
	@mkdir -p $(DATA_DIR)
	@mkdir -p $(PREPROCESSED_DATA_DIR)
	@mkdir -p $(MODEL_DIR)
	@ln -sfn $(DATA_DIR) $(BUILD_DIR)/data
	@ln -sfn $(PREPROCESSED_DATA_DIR) $(BUILD_DIR)/preprocessed_data
	@ln -sfn $(MODEL_DIR) $(BUILD_DIR)/models

# Remove build directory.
.PHONY: clean
clean: clean_shallow
	rm -rf build


# Remove only the files necessary for a clean build.
.PHONY: clean_shallow
clean_shallow:
	rm -rf build/bin
	rm -rf $(PROJECT_ROOT)/3rdparty/trtllm
	rm -rf $(LOADGEN_LIB_DIR)
