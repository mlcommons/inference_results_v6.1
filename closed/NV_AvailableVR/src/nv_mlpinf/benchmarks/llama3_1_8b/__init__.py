# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

from nv_mlpinf.llmlib import TrtllmServeClientHarnessOp, CoreType
import nv_mlpinf.common.paths as paths
from nv_mlpinf.ops.loadgen import LoadgenConfFilesOp
from nv_mlpinf.ops.harness import ResultSummaryOp

# Llama3.1-8b uses the same DataLoader class as Llama2-70b
DataLoader = importlib.import_module("nv_mlpinf.benchmarks.llama2_70b.dataset").LlamaDataset

MODEL_CHECKPOINT_PATH = paths.MODEL_DIR / "Llama3.1-8B" / "fp4-quantized-modelopt" / "llama3_1-8b-instruct-hf-torch-fp4"

DEFAULT_CORE_TYPE = CoreType.TRTLLM_ENDPOINT
HF_MODEL_REPO = {"meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659"}

