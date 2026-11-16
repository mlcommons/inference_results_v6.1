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

from nv_mlpinf.llmlib import TrtllmServeClientHarnessOp, CoreType
from .dataset import DeepseekDataset as DataLoader
import nv_mlpinf.common.paths as paths
from nv_mlpinf.ops.loadgen import LoadgenConfFilesOp
from nv_mlpinf.ops.harness import ResultSummaryOp

MODEL_CHECKPOINT_PATH = paths.MODEL_DIR / "deepseek-r1" / "fp4-quantized-modelopt" / "deepseek_r1-torch-fp4"

DEFAULT_CORE_TYPE = CoreType.TRTLLM_ENDPOINT
HF_MODEL_REPO = {"deepseek-ai/deepseek-r1": '56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad'}

