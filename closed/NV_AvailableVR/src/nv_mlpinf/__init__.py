#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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

from importlib import import_module
from pathlib import Path
from typing import Dict, Tuple
from nv_mlpinf.common.constants import Benchmark


class ModuleLocation:
    def __init__(self, path: Path, op_names: Tuple[str, ...] = ()):
        self.path = path
        self.op_names = op_names
        self._m = None
        self.custom_op_impls = {}

    def load(self):
        if not self._m:
            self._m = import_module(self.path)
        for op_name in self.op_names:
            if hasattr(self._m, op_name):
                self.custom_op_impls[op_name] = getattr(self._m, op_name)
            else:
                raise ValueError(f"Op {op_name} is required by {self.path} but not found in the module.")
        return self._m


_llm_ops = ("LoadgenConfFilesOp", "TrtllmServeClientHarnessOp", "ResultSummaryOp")
G_BENCHMARK_MODULES: Dict[Benchmark, ModuleLocation] = {
    Benchmark.LLAMA2:        ModuleLocation("nv_mlpinf.benchmarks.llama2_70b",     op_names=_llm_ops),
    Benchmark.LLAMA3_1_8B:   ModuleLocation("nv_mlpinf.benchmarks.llama3_1_8b",   op_names=_llm_ops),
    Benchmark.DeepSeek_R1:   ModuleLocation("nv_mlpinf.benchmarks.deepseek_r1",   op_names=_llm_ops),
    Benchmark.GPT_OSS_120B:  ModuleLocation("nv_mlpinf.benchmarks.gpt_oss_120b",  op_names=_llm_ops),
    Benchmark.WHISPER:       ModuleLocation("nv_mlpinf.benchmarks.whisper",       op_names=("CalibrateEngineOp", "EngineBuilderOp", "LoadgenConfFilesOp", "WhisperHarnessOp", "ResultSummaryOp")),
}
