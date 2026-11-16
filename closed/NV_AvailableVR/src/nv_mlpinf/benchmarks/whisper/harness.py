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

import contextlib
from typing import Dict, List, Optional

from nvmitten.configurator import autoconfigure, bind
from nvmitten.nvidia.accelerator import GPU
import mlperf_loadgen as lg

from nv_mlpinf.common import logging
from nv_mlpinf.common.constants import Scenario
from nv_mlpinf.common.systems.system_list import DETECTED_SYSTEM
from nv_mlpinf.ops.harness import PyHarnessOp
from nv_mlpinf.ops.loadgen import LoadgenConfFilesOp
from nv_mlpinf.fields import models as model_fields

from .dataset import WhisperDataLoader as Dataset
from .backend import WhisperServer
from .builder import WhisperBuilderOp


@autoconfigure
@bind(model_fields.gpu_batch_size)
class WhisperHarnessOp(PyHarnessOp):

    @classmethod
    def immediate_dependencies(cls):
        return {LoadgenConfFilesOp, WhisperBuilderOp}

    @classmethod
    def output_keys(cls):
        return ["log_dir", "result_metadata"]

    def __init__(self, *args, gpu_batch_size: Optional[Dict[str, int]] = None, **kwargs):
        super().__init__(Dataset, *args, **kwargs)

        if isinstance(gpu_batch_size, dict):
            self._batch_size = list(gpu_batch_size.values())[0] if gpu_batch_size else 1
        else:
            self._batch_size = int(gpu_batch_size) if gpu_batch_size else 1
        self._server_inst = None

    def issue_queries(self, query_samples: List[lg.QuerySample]):

        self._server_inst.issue_queries(query_samples)

    def flush_queries(self):
        self._server_inst.flush_queries()

    @contextlib.contextmanager
    def wrap_lg_test(self, scratch_space, dependency_outputs):
        engine_dir = dependency_outputs[WhisperBuilderOp]["engine_dir"]
        devices = [gpu.gpu_index for gpu in DETECTED_SYSTEM.accelerators[GPU]]
        user_conf = dependency_outputs[LoadgenConfFilesOp]["user_conf"]

        try:
            self._server_inst = WhisperServer(
                devices=devices,
                dataset=self._qsl_inst,
                engine_dir=engine_dir,
                batch_size=self._batch_size,
                # gpu_inference_streams=1,  # Change this when whisper supports multiple cores per device
                # gpu_copy_streams=1,
                assets_dir="build/models/whisper-large-v3/",
                enable_batcher=(self.wl.scenario == Scenario.Server))
            logging.info("Start Warm Up!")
            self._server_inst.warm_up()
            logging.info("Warm Up Done!")
            yield None
        finally:
            self._server_inst.finish_test()
