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

from nv_mlpinf.llmlib.dataset import LLMDataLoader
from nv_mlpinf.llmlib.utils import prefix_logger as logging
from nv_mlpinf.fields import loadgen as lg_fields
from nvmitten.configurator import autoconfigure, bind


@autoconfigure
@bind(lg_fields.test_mode)
class GptOssDataset(LLMDataLoader):

    def __init__(self, *args, test_mode: str = "PerformanceOnly", **kwargs):
        suffix = '_acc_eval' if test_mode == 'AccuracyOnly' else '_perf_eval'
        candidate_fnames = [f'input_ids_padded{suffix}.npy', f'input_lens{suffix}.npy']

        super().__init__(
            candidate_fnames,
            *args,
            **kwargs
        )

        self.tensors = {
            (k[:-len(suffix)] if k.endswith(suffix) else k): v
            for k, v in self.tensors.items()
        }
        # Truncate inputs to proper lengths
        self.input_table = [
            input_ids[:input_len].reshape(-1).tolist()
            for input_ids, input_len in zip(self.input_ids_padded, self.input_lens)
        ]
        logging.debug("Completed pre-processing input tokens. Ready for inference.")

        if len(self) != self.total_sample_count:
            logging.warning(f"Total sample count mismatch: {len(self)} != {self.total_sample_count}")
            logging.warning(f"Using {len(self)} samples for inference.")
            self.total_sample_count = len(self)

    def get_input_tokens(self, sample_indices):
        return [
            self.input_table[sample_index]
            for sample_index in sample_indices
        ]

    def __len__(self):
        return len(self.input_table)

