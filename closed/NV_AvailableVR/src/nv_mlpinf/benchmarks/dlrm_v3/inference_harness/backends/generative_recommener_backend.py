from inference_harness.backends.base import DLRMBackend
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from generative_recommenders.dlrm_v3.datasets.dataset import Samples
from typing import Dict, List

import os

from generative_recommenders.dlrm_v3.inference.model_family import HSTUModelFamily
from generative_recommenders.dlrm_v3.inference.inference_modules import set_is_inference

# this implementation couples the embedding table config and hstu config to the backend
# so user cannot switch between using different embedding config and hstu config.


class GenerativeRecommenderBackend(DLRMBackend):
    def __init__(self, model_name: str, perf_mode: str, device_id: List[int]):
        super().__init__(model_name=model_name, perf_mode=perf_mode)
        self.backend = "GR"
        self.device_id = device_id

    def __del__(self):
        # clean up the worker
        if len(self.device_id) > 1:
            self.model_impl.predict(None)

    def initialize(self, hstu_config: DlrmHSTUConfig, embedding_table_config: Dict[str, EmbeddingConfig]):
        set_is_inference(is_inference=True if self.perf_mode == "performance" else False)
        os.environ["WORLD_SIZE"] = str(len(self.device_id))
        self.model_impl = HSTUModelFamily(
            hstu_config=hstu_config,
            table_config=embedding_table_config,
            # profiler currently turned off
            output_trace=False,
            sparse_quant=False,
        )
        self.load_model(checkpoint_path="")

    def load_model(self, checkpoint_path: str):
        self.model_impl.load(checkpoint_path)

    def predict(self, feed: Samples):
        return self.model_impl.predict(feed)
