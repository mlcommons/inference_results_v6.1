# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the endpoint harness's tokenizer checkpoint override."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nv_mlpinf import llmlib
from nv_mlpinf.fields import models as model_fields
from nvmitten.configurator import Configuration


class EndpointModelPathTest(unittest.TestCase):
    def make_op(self, **kwargs):
        with patch.object(llmlib.LLMHarnessOp, "__init__", return_value=None):
            op = llmlib.TrtllmServeClientHarnessOp(**kwargs)
        op.wl = SimpleNamespace(benchmark="llama")
        return op

    def test_explicit_path_is_forwarded_without_loading_default(self):
        op = self.make_op(model_path=Path("/models/fp8/tokenizer"))
        with patch.object(llmlib, "G_BENCHMARK_MODULES", {}):
            self.assertEqual(op.get_backend_kwargs({}), {"model_path": "/models/fp8/tokenizer"})

    def test_string_path_is_supported(self):
        op = self.make_op(model_path="/models/custom")
        with patch.object(llmlib, "G_BENCHMARK_MODULES", {}):
            self.assertEqual(op.get_backend_kwargs({})["model_path"], "/models/custom")

    def test_configured_model_path_is_bound_to_the_operation(self):
        configured_path = Path("/models/configured-fp8")
        with Configuration({model_fields.model_path: configured_path}).autoapply():
            op = self.make_op()
        with patch.object(llmlib, "G_BENCHMARK_MODULES", {}):
            self.assertEqual(op.get_backend_kwargs({})["model_path"], str(configured_path))

    def test_default_checkpoint_is_preserved(self):
        checkpoint = Path("/models/llama/fp4")
        benchmark = SimpleNamespace(load=lambda: SimpleNamespace(MODEL_CHECKPOINT_PATH=checkpoint))
        op = self.make_op()
        with patch.object(llmlib, "G_BENCHMARK_MODULES", {"llama": benchmark}):
            self.assertEqual(op.get_backend_kwargs({})["model_path"], str(checkpoint))

    def test_other_backend_kwargs_are_preserved(self):
        op = self.make_op(model_path=Path("/models/fp8"))
        with patch.object(llmlib.LLMHarnessOp, "get_backend_kwargs", return_value={"other": 42}):
            self.assertEqual(op.get_backend_kwargs({}), {"other": 42, "model_path": "/models/fp8"})


if __name__ == "__main__":
    unittest.main()
