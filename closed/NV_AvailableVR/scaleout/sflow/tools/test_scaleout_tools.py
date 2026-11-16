#!/usr/bin/env python3
"""
Tests for tools/generate_master_yaml.py and tools/run_with_env.py.
Run with:
    .venv/bin/pytest tools/test_tools.py -v
"""
import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

TOOLS_DIR = Path(__file__).parent
GENERATE = TOOLS_DIR / "generate_master_yaml.py"
RUN_WITH_ENV = TOOLS_DIR / "run_with_env.py"
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# generate_master_yaml.py
# ---------------------------------------------------------------------------

class TestGenerateMasterYaml:

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [PYTHON, str(GENERATE)] + args,
            capture_output=True, text=True,
        )

    def test_basic_output(self, tmp_path):
        out = tmp_path / "server_config.yaml"
        result = self._run([
            "--hostname=10.0.0.1",
            "--port=8000",
            "--ctx-urls=10.0.0.2:8336,10.0.0.3:8336",
            "--gen-urls=10.0.0.4:8337",
            f"--output={out}",
        ])
        assert result.returncode == 0
        config = yaml.safe_load(out.read_text())
        assert config["hostname"] == "10.0.0.1"
        assert config["port"] == 8000
        assert config["backend"] == "pytorch"
        assert config["context_servers"]["num_instances"] == 2
        assert config["context_servers"]["urls"] == ["10.0.0.2:8336", "10.0.0.3:8336"]
        assert config["generation_servers"]["num_instances"] == 1
        assert config["generation_servers"]["urls"] == ["10.0.0.4:8337"]

    def test_single_ctx_single_gen(self, tmp_path):
        out = tmp_path / "cfg.yaml"
        result = self._run([
            "--hostname=127.0.0.1",
            "--port=9000",
            "--ctx-urls=192.168.1.1:8336",
            "--gen-urls=192.168.1.2:8337",
            f"--output={out}",
        ])
        assert result.returncode == 0
        config = yaml.safe_load(out.read_text())
        assert config["context_servers"]["num_instances"] == 1
        assert config["generation_servers"]["num_instances"] == 1

    def test_many_ctx_many_gen(self, tmp_path):
        out = tmp_path / "cfg.yaml"
        ctx = ",".join(f"10.0.0.{i}:8336" for i in range(1, 5))
        gen = ",".join(f"10.0.1.{i}:8337" for i in range(1, 3))
        result = self._run([
            "--hostname=10.0.0.99",
            "--port=8000",
            f"--ctx-urls={ctx}",
            f"--gen-urls={gen}",
            f"--output={out}",
        ])
        assert result.returncode == 0
        config = yaml.safe_load(out.read_text())
        assert config["context_servers"]["num_instances"] == 4
        assert config["generation_servers"]["num_instances"] == 2

    def test_register_url_to_appends(self, tmp_path):
        registry = tmp_path / "frontend_urls.txt"
        out = tmp_path / "cfg.yaml"
        self._run([
            "--hostname=10.0.0.1", "--port=8000",
            "--ctx-urls=10.0.0.2:8336", "--gen-urls=10.0.0.3:8337",
            f"--output={out}", f"--register-url-to={registry}",
        ])
        out2 = tmp_path / "cfg2.yaml"
        self._run([
            "--hostname=10.0.0.1", "--port=8001",
            "--ctx-urls=10.0.0.2:8336", "--gen-urls=10.0.0.3:8337",
            f"--output={out2}", f"--register-url-to={registry}",
        ])
        lines = [l for l in registry.read_text().splitlines() if l.strip()]
        assert lines == ["10.0.0.1:8000", "10.0.0.1:8001"]

    def test_output_parent_dir_created(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "cfg.yaml"
        result = self._run([
            "--hostname=10.0.0.1", "--port=8000",
            "--ctx-urls=10.0.0.2:8336", "--gen-urls=10.0.0.3:8337",
            f"--output={out}",
        ])
        assert result.returncode == 0
        assert out.exists()

    def test_empty_ctx_urls_exits_nonzero(self, tmp_path):
        result = self._run([
            "--hostname=10.0.0.1", "--port=8000",
            "--ctx-urls=", "--gen-urls=10.0.0.3:8337",
            f"--output={tmp_path}/cfg.yaml",
        ])
        assert result.returncode != 0

    def test_empty_gen_urls_exits_nonzero(self, tmp_path):
        result = self._run([
            "--hostname=10.0.0.1", "--port=8000",
            "--ctx-urls=10.0.0.2:8336", "--gen-urls=",
            f"--output={tmp_path}/cfg.yaml",
        ])
        assert result.returncode != 0

    def test_missing_required_args_exits_nonzero(self, tmp_path):
        result = self._run(["--hostname=10.0.0.1", "--port=8000"])
        assert result.returncode != 0

    def test_output_is_valid_yaml(self, tmp_path):
        out = tmp_path / "cfg.yaml"
        self._run([
            "--hostname=10.0.0.1", "--port=8000",
            "--ctx-urls=10.0.0.2:8336,10.0.0.3:8336",
            "--gen-urls=10.0.0.4:8337",
            f"--output={out}",
        ])
        config = yaml.safe_load(out.read_text())
        assert isinstance(config, dict)

    def test_whitespace_trimmed_from_urls(self, tmp_path):
        out = tmp_path / "cfg.yaml"
        result = self._run([
            "--hostname=10.0.0.1", "--port=8000",
            "--ctx-urls= 10.0.0.2:8336 , 10.0.0.3:8336 ",
            "--gen-urls= 10.0.0.4:8337 ",
            f"--output={out}",
        ])
        assert result.returncode == 0
        config = yaml.safe_load(out.read_text())
        assert config["context_servers"]["urls"] == ["10.0.0.2:8336", "10.0.0.3:8336"]
        assert config["generation_servers"]["urls"] == ["10.0.0.4:8337"]


# ---------------------------------------------------------------------------
# run_with_env.py
# ---------------------------------------------------------------------------

import importlib.util

def _load_run_with_env():
    spec = importlib.util.spec_from_file_location("run_with_env", RUN_WITH_ENV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRunWithEnv:

    @pytest.fixture(autouse=True)
    def mod(self):
        self._mod = _load_run_with_env()

    def _captured(self, env_vars: dict) -> str:
        """Call print_exports and return captured stdout."""
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            self._mod.print_exports(env_vars)
        finally:
            sys.stdout = old
        return buf.getvalue()

    def test_basic_export(self):
        out = self._captured({"FOO": "bar", "BAZ": 42})
        assert "export FOO='bar'" in out
        assert "export BAZ='42'" in out

    def test_empty_dict(self):
        out = self._captured({})
        assert out.strip() == ""

    def test_special_chars_in_value(self):
        out = self._captured({"PATH_VAR": "/usr/local/bin:/usr/bin"})
        assert "export PATH_VAR='/usr/local/bin:/usr/bin'" in out

    def test_integer_value(self):
        out = self._captured({"NUM_GPUS": 8})
        assert "export NUM_GPUS='8'" in out

    def test_multiple_keys_all_exported(self):
        env = {"KEY1": "value1", "KEY2": "value2", "KEY3": 123}
        out = self._captured(env)
        assert "export KEY1='value1'" in out
        assert "export KEY2='value2'" in out
        assert "export KEY3='123'" in out

    def test_output_is_eval_safe(self):
        """Every line must be a valid 'export KEY=VALUE' shell statement."""
        env = {"KEY1": "value1", "KEY2": "value2", "KEY3": 123}
        out = self._captured(env)
        for line in out.strip().splitlines():
            assert line.startswith("export ")
            assert "=" in line

    def test_fake_trtllm_env_yaml(self):
        """Simulate a realistic trtllm-serve env YAML."""
        env = {
            "TRTLLM_USE_DISAGG": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "MAX_BATCH_SIZE": 64,
            "MODEL_PATH": "/home/mlperf_inference_storage/models/gpt-oss/gpt-oss-120b",
        }
        out = self._captured(env)
        assert "export TRTLLM_USE_DISAGG='1'" in out
        assert "export CUDA_VISIBLE_DEVICES='0,1,2,3'" in out
        assert "export MAX_BATCH_SIZE='64'" in out
        assert "export MODEL_PATH='/home/mlperf_inference_storage/models/gpt-oss/gpt-oss-120b'" in out
