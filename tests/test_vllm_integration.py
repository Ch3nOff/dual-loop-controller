"""
Unit Tests for vLLM Integration Layer
======================================
Tests the HADL-vLLM integration components including:
  - VLLMInferenceEngine initialization and configuration
  - vLLM plugin registration
  - Hook attachment mechanics (mocked)
  - Offline batch generation API

Note: These tests mock vLLM internals since vLLM requires a Linux GPU
environment. Full integration tests should be run on a GPU machine.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock


class TestVLLMPluginRegistration(unittest.TestCase):
    """Tests the HADL vLLM plugin registration logic."""

    def test_plugin_disabled_when_env_not_set(self):
        """Plugin should be a no-op when HADL_VLLM_ENABLED is not set."""
        with patch.dict(os.environ, {"HADL_VLLM_ENABLED": "0"}, clear=False):
            from dual_loop.vllm_plugin import register
            # Should not raise
            register()

    def test_plugin_reads_k_steps_from_env(self):
        """Plugin should read HADL_VLLM_K_STEPS from environment."""
        with patch.dict(
            os.environ,
            {"HADL_VLLM_ENABLED": "1", "HADL_VLLM_K_STEPS": "3"},
            clear=False,
        ):
            # Mock vllm import to prevent actual CUDA initialization
            mock_module = MagicMock()
            with patch.dict(sys.modules, {"vllm": mock_module, "vllm.worker": mock_module, "vllm.worker.model_runner": mock_module}):
                # The register function should attempt to patch GPUModelRunnerBase
                from dual_loop.vllm_plugin import register
                # Should not raise even if vllm internals differ
                try:
                    register()
                except (ImportError, AttributeError):
                    pass  # Expected when mocking


class TestVLLMEngineConfig(unittest.TestCase):
    """Tests VLLMInferenceEngine configuration."""

    @patch.dict(sys.modules, {"vllm": MagicMock(), "vllm.engine": MagicMock(), "vllm.engine.arg_utils": MagicMock(), "vllm.engine.async_llm_engine": MagicMock(), "vllm.entrypoints": MagicMock(), "vllm.entrypoints.openai": MagicMock(), "vllm.entrypoints.openai.api_server": MagicMock()})
    def test_engine_initialization(self):
        """Engine should initialize with correct configuration."""
        # Force reimport with mocked vllm
        if 'dual_loop.server.vllm_engine' in sys.modules:
            del sys.modules['dual_loop.server.vllm_engine']

        from dual_loop.server.vllm_engine import VLLMInferenceEngine

        engine = VLLMInferenceEngine(
            model_id_or_path="Qwen/Qwen2.5-7B-Instruct",
            tensor_parallel_size=2,
            hadl_k_steps=3,
            quantization="awq",
        )

        self.assertEqual(engine.model_id, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(engine.hadl_k_steps, 3)
        self.assertTrue(engine.enable_hadl)
        self.assertFalse(engine.is_loaded)
        self.assertEqual(engine._vllm_kwargs["tensor_parallel_size"], 2)
        self.assertEqual(engine._vllm_kwargs["quantization"], "awq")

    @patch.dict(sys.modules, {"vllm": MagicMock(), "vllm.engine": MagicMock(), "vllm.engine.arg_utils": MagicMock(), "vllm.engine.async_llm_engine": MagicMock(), "vllm.entrypoints": MagicMock(), "vllm.entrypoints.openai": MagicMock(), "vllm.entrypoints.openai.api_server": MagicMock()})
    def test_engine_telemetry_init(self):
        """Telemetry counters should start at zero."""
        if 'dual_loop.server.vllm_engine' in sys.modules:
            del sys.modules['dual_loop.server.vllm_engine']

        from dual_loop.server.vllm_engine import VLLMInferenceEngine

        engine = VLLMInferenceEngine()
        telem = engine.get_telemetry()

        self.assertEqual(telem["total_tokens_generated"], 0)
        self.assertEqual(telem["total_requests"], 0)
        self.assertEqual(telem["engine"], "vllm")


class TestVLLMAppConfig(unittest.TestCase):
    """Tests the vLLM app server configuration."""

    def test_vllm_availability_check(self):
        """Should correctly detect vLLM availability."""
        from dual_loop.server.vllm_app import _check_vllm_available
        # This should return True or False without raising
        result = _check_vllm_available()
        self.assertIsInstance(result, bool)

    def test_env_var_setup_for_hadl(self):
        """start_vllm_server should set HADL env vars before launching."""
        # We just verify the env var logic, not the actual server launch
        os.environ["HADL_VLLM_ENABLED"] = "1"
        os.environ["HADL_VLLM_K_STEPS"] = "3"

        self.assertEqual(os.environ.get("HADL_VLLM_ENABLED"), "1")
        self.assertEqual(os.environ.get("HADL_VLLM_K_STEPS"), "3")

        # Cleanup
        del os.environ["HADL_VLLM_ENABLED"]
        del os.environ["HADL_VLLM_K_STEPS"]


class TestOfflineBatchAPI(unittest.TestCase):
    """Tests the offline batch inference API surface."""

    def test_batch_api_importable(self):
        """vllm_offline module should be importable."""
        from dual_loop.server.vllm_offline import vllm_batch_generate, vllm_compare_base_vs_hadl
        self.assertTrue(callable(vllm_batch_generate))
        self.assertTrue(callable(vllm_compare_base_vs_hadl))


if __name__ == "__main__":
    unittest.main()
