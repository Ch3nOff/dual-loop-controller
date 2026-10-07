"""
Unit tests for Dual-Loop vLLM Inference Server and Engine Customization
========================================================================
Validates vLLM server configuration, security bindings, HADL v4.5 hooks,
logits processing, memory state resets, and batch generation.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import torch

from dual_loop.server import (
    start_server,
    start_vllm_server,
    VLLMInferenceEngine,
    vllm_batch_generate,
    vllm_compare_base_vs_hadl,
)
from dual_loop.vllm_plugin import (
    register as register_vllm_plugin,
    HADLDeliberationLogitsProcessor,
    create_hadl_logits_processor,
)


class TestVLLMServerSecurityAndConfig(unittest.TestCase):
    def setUp(self):
        self._orig_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)

    def test_security_mandates_api_key_on_public_interface(self):
        """Binding to non-loopback interface (0.0.0.0) without API key must raise ValueError."""
        old_vllm = os.environ.pop("VLLM_API_KEY", None)
        old_dual = os.environ.pop("DUAL_LOOP_API_KEY", None)
        try:
            with self.assertRaises(ValueError) as ctx:
                start_vllm_server(host="0.0.0.0", api_key=None)
            self.assertIn("requires an API key", str(ctx.exception))

            # Also check start_server entrypoint alias
            with self.assertRaises(ValueError) as ctx2:
                start_server(host="0.0.0.0", api_key=None)
            self.assertIn("requires an API key", str(ctx2.exception))
        finally:
            if old_vllm:
                os.environ["VLLM_API_KEY"] = old_vllm
            if old_dual:
                os.environ["DUAL_LOOP_API_KEY"] = old_dual

    @patch("dual_loop.server.vllm_app._check_vllm_available", return_value=True)
    def test_security_allows_public_interface_with_api_key(self, mock_check):
        """Binding to non-loopback interface with an explicit API key passes auth check."""
        with patch("subprocess.run"):
            with patch("sys.argv", ["vllm", "serve"]):
                try:
                    start_vllm_server(host="0.0.0.0", api_key="secret-token-xyz")
                except (SystemExit, Exception):
                    pass
                self.assertEqual(os.environ.get("VLLM_API_KEY"), "secret-token-xyz")

    @patch("dual_loop.server.vllm_app._check_vllm_available", return_value=True)
    def test_environment_variables_configured_for_plugin(self, mock_check):
        """start_vllm_server sets environment variables expected by dual_loop_vllm_plugin."""
        with patch("subprocess.run"):
            try:
                start_vllm_server(
                    host="127.0.0.1",
                    enable_hadl=True,
                    hadl_version="v4.5",
                    hadl_k_steps=3,
                    checkpoint="checkpoints/hadl_distilled.pt",
                )
            except (SystemExit, Exception):
                pass

            self.assertEqual(os.environ.get("HADL_VLLM_ENABLED"), "1")
            self.assertEqual(os.environ.get("HADL_VLLM_VERSION"), "v4.5")
            self.assertEqual(os.environ.get("HADL_VLLM_K_STEPS"), "3")
            self.assertEqual(os.environ.get("HADL_VLLM_CHECKPOINT"), "checkpoints/hadl_distilled.pt")


class TestVLLMInferenceEngine(unittest.TestCase):
    def setUp(self):
        self.engine = VLLMInferenceEngine(
            model_id_or_path="Qwen/Qwen2.5-7B-Instruct",
            enable_hadl=True,
            hadl_version="v4.5",
            hadl_k_steps=2,
        )

    def test_initialization_defaults(self):
        self.assertEqual(self.engine.model_id, "Qwen/Qwen2.5-7B-Instruct")
        self.assertTrue(self.engine.enable_hadl)
        self.assertEqual(self.engine.hadl_version, "v4.5")
        self.assertEqual(self.engine.hadl_k_steps, 2)
        self.assertFalse(self.engine.is_loaded)

    def test_telemetry_reporting(self):
        telem = self.engine.get_telemetry()
        self.assertEqual(telem["engine"], "vllm")
        self.assertEqual(telem["model_id"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertTrue(telem["hadl_enabled"])
        self.assertEqual(telem["hadl_version"], "v4.5")
        self.assertEqual(telem["hadl_k_steps"], 2)
        self.assertEqual(telem["total_tokens_generated"], 0)

    def test_format_chat_prompt_fallback(self):
        messages = [
            {"role": "user", "content": "What is latent deliberation?"},
        ]
        formatted = self.engine.format_chat_prompt(messages)
        self.assertIn("<|im_start|>user", formatted)
        self.assertIn("What is latent deliberation?", formatted)
        self.assertIn("<|im_start|>assistant", formatted)

    def test_reset_memory_states_leakage_prevention(self):
        """Verify reset_memory_states clears all associative memory traces."""
        from dual_loop.plasticity import HeteroAssociativePlasticMemory

        cross_mem = HeteroAssociativePlasticMemory(d_model=32, rank=8)
        cross_mem.bind_concept(torch.randn(1, 4, 32), torch.randn(1, 4, 32))
        self.assertIsNotNone(cross_mem.last_m_cross)

        # Inject into engine's simulated adapter
        mock_adapter = MagicMock()
        mock_adapter.plastic_memory = cross_mem
        self.engine._hadl_adapter = mock_adapter

        self.engine.reset_memory_states(force=True)
        self.assertIsNone(cross_mem.last_m_cross)

    def test_generate_sync_and_stream_with_mocked_llm(self):
        """Verify synchronous and streaming generation using mocked vLLM LLM output."""
        mock_output = MagicMock()
        mock_inner = MagicMock()
        mock_inner.text = "Latent deliberation improves cognitive reasoning."
        mock_inner.token_ids = [101, 102, 103, 104, 105]
        mock_inner.finish_reason = "stop"
        mock_output.outputs = [mock_inner]
        mock_output.prompt_token_ids = [1, 2, 3]

        self.engine.llm = MagicMock()
        self.engine.llm.generate.return_value = [mock_output]
        self.engine.is_loaded = True

        # Test generate_sync
        res = self.engine.generate_sync("Explain HADL.", max_tokens=64)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["text"], "Latent deliberation improves cognitive reasoning.")
        self.assertEqual(res[0]["completion_tokens"], 5)
        self.assertEqual(res[0]["prompt_tokens"], 3)
        self.assertEqual(res[0]["engine"], "vllm")
        self.assertEqual(res[0]["hadl_version"], "v4.5")

        # Test generate_stream
        chunks = list(self.engine.generate_stream("Explain HADL.", max_tokens=64))
        full_stream_text = "".join(chunks)
        self.assertEqual(full_stream_text, "Latent deliberation improves cognitive reasoning.")


class TestHADLLogitsProcessor(unittest.TestCase):
    def test_deliberation_logits_processor(self):
        """Verifies HADLDeliberationLogitsProcessor applies repetition dampening."""
        proc = HADLDeliberationLogitsProcessor(repetition_penalty=1.20)
        input_ids = [10, 20, 30]
        logits = torch.zeros(50)
        logits[30] = 5.0  # Already generated token
        logits[40] = 5.0  # Fresh token

        modified = proc(input_ids, logits)
        # Token 30 should be dampened (5.0 / 1.20 ≈ 4.167)
        self.assertLess(modified[30].item(), modified[40].item())
        self.assertAlmostEqual(modified[40].item(), 5.0, places=4)

    def test_create_hadl_logits_processor_factory(self):
        proc = create_hadl_logits_processor(repetition_penalty=1.15)
        self.assertIsInstance(proc, HADLDeliberationLogitsProcessor)
        self.assertEqual(proc.repetition_penalty, 1.15)


class TestVLLMOfflineHelpers(unittest.TestCase):
    @patch("dual_loop.server.vllm_engine.VLLMInferenceEngine.load_model")
    @patch("dual_loop.server.vllm_engine.VLLMInferenceEngine.generate_sync")
    def test_vllm_batch_generate_and_compare(self, mock_gen, mock_load):
        mock_gen.return_value = [
            {"text": "Response 1", "tokens_per_second": 120.0, "total_tokens": 50, "prompt_tokens": 10, "completion_tokens": 40, "finish_reason": "stop"}
        ]

        # Test vllm_batch_generate
        results = vllm_batch_generate("mock-model", prompts="Hello world", enable_hadl=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Response 1")

        # Test vllm_compare_base_vs_hadl
        comp = vllm_compare_base_vs_hadl("mock-model", prompts=["Hello"], hadl_k_steps=2)
        self.assertIn("base_results", comp)
        self.assertIn("hadl_results", comp)
        self.assertIn("comparison", comp)
        self.assertEqual(comp["comparison"]["hadl_k_steps"], 2)


if __name__ == "__main__":
    unittest.main()
