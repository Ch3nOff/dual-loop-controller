"""
Unit tests for Dual-Loop Inference Engine Server & VRAM Auto-Tuner
"""

import unittest
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from dual_loop.server.vram_tuner import VRAMAutoTuner, HardwareProfile, ModelSpec, AllocationPlan
from dual_loop.server.engine import DualLoopInferenceEngine
from dual_loop.server.app import create_app


class TestVRAMAutoTuner(unittest.TestCase):
    def test_hardware_profiling(self):
        hw = VRAMAutoTuner.profile_hardware()
        self.assertIsInstance(hw, HardwareProfile)
        self.assertGreaterEqual(hw.total_vram_gib, 0.0)
        self.assertGreaterEqual(hw.host_ram_total_gib, 0.0)

    def test_estimate_model_spec_known(self):
        spec_7b = VRAMAutoTuner.estimate_model_spec("Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(spec_7b.hidden_size, 3584)
        self.assertAlmostEqual(spec_7b.parameters_billion, 7.61, delta=0.5)
        self.assertGreater(spec_7b.vram_bf16_gib, 14.0)

        spec_3b = VRAMAutoTuner.estimate_model_spec("Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(spec_3b.hidden_size, 2048)
        self.assertAlmostEqual(spec_3b.parameters_billion, 3.40, delta=0.5)

    def test_calculate_plan_auto_and_custom_headroom(self):
        # Test plan generation with custom 4.0 GiB headroom
        plan = VRAMAutoTuner.calculate_plan("Qwen/Qwen2.5-7B-Instruct", headroom_gib=4.0)
        self.assertEqual(plan.requested_headroom_gib, 4.0)
        self.assertIn(plan.selected_regime, ["BF16", "INT8", "NF4"])
        self.assertTrue(plan.enable_allostasis)

    def test_forced_regime(self):
        plan = VRAMAutoTuner.calculate_plan("Qwen/Qwen2.5-7B-Instruct", forced_regime="NF4")
        self.assertEqual(plan.selected_regime, "NF4")
        self.assertTrue(plan.load_in_4bit)
        self.assertFalse(plan.load_in_8bit)


class TestInferenceServerAPI(unittest.TestCase):
    def setUp(self):
        # Setup mock engine to avoid loading real GPU weights during fast unit testing
        self.mock_engine = MagicMock(spec=DualLoopInferenceEngine)
        self.mock_engine.model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.mock_engine.k_steps = 2
        self.mock_engine.total_tokens_generated = 128
        self.mock_engine.total_generation_time_sec = 2.5

        hw = HardwareProfile(
            device_id=0,
            device_name="NVIDIA GeForce RTX 4070 (Simulated)",
            total_vram_gib=12.0,
            free_vram_gib=11.2,
            host_ram_total_gib=32.0,
            host_ram_avail_gib=24.0,
            cuda_available=True
        )
        spec = VRAMAutoTuner.estimate_model_spec("Qwen/Qwen2.5-7B-Instruct")
        self.mock_engine.plan = AllocationPlan(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            hardware=hw,
            model_spec=spec,
            requested_headroom_gib=4.0,
            effective_budget_gib=8.0,
            selected_regime="NF4",
            estimated_model_vram_gib=4.19,
            projected_free_vram_gib=7.81,
            enable_allostasis=True,
            torch_dtype="bfloat16",
            load_in_4bit=True,
            load_in_8bit=False,
            description="4-bit NormalFloat (NF4) quantization + Dual-Loop Deliberation"
        )

        self.mock_engine.format_chat_prompt.side_effect = lambda msgs: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        self.mock_engine.generate_sync.return_value = {
            "text": "Hello! I am Dual-Loop Deliberative Agent.",
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
            "elapsed_seconds": 0.15,
            "tokens_per_second": 53.3,
            "regime": "NF4"
        }
        self.mock_engine.generate_stream.return_value = iter(["Hello", "! ", "I ", "am ", "Dual-Loop."])

        app = create_app(self.mock_engine)
        self.client = TestClient(app)

    def test_root_dashboard(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Dual-Loop Inference Engine", resp.text)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", resp.text)

    def test_models_endpoint(self):
        resp = self.client.get("/v1/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["id"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(data["data"][0]["dual_loop_regime"], "NF4")
        self.assertEqual(data["data"][0]["vram_headroom_gib"], 4.0)

    def test_health_endpoint(self):
        resp = self.client.get("/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["regime"], "NF4")
        self.assertEqual(data["vram_headroom_reserved_gib"], 4.0)

    def test_chat_completions_non_streaming(self):
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "user", "content": "Explain active inference."}
            ],
            "stream": False
        }
        resp = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["content"], "Hello! I am Dual-Loop Deliberative Agent.")
        self.assertEqual(data["usage"]["total_tokens"], 18)
        self.assertIn("dual_loop_telemetry", data)

    def test_chat_completions_streaming(self):
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "user", "content": "Explain active inference."}
            ],
            "stream": True
        }
        resp = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers["content-type"])
        chunks = resp.text.split("\n\n")
        self.assertTrue(any("data: [DONE]" in c for c in chunks))
        self.assertTrue(any("Hello" in c for c in chunks))

    def test_legacy_completions(self):
        payload = {
            "prompt": "Once upon a time",
            "max_tokens": 50
        }
        resp = self.client.post("/v1/completions", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["object"], "text_completion")
        self.assertEqual(data["choices"][0]["text"], "Hello! I am Dual-Loop Deliberative Agent.")

    def test_api_key_authentication(self):
        # SEC-01: When api_key is configured, unauthenticated requests must be rejected with 401
        protected_app = create_app(self.mock_engine, api_key="secret-token-xyz")
        protected_client = TestClient(protected_app)

        # 1. Health check should remain accessible
        resp_health = protected_client.get("/health")
        self.assertEqual(resp_health.status_code, 200)

        # 2. Protected endpoint without token should return 401
        resp_unauth = protected_client.get("/v1/models")
        self.assertEqual(resp_unauth.status_code, 401)
        self.assertIn("invalid_api_key", resp_unauth.text)

        # 3. Protected endpoint with invalid token should return 401
        resp_bad = protected_client.get("/v1/models", headers={"Authorization": "Bearer wrong-token"})
        self.assertEqual(resp_bad.status_code, 401)

        # 4. Protected endpoint with valid Bearer token should succeed
        resp_ok = protected_client.get("/v1/models", headers={"Authorization": "Bearer secret-token-xyz"})
        self.assertEqual(resp_ok.status_code, 200)

        # 5. Protected endpoint with valid x-api-key header should succeed
        resp_key_ok = protected_client.get("/v1/models", headers={"x-api-key": "secret-token-xyz"})
        self.assertEqual(resp_key_ok.status_code, 200)

    def test_cors_configuration(self):
        # SEC-01: Test CORS middleware default headers
        resp = self.client.options(
            "/v1/models",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"}
        )
        self.assertEqual(resp.status_code, 200)
        # When allow_origins=["*"], allow-credentials must NOT be true
        self.assertNotEqual(resp.headers.get("access-control-allow-credentials"), "true")


if __name__ == "__main__":
    unittest.main()

