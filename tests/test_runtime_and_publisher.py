import os
import json
import unittest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from dual_loop.runtime.detector import (
    detect_architecture_family,
    determine_optimal_hook,
    determine_optimal_bottleneck,
    auto_attach_hadl,
    create_mock_detected_model,
    MockTransformerBackbone,
)
from dual_loop.runtime.server import app, initialize_engine, engine_state
from dual_loop.runtime.hf_publisher import package_hf_bundle


class TestModelAutoDetector(unittest.TestCase):
    def test_detect_architecture_families(self):
        cfg_glm = type("Config", (), {"model_type": "chatglm", "architectures": ["ChatGLMForConditionalGeneration"]})()
        self.assertEqual(detect_architecture_family(cfg_glm), "ChatGLM")

        cfg_qwen = type("Config", (), {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})()
        self.assertEqual(detect_architecture_family(cfg_qwen), "Qwen")

        cfg_llama = type("Config", (), {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})()
        self.assertEqual(detect_architecture_family(cfg_llama), "LLaMA")

        cfg_mistral = type("Config", (), {"model_type": "mistral", "architectures": ["MistralForCausalLM"]})()
        self.assertEqual(detect_architecture_family(cfg_mistral), "Mistral")

    def test_determine_optimal_bottleneck(self):
        # GLM-4 / Large model D=4096 -> compressed to d=1024 (saves >90% VRAM)
        self.assertEqual(determine_optimal_bottleneck(4096), 1024)
        self.assertEqual(determine_optimal_bottleneck(8192), 1024)
        
        # Qwen-2B / Small model D=2048 -> full width
        self.assertIsNone(determine_optimal_bottleneck(2048))
        self.assertIsNone(determine_optimal_bottleneck(1024))

    def test_determine_optimal_hook(self):
        backbone = MockTransformerBackbone(num_layers=40)
        hook = determine_optimal_hook(backbone, 40)
        self.assertEqual(hook, 20)

    def test_mock_model_auto_attach(self):
        res = create_mock_detected_model(family="ChatGLM", d_model=4096, num_layers=40, k_steps=2)
        self.assertEqual(res.family, "ChatGLM")
        self.assertEqual(res.hook_layer, 20)
        self.assertEqual(res.bottleneck_dim, 1024)
        self.assertIn("engine_version", res.manifest)
        self.assertEqual(res.manifest["engine_version"], "v2.4.0-autopoietic")


class TestRuntimeServerAndCockpit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_engine(mock=True)
        cls.client = TestClient(app)

    def test_cockpit_html_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("HADL Cognitive Runtime Cockpit", response.text)
        self.assertIn("The Latent Mind HUD", response.text)
        self.assertIn("Allostatic Energy Gauge", response.text)

    def test_models_endpoint(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "list")
        self.assertTrue(len(data["data"]) > 0)
        self.assertIn("hadl_manifest", data["data"][0])

    def test_chat_completions_telemetry(self):
        payload = {
            "model": "hadl-v24-autopoietic",
            "messages": [
                {"role": "user", "content": "Explain the autopoietic dual-loop architecture"}
            ],
            "stream": False,
            "k_steps": 2
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("choices", data)
        self.assertIn("hadl_telemetry", data)
        telem = data["hadl_telemetry"]
        self.assertEqual(telem["k_steps"], 2)
        self.assertEqual(telem["policy"], "pi_1_deliberation")
        self.assertIn("allostatic_energy", telem)
        self.assertEqual(telem["nullspace_leakage"], 0.000000)
        self.assertIn("zero-token", telem["token_bloat_saved"])

    def test_curiosity_dream_feed(self):
        # Trigger on-demand dream
        r_dream = self.client.post("/v1/dream")
        self.assertEqual(r_dream.status_code, 200)
        dream_data = r_dream.json()
        self.assertIn("state", dream_data)
        self.assertIn("cycle_latency_ms", dream_data)

        # Inspect chronological feed
        r_feed = self.client.get("/v1/dream/feed")
        self.assertEqual(r_feed.status_code, 200)
        feed = r_feed.json()
        self.assertIsInstance(feed, list)
        self.assertTrue(len(feed) > 0)


class TestHFHubPackaging(unittest.TestCase):
    def test_package_glm4_bundle(self):
        temp_out = os.path.abspath("dist/test_hf_glm4_bundle")
        bundle_dir = package_hf_bundle(model_type="glm4", output_dir=temp_out)
        self.assertTrue(os.path.isdir(bundle_dir))
        
        # Verify files
        self.assertTrue(os.path.isfile(os.path.join(bundle_dir, "adapter_model.safetensors")))
        self.assertTrue(os.path.isfile(os.path.join(bundle_dir, "adapter_config.json")))
        self.assertTrue(os.path.isfile(os.path.join(bundle_dir, "README.md")))
        
        with open(os.path.join(bundle_dir, "adapter_config.json"), "r") as f:
            cfg = json.load(f)
            self.assertEqual(cfg["architecture"], "ChatGLM")
            self.assertEqual(cfg["bottleneck_dim"], 1024)
            self.assertEqual(cfg["d_model"], 4096)

        with open(os.path.join(bundle_dir, "README.md"), "r") as f:
            readme = f.read()
            self.assertIn("zai-org/glm-4-9b-chat", readme)
            self.assertIn("3.76", readme)
            self.assertIn("0.000000", readme)


if __name__ == "__main__":
    unittest.main()
