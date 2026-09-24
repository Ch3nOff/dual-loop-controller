import os
import json
import unittest
import torch
import torch.nn as nn

from dual_loop.runtime.detector import (
    detect_architecture_family,
    determine_optimal_hook,
    determine_optimal_bottleneck,
    auto_attach_hadl,
    create_mock_detected_model,
    MockTransformerBackbone,
)
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
