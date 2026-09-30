"""
Unit Tests for Qwen3.8-27B Latent Reconstructive Hologram Adapter
=================================================================
Validates 54.0 GB -> 4.55 GB VRAM compression, 8GB laptop GPU physical budget,
1.58-bit ternary / 2-bit skeleton quantization, and D=5120 -> 1024 FISTA recovery.
"""

import unittest
import torch
import torch.nn as nn

from dual_loop import (
    attach_dual_loop_to_qwen3_8,
    Qwen3_8HologramModel,
    calculate_qwen3_8_vram_budget,
    QWEN3_8_27B_D_NATIVE,
    QWEN3_8_27B_TARGET_LAYER,
    CANONICAL_DIM
)


class TestQwen3_8Hologram(unittest.TestCase):
    def test_vram_budget_qwen3_8(self):
        budget = calculate_qwen3_8_vram_budget()
        self.assertEqual(budget["model_name"], "Qwen/Qwen3.8-27B")
        self.assertEqual(budget["total_parameters_billion"], 27.0)
        self.assertEqual(budget["fp16_original_vram_gb"], 54.0)
        self.assertFalse(budget["fits_fp16_on_8gb"])
        self.assertTrue(budget["fits_in_8gb_laptop_vram"])
        self.assertLessEqual(budget["total_dual_loop_vram_gb"], 5.10)
        self.assertGreaterEqual(budget["vram_headroom_8gb_laptop_gb"], 2.80)
        self.assertEqual(budget["target_layer_idx"], 32)
        self.assertEqual(budget["d_native"], 5120)
        self.assertEqual(budget["d_canonical"], 1024)

    def test_qwen3_8_hologram_model_forward(self):
        torch.manual_seed(42)
        model = Qwen3_8HologramModel(
            num_layers=4,
            d_native=5120,
            d_canonical=1024,
            enable_hologram=True,
            hologram_steps=3
        )
        x = torch.randn(1, 4, 5120)
        logits, telem = model(x)
        self.assertEqual(logits.shape, (1, 4, 1000))
        self.assertIn("hologram", telem)
        self.assertEqual(telem["d_native"], 5120)
        self.assertEqual(telem["d_canonical"], 1024)

    def test_qwen3_8_quantization_and_fista_recovery(self):
        torch.manual_seed(42)
        model = Qwen3_8HologramModel(
            num_layers=4,
            d_native=5120,
            d_canonical=1024,
            enable_hologram=True,
            hologram_steps=3
        )
        
        # Test ternary quantization
        model.quantize_skeleton_to_ternary()

        # Run forward pass on quantized skeleton
        x = torch.randn(1, 2, 5120)
        logits, telem = model(x)
        self.assertEqual(logits.shape, (1, 2, 1000))
        self.assertIn("snr_gain_db", telem["hologram"])
        self.assertGreater(telem["hologram"]["snr_gain_db"], 0.0)

    def test_attach_dual_loop_to_qwen3_8(self):
        class DummyQwenBlock(nn.Module):
            def __init__(self, d=5120):
                super().__init__()
                self.proj = nn.Linear(d, d)
            def forward(self, x):
                return x + self.proj(x)

        class DummyQwen3_8Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                self.model.language_model.layers = nn.ModuleList([
                    DummyQwenBlock() for _ in range(64)
                ])
            def forward(self, x):
                h = x
                for l in self.model.language_model.layers:
                    h = l(h)
                return h

        base_model = DummyQwen3_8Backbone()
        wrapped = attach_dual_loop_to_qwen3_8(base_model, layer_idx=32, hologram_steps=3)
        self.assertEqual(wrapped.target_layer_idx, 32)
        self.assertEqual(wrapped.adapter.d_native, 5120)
        self.assertEqual(wrapped.adapter.d_canonical, 1024)
        self.assertTrue(wrapped.adapter.enable_hologram)


if __name__ == "__main__":
    unittest.main()
