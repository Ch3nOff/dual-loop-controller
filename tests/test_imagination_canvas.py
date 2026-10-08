"""
Unit Tests for Emergent Latent Imagination Canvas (ELIC / H-DREAM)
==================================================================
Verifies:
1. Free-form fragment unrolling and dimensional stability (3D and 2D).
2. Continuous associative flux field & multi-step dream cycles.
3. Energy coherence evaluation (no NaNs, bounded variance).
4. Strict contractive boundedness (||Delta_dream|| bounded).
5. End-to-end integration within HADLDynamicRuntime.
"""

import unittest
import torch
import torch.nn as nn

from dual_loop.imagination_canvas import EmergentLatentCanvas
from dual_loop.runtime.dynamic_runtime import HADLDynamicRuntime


class DummyBlock(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        return (self.linear(x),)


class DummyModel(nn.Module):
    def __init__(self, num_layers=4, d_model=64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyBlock(d_model) for _ in range(num_layers)])
        
    def forward(self, x):
        h = x
        for layer in self.model.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return h


class TestImaginationCanvas(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.L = 16
        self.d_model = 64
        self.num_fragments = 8
        self.canvas = EmergentLatentCanvas(
            d_model=self.d_model,
            num_fragments=self.num_fragments,
            dream_steps=2,
            contraction_factor=0.20
        )

    def test_shape_preservation_3d_and_2d(self):
        """Verifies shape preservation for both 3D sequential and 2D pooled representations."""
        # 3D tensor: (B, L, D)
        h_3d = torch.randn(self.B, self.L, self.d_model)
        delta_3d, energy_3d = self.canvas(h_3d, return_energy=True)
        self.assertEqual(delta_3d.shape, (self.B, self.L, self.d_model))
        self.assertIsNotNone(energy_3d)
        self.assertGreaterEqual(energy_3d.item(), 0.0)
        
        # 2D tensor: (B, D)
        h_2d = torch.randn(self.B, self.d_model)
        delta_2d, energy_2d = self.canvas(h_2d, return_energy=True)
        self.assertEqual(delta_2d.shape, (self.B, self.d_model))
        self.assertIsNotNone(energy_2d)
        self.assertGreaterEqual(energy_2d.item(), 0.0)

    def test_multi_step_dream_cycles_boundedness(self):
        """Verifies that multi-step contractive evolution does not explode numerically."""
        for steps in [1, 2, 4]:
            c = EmergentLatentCanvas(
                d_model=self.d_model,
                num_fragments=4,
                dream_steps=steps,
                contraction_factor=0.20
            )
            h = torch.randn(self.B, self.L, self.d_model)
            delta, energy = c(h, return_energy=True)
            self.assertFalse(torch.isnan(delta).any(), f"NaN detected at dream_steps={steps}")
            self.assertFalse(torch.isinf(delta).any(), f"Inf detected at dream_steps={steps}")
            # Delta norm should be contractively bounded
            self.assertLess(torch.norm(delta).item(), 100.0)

    def test_energy_coherence_computation(self):
        """Verifies that the latent dispersion energy measures mental dissonance properly."""
        h = torch.randn(self.B, self.L, self.d_model)
        _, energy = self.canvas(h, return_energy=True)
        self.assertTrue(torch.is_tensor(energy))
        self.assertFalse(torch.isnan(energy))
        self.assertGreater(energy.item(), 0.0)

    def test_adaptive_fragment_divisor(self):
        """Verifies that EmergentLatentCanvas adapts when d_model is not divisible by 8."""
        # 12 is divisible by 4, 2, 1 but not 8
        canvas_adaptive = EmergentLatentCanvas(d_model=12, num_fragments=8)
        self.assertEqual(canvas_adaptive.num_fragments, 4)
        self.assertEqual(canvas_adaptive.d_frag, 3)
        h = torch.randn(2, 4, 12)
        out, _ = canvas_adaptive(h)
        self.assertEqual(out.shape, (2, 4, 12))

    def test_runtime_integration_with_canvas(self):
        """Verifies end-to-end execution of HADLDynamicRuntime with active canvas."""
        base_model = DummyModel(num_layers=4, d_model=self.d_model)
        runtime = HADLDynamicRuntime(base_model=base_model, device="cpu")
        
        runtime.attach()
        self.assertTrue(runtime._active)
        self.assertTrue(hasattr(runtime, "canvas"))
        
        # Pass input through base model with hooks
        x = torch.randn(self.B, self.L, self.d_model)
        out = base_model(x)
        self.assertEqual(out.shape, (self.B, self.L, self.d_model))
        self.assertFalse(torch.isnan(out).any())
        
        runtime.detach()
        self.assertFalse(runtime._active)


    def test_confusion_tolerance_gating(self):
        """Verifies that imagination is only triggered when confusion exceeds tolerance."""
        canvas = EmergentLatentCanvas(
            d_model=self.d_model,
            num_fragments=self.num_fragments,
            confusion_tolerance=0.50,
            enable_confusion_gating=True
        )
        # Mock low confusion by setting bias negative
        canvas.w_confusion.bias.data.fill_(-10.0)
        h_clear = torch.randn(self.B, self.L, self.d_model)
        delta_clear, energy_clear, conf_clear = canvas(h_clear, return_confusion=True)
        # Should be bypassed (all zeros)
        self.assertTrue((delta_clear == 0.0).all())
        self.assertTrue((conf_clear < 0.50).all())
        
        # Mock high confusion by setting bias positive and non-zero collapse projection
        canvas.w_collapse.weight.data.normal_(0.0, 0.02)
        canvas.w_confusion.bias.data.fill_(10.0)
        h_confused = torch.randn(self.B, self.L, self.d_model)
        delta_confused, energy_confused, conf_confused = canvas(h_confused, return_confusion=True)
        # Should engage dream cycles (non-zero delta)
        self.assertTrue((conf_confused > 0.50).all())
        self.assertFalse((delta_confused == 0.0).all())


if __name__ == "__main__":
    unittest.main()
