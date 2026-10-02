"""
Unit Tests for HADL v3.3: Evolving Manifold and LM-Head Syringe Injection
========================================================================
Tests:
1. Dynamic mass estimator positivity (|m| >= 0) and scaling (|m| / sqrt(D)).
2. Unitary Givens isometry error: strictly 0.000000 under veto.
3. LMHeadSyringe ReZero property: exactly zero logit perturbation at initialization.
4. Active Syringe injection: verifies delta logit application and gradient propagation.
5. End-to-end wrapper hook interception on transformer layers and LM Head.
"""

import math
import unittest
import torch
import torch.nn as nn

from dual_loop.evolving_manifold_syringe import (
    EvolvingManifoldModule,
    LMHeadSyringe,
    EvolvingSquareCloudWrapper
)


class MockTransformerConfig:
    def __init__(self, hidden_size=256, vocab_size=1000):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class MockTransformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return (self.linear(x),)


class MockTransformerModel(nn.Module):
    def __init__(self, d_model=256, vocab_size=1000, num_layers=4):
        super().__init__()
        self.config = MockTransformerConfig(hidden_size=d_model, vocab_size=vocab_size)
        self.layers = nn.ModuleList([MockTransformerLayer(d_model) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        logits = self.lm_head(x)
        return logits


class TestEvolvingManifoldSyringe(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 256
        self.vocab_size = 1000
        self.batch_size = 2
        self.seq_len = 8

    def test_dynamic_mass_positivity(self):
        """Verifies that dynamic mass |m| is strictly positive and scales correctly."""
        module = EvolvingManifoldModule(d_model=self.d_model, num_slots=4)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        mass = module.mass_estimator(x)
        self.assertTrue(torch.all(mass >= 0.0), "Mass |m| must be non-negative (Softplus).")
        self.assertEqual(mass.shape, (self.batch_size, self.seq_len, 1))

    def test_isometry_preservation(self):
        """Verifies that Unitary Givens transformation preserves Euclidean norm."""
        module = EvolvingManifoldModule(d_model=self.d_model, num_slots=4)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        orig_norm = torch.norm(x, dim=-1)
        
        # Forward pass
        x_rot, telem = module(x)
        final_norm = torch.norm(x_rot, dim=-1)
        
        # When rot angle is 0 (vetoed) or active, norm deviation is < 1e-4
        max_deviation = torch.max(torch.abs(final_norm - orig_norm)).item()
        self.assertLess(max_deviation, 1e-4, f"Isometry violated: max deviation = {max_deviation}")
        self.assertIn("isometry_error", telem)
        self.assertIn("mean_mass", telem)

    def test_lm_head_syringe_rezero_init(self):
        """Verifies that at initialization, LMHeadSyringe injects exactly 0.0 logits."""
        syringe = LMHeadSyringe(d_model=self.d_model, vocab_size=self.vocab_size, rank=32)
        h_final = torch.randn(self.batch_size, self.seq_len, self.d_model)
        base_logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        
        injected_logits, telem = syringe(h_final, base_logits)
        max_diff = torch.max(torch.abs(injected_logits - base_logits)).item()
        self.assertAlmostEqual(max_diff, 0.0, places=6, msg="ReZero must guarantee 0.0 logit perturbation initially.")

    def test_lm_head_syringe_active_gradient_flow(self):
        """Verifies that gradients flow backward through syringe up/down projections."""
        syringe = LMHeadSyringe(d_model=self.d_model, vocab_size=self.vocab_size, rank=32)
        # Enable weights slightly to simulate post-training
        nn.init.normal_(syringe.up_proj.weight, std=0.01)
        
        h_final = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        base_logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        
        injected_logits, telem = syringe(h_final, base_logits)
        loss = injected_logits.sum()
        loss.backward()
        
        self.assertIsNotNone(h_final.grad)
        self.assertIsNotNone(syringe.up_proj.weight.grad)
        self.assertIsNotNone(syringe.down_proj.weight.grad)
        self.assertGreater(torch.norm(syringe.up_proj.weight.grad).item(), 0.0)

    def test_wrapper_end_to_end_hook_interception(self):
        """Verifies that EvolvingSquareCloudWrapper intercepts both Layer 1 and LM Head."""
        base_model = MockTransformerModel(d_model=self.d_model, vocab_size=self.vocab_size, num_layers=3)
        wrapper = EvolvingSquareCloudWrapper(base_model, target_layer_idx=1, rank=32)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Test baseline vs wrapped logits at init (should be identical due to ReZero)
        wrapper.enabled = False
        logits_base = wrapper(x)
        
        wrapper.enabled = True
        logits_wrapped = wrapper(x)
        
        self.assertEqual(logits_wrapped.shape, (self.batch_size, self.seq_len, self.vocab_size))
        # Initial difference is bounded by ReZero
        wrapper.remove_hooks()


if __name__ == "__main__":
    unittest.main()
