import unittest
import tempfile
import os
import torch
import torch.nn as nn
from dual_loop import attach_dual_loop, LatentDeliberationAdapter, DualLoopQwenModel


class MockLargeTransformer(nn.Module):
    """Simulates a large transformer backbone (e.g. GLM-4-9B, D=4096, 40 layers)."""
    def __init__(self, vocab_size: int = 2000, hidden_size: int = 4096, num_layers: int = 8):
        super().__init__()
        self.config = type("Config", (), {
            "hidden_size": hidden_size,
            "num_attention_heads": 32,
            "vocab_size": vocab_size
        })()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU()
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return type("Output", (), {"logits": logits, "loss": loss})()


class TestBottleneckAdapter(unittest.TestCase):
    """
    Comprehensive test suite for Bottleneck Latent Deliberation.
    Validates parameter compression, gradient propagation, and zero-init stability.
    """
    def setUp(self):
        torch.manual_seed(42)

    def test_parameter_compression_ratio(self):
        """Verify bottleneck_dim=1024 achieves >90% parameter reduction for D=4096."""
        ad_full = LatentDeliberationAdapter(d_model=4096, bottleneck_dim=None)
        p_full = sum(p.numel() for p in ad_full.parameters())
        
        ad_bottle = LatentDeliberationAdapter(d_model=4096, bottleneck_dim=1024)
        p_bottle = sum(p.numel() for p in ad_bottle.parameters())

        self.assertEqual(ad_bottle.d_model, 4096)
        self.assertEqual(ad_bottle.d_inner, 1024)
        self.assertEqual(ad_bottle.bottleneck_dim, 1024)
        self.assertIsNotNone(ad_bottle.down_proj)
        self.assertIsNotNone(ad_bottle.up_proj)
        
        # Verify > 90% parameter reduction
        reduction_pct = (1.0 - p_bottle / p_full) * 100.0
        self.assertGreater(reduction_pct, 90.0)
        self.assertLess(p_bottle, 65_000_000) # Under 65M parameters

    def test_zero_init_identity_at_initialization(self):
        """Verify up_proj zero-init guarantees delta is 0 at step 0 of training."""
        adapter = LatentDeliberationAdapter(d_model=4096, bottleneck_dim=1024, adapter_mode="residual")
        h = torch.randn(2, 10, 4096)
        
        with torch.no_grad():
            enhanced, telemetry = adapter(h, k_steps=2)
            
        # Delta must be 0 because up_proj weights & bias are initialized to zero
        self.assertTrue(torch.allclose(enhanced, h, atol=1e-6))
        self.assertEqual(telemetry["bottleneck_dim"], 1024)
        self.assertEqual(telemetry["d_inner"], 1024)

    def test_gradient_flow_through_bottleneck(self):
        """Verify end-to-end backprop flows gradients through up_proj, inner controller, and down_proj."""
        adapter = LatentDeliberationAdapter(d_model=4096, bottleneck_dim=1024, adapter_mode="residual")
        # Nudge up_proj so gradients flow back
        with torch.no_grad():
            adapter.up_proj.weight.fill_(0.01)
            
        h = torch.randn(2, 8, 4096, requires_grad=True)
        enhanced, _ = adapter(h, k_steps=2)
        
        loss = enhanced.sum()
        loss.backward()
        
        self.assertIsNotNone(adapter.up_proj.weight.grad)
        self.assertGreater(adapter.up_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(adapter.down_proj.weight.grad)
        self.assertGreater(adapter.down_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(adapter.controller.capacity_cross_attn.mha.in_proj_weight.grad)
        self.assertGreater(adapter.controller.capacity_cross_attn.mha.in_proj_weight.grad.abs().sum().item(), 0.0)

    def test_universal_attach_on_large_mock_model(self):
        """Verify attach_dual_loop seamlessly attaches bottleneck adapter to D=4096 model."""
        base_model = MockLargeTransformer(vocab_size=1000, hidden_size=4096, num_layers=6)
        wrapped = attach_dual_loop(base_model, layer_idx=3, k_steps=2, bottleneck_dim=1024)
        
        self.assertIsInstance(wrapped, DualLoopQwenModel)
        summary = wrapped.get_parameter_summary()
        self.assertEqual(summary["hidden_size"], 4096)
        self.assertEqual(summary["bottleneck_dim"], 1024)
        self.assertEqual(summary["d_inner"], 1024)
        self.assertLess(summary["adapter_parameters"], 65_000_000)

        # Run forward pass with input_ids and labels
        input_ids = torch.randint(0, 1000, (2, 8))
        labels = torch.randint(0, 1000, (2, 8))
        out = wrapped(input_ids=input_ids, labels=labels)
        
        self.assertIsNotNone(out.logits)
        self.assertEqual(out.logits.shape, (2, 8, 1000))
        self.assertIsNotNone(out.loss)
        self.assertGreater(out.loss.item(), 0.0)
        
        wrapped.remove_hook()

    def test_save_and_load_bottleneck_adapter(self):
        """Verify saving and loading weights preserves bottleneck down_proj and up_proj."""
        base_model = MockLargeTransformer(vocab_size=500, hidden_size=4096, num_layers=4)
        wrapped = attach_dual_loop(base_model, layer_idx=2, k_steps=2, bottleneck_dim=1024)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_bottleneck_adapter.safetensors")
            wrapped.save_adapter(save_path)
            self.assertTrue(os.path.isfile(save_path))
            
            # Create a second fresh instance
            wrapped2 = attach_dual_loop(base_model, layer_idx=2, k_steps=2, bottleneck_dim=1024)
            wrapped2.load_adapter(save_path)
            
            # Check parameter equality
            for p1, p2 in zip(wrapped.adapter.parameters(), wrapped2.adapter.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
                
        wrapped.remove_hook()
        wrapped2.remove_hook()


if __name__ == "__main__":
    unittest.main()
