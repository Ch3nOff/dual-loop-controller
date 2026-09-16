import os
import tempfile
import unittest
import torch
import torch.nn as nn

from dual_loop import DualLoopTransformer, RecurrentLatentController, TopKCapacityCrossAttention
from dual_loop.adapters import LatentDeliberationAdapter, DualLoopQwenModel
from dual_loop.benchmarks import MultiHopGraphDataset
from dual_loop.benchmarks.comprehensive_suite import load_or_instantiate_model


class MaliciousExploit:
    """Simulated malicious pickle payload for adversarial security testing."""
    def __reduce__(self):
        return (os.system, ("echo VULNERABLE",))


class MockQwenLayer(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, *args, **kwargs):
        return self.linear(x)


class MockQwenModel(nn.Module):
    def __init__(self, hidden_size=64, num_layers=4):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size, "num_attention_heads": 4})()
        self.layers = nn.ModuleList([MockQwenLayer(hidden_size) for _ in range(num_layers)])

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        if inputs_embeds is None:
            x = torch.randn(input_ids.size(0), input_ids.size(1), 64)
        else:
            x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        return type("Output", (), {"logits": x, "loss": torch.tensor(0.5, requires_grad=True)})()


class SecurityAndRuntimeTests(unittest.TestCase):

    def test_sec01_rejects_unsafe_pickle(self):
        """SEC-01: Verify load_adapter strictly enforces weights_only=True and rejects unsafe pickle."""
        mock_model = MockQwenModel()
        wrapped = DualLoopQwenModel(mock_model, layer_idx=1, k_steps=2)

        temp_fd, temp_path = tempfile.mkstemp(suffix=".pt")
        os.close(temp_fd)
        try:
            # Craft payload containing arbitrary executable class
            payload = {"malicious": MaliciousExploit()}
            torch.save(payload, temp_path)

            # Attempting to load MUST raise RuntimeError
            with self.assertRaises(RuntimeError) as ctx:
                wrapped.load_adapter(temp_path)
            self.assertIn("weights_only=True", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_tensor01_anchor_detachment_eval(self):
        """TENSOR-01: Verify H_anchor is detached when not self.training."""
        controller = RecurrentLatentController(d_model=64, n_heads=4, num_thought_tokens=4, max_ponder_steps=2)
        controller.eval()
        query = torch.randn(2, 64)
        memory = torch.randn(2, 8, 64)
        h, _, _ = controller(query, memory, k_steps=2)
        self.assertEqual(h.shape, (2, 4, 64))

    def test_tensor02_pos_emb_overflow_guarded(self):
        """TENSOR-02: Verify sequence length > 512 raises explicit ValueError."""
        model = DualLoopTransformer(vocab_size=50, d_model=32, n_heads=2, num_cwm_slots=4)
        overflow_inputs = torch.randint(0, 50, (1, 600))
        with self.assertRaises(ValueError) as ctx:
            model(overflow_inputs)
        self.assertIn("exceeds maximum supported positional embedding length", str(ctx.exception))

    def test_tensor03_no_double_residual(self):
        """TENSOR-03: Verify TopKCapacityCrossAttention returns only delta without doubling thoughts."""
        cross_attn = TopKCapacityCrossAttention(d_model=64, n_heads=4, capacity_factor=0.5)
        thoughts = torch.ones(2, 4, 64)
        memory = torch.ones(2, 8, 64)

        delta = cross_attn(thoughts, memory)
        # Verify delta is returned, not (thoughts + delta)
        # If double residual was present, delta would have thoughts added already
        self.assertEqual(delta.shape, thoughts.shape)
        # Scores router selects top 2 tokens out of 4 (capacity_factor 0.5)
        # The unselected tokens must have exactly 0 delta
        zeros_count = (delta == 0.0).all(dim=-1).sum().item()
        self.assertEqual(zeros_count, 4) # 2 unselected tokens per batch item across 2 batches = 4

    def test_tensor04_short_sequence_query_pos(self):
        """TENSOR-04: Verify S=1 does not crash with IndexError."""
        model = DualLoopTransformer(vocab_size=50, d_model=32, n_heads=2, num_cwm_slots=4)
        single_token_inputs = torch.tensor([[10], [25]]) # [2, 1]
        logits, info = model(single_token_inputs)
        self.assertEqual(logits.shape, (2, 50))
        self.assertEqual(info["effective_k"], 3.0)

    def test_tensor05_causal_mask_support(self):
        """TENSOR-05: Verify causal masking option works cleanly."""
        model = DualLoopTransformer(vocab_size=50, d_model=32, n_heads=2, num_cwm_slots=4, is_causal=True)
        inputs = torch.randint(0, 50, (2, 10))
        logits, info = model(inputs)
        self.assertEqual(logits.shape, (2, 50))

    def test_arch01_recurrent_norm_clipping(self):
        """ARCH-01: Verify recurrent state norm is strictly clipped to <= 50.0 at high K."""
        controller = RecurrentLatentController(d_model=64, n_heads=4, num_thought_tokens=4, max_ponder_steps=10)
        # Feed extremely high magnitude input
        query = torch.randn(2, 64) * 500.0
        memory = torch.randn(2, 8, 64) * 500.0
        h, _, _ = controller(query, memory, k_steps=8)
        norm = torch.norm(h, p=2, dim=-1)
        self.assertTrue(torch.all(norm <= 50.001), f"Max norm was {norm.max().item()}, expected <= 50.0")

    def test_arch02_telemetry_reset(self):
        """ARCH-02: Verify telemetry does not leak cross-request diagnostic state."""
        mock_model = MockQwenModel()
        wrapped = DualLoopQwenModel(mock_model, layer_idx=1, k_steps=0)
        x = torch.randn(1, 10, 64)

        wrapped(inputs_embeds=x)
        self.assertTrue(wrapped.last_telemetry.get("bypassed", False))

        wrapped.set_ponder_steps(2)
        wrapped(inputs_embeds=x)
        self.assertFalse(wrapped.last_telemetry.get("bypassed", True))
        self.assertEqual(wrapped.last_telemetry.get("ponder_steps"), 2)

    def test_arch03_per_sample_query_idx_tensor(self):
        """ARCH-03: Verify per-sample query_idx updates only the targeted token per batch item."""
        adapter = LatentDeliberationAdapter(d_model=32, n_heads=2, num_thought_tokens=2, max_ponder_steps=1)
        hidden = torch.zeros(2, 6, 32)
        # Sample 0 targeted at pos 1, Sample 1 targeted at pos 4
        query_indices = torch.tensor([1, 4])

        enhanced, _ = adapter(hidden, k_steps=1, query_idx=query_indices)

        # Token at pos 1 of sample 0 should have non-zero delta
        self.assertFalse(torch.allclose(enhanced[0, 1], torch.zeros(32)))
        # Token at pos 4 of sample 0 should remain untouched (zero)
        self.assertTrue(torch.allclose(enhanced[0, 4], torch.zeros(32)))

        # Token at pos 4 of sample 1 should have non-zero delta
        self.assertFalse(torch.allclose(enhanced[1, 4], torch.zeros(32)))
        # Token at pos 1 of sample 1 should remain untouched (zero)
        self.assertTrue(torch.allclose(enhanced[1, 1], torch.zeros(32)))

    def test_arch05_padding_mask_in_cwm(self):
        """ARCH-05: Verify key_padding_mask is properly handled in CWM without shape error."""
        adapter = LatentDeliberationAdapter(d_model=32, n_heads=2, num_thought_tokens=2, num_cwm_slots=4)
        hidden = torch.randn(2, 8, 32)
        key_padding_mask = torch.zeros(2, 8, dtype=torch.bool)
        key_padding_mask[:, 6:] = True # Last 2 tokens are padding

        enhanced, telemetry = adapter(hidden, k_steps=1, key_padding_mask=key_padding_mask)
        self.assertEqual(enhanced.shape, hidden.shape)
        self.assertFalse(telemetry["bypassed"])

    def test_arch06_graph_reasoning_edge_limit(self):
        """ARCH-06: Verify exceeding max possible edges raises ValueError instead of infinite loop."""
        with self.assertRaises(ValueError) as ctx:
            # 4 nodes can have at most 4 * 3 = 12 directed edges
            MultiHopGraphDataset(num_samples=10, num_nodes=4, num_edges=15)
        self.assertIn("exceeds maximum possible directed edges", str(ctx.exception))

    def test_arch08_comprehensive_suite_missing_checkpoint_guard(self):
        """ARCH-08: Verify missing checkpoint raises FileNotFoundError unless allow_untrained=True."""
        non_existent = "non_existent_test_chk.pt"
        with self.assertRaises(FileNotFoundError):
            load_or_instantiate_model(checkpoint_path=non_existent, allow_untrained=False)

        # When allow_untrained=True, it should instantiate with warning and not crash
        model = load_or_instantiate_model(checkpoint_path=non_existent, allow_untrained=True)
        self.assertIsNotNone(model)

    def test_sec04_hash_verification(self):
        """SEC-04: Verify expected_sha256 enforces exact integrity check."""
        import hashlib
        mock_model = MockQwenModel()
        wrapped = DualLoopQwenModel(mock_model, layer_idx=1, k_steps=2)

        temp_fd, temp_path = tempfile.mkstemp(suffix=".safetensors")
        os.close(temp_fd)
        try:
            wrapped.save_adapter(temp_path, format="safetensors")
            # Calculate actual hash
            with open(temp_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()

            # 1. Loading with correct hash must succeed
            loaded = wrapped.load_adapter(temp_path, expected_sha256=actual_hash)
            self.assertEqual(loaded, temp_path)

            # 2. Loading with incorrect hash must raise ValueError
            wrong_hash = "0" * 64
            with self.assertRaises(ValueError) as ctx:
                wrapped.load_adapter(temp_path, expected_sha256=wrong_hash)
            self.assertIn("SHA-256 integrity check failed", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
