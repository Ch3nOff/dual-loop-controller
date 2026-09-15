import unittest
import torch
import torch.nn as nn
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter

class TestLatentAdapterIntegration(unittest.TestCase):
    """
    Tests integration of LatentDeliberationAdapter with standard LLM dimensions.
    Simulates injecting the adapter into a standard pretrained Transformer backbone.
    """
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 64
        # Standard hidden dimension of small-to-medium foundation models (e.g. Qwen-2.5-0.5B / Llama-3.2-1B)
        self.d_model = 768
        self.num_thoughts = 4

    def test_prefix_mode_forward_and_backward(self):
        adapter = LatentDeliberationAdapter(
            d_model=self.d_model,
            n_heads=8,
            num_thought_tokens=self.num_thoughts,
            max_ponder_steps=3,
            adapter_mode="prefix"
        )
        
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        enhanced, telemetry = adapter(hidden_states, k_steps=3)
        
        # In prefix mode: sequence length extends by num_thoughts
        expected_seq_len = self.seq_len + self.num_thoughts
        self.assertEqual(enhanced.shape, (self.batch_size, expected_seq_len, self.d_model))
        self.assertEqual(telemetry["ponder_steps"], 3)
        
        # Test gradient propagation back to input hidden states
        loss = enhanced.sum()
        loss.backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertGreater(hidden_states.grad.norm().item(), 0.0)

    def test_residual_mode_forward_and_backward(self):
        adapter = LatentDeliberationAdapter(
            d_model=self.d_model,
            n_heads=8,
            num_thought_tokens=self.num_thoughts,
            max_ponder_steps=3,
            adapter_mode="residual"
        )
        
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        enhanced, telemetry = adapter(hidden_states, k_steps=3)
        
        # In residual mode: sequence length remains exactly identical (plug-and-play for fixed KV caches)
        self.assertEqual(enhanced.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Verify delta applied to last token
        delta = (enhanced[:, -1, :] - hidden_states[:, -1, :]).abs().sum()
        self.assertGreater(delta.item(), 0.0)
        
        # Verify gradient flows to adapter parameters
        loss = enhanced.sum()
        loss.backward()
        self.assertIsNotNone(adapter.residual_proj[0].weight.grad)
        self.assertGreater(adapter.residual_proj[0].weight.grad.norm().item(), 0.0)

    def test_simulated_llm_pipeline(self):
        """
        Simulates an LLM with 4 layers where the adapter is inserted
        between Layer 2 and Layer 3 (mid-network deliberation).
        """
        layer_early = nn.TransformerEncoderLayer(self.d_model, 8, self.d_model * 2, batch_first=True)
        adapter = LatentDeliberationAdapter(self.d_model, n_heads=8, adapter_mode="residual")
        layer_late = nn.TransformerEncoderLayer(self.d_model, 8, self.d_model * 2, batch_first=True)
        lm_head = nn.Linear(self.d_model, 1000)

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Layer 1..2 (Sensory encoding)
        h = layer_early(x)
        # Mid-network Latent Deliberation (System 2 thinking)
        h_deliberated, _ = adapter(h, k_steps=2)
        # Layer 3..4 (Action/language generation)
        h_final = layer_late(h_deliberated)
        logits = lm_head(h_final[:, -1, :])

        self.assertEqual(logits.shape, (self.batch_size, 1000))
        print(f"\n[OK] Simulated LLM Pipeline with LatentDeliberationAdapter executed successfully.")
        print(f"     Hidden dimension: {self.d_model}, Deliberation steps: 2, Logits shape: {logits.shape}")

if __name__ == "__main__":
    unittest.main()
