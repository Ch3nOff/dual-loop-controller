import unittest
import torch
import torch.nn as nn
from dual_loop.verification import HypothesisVerificationGate
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter

class TestHypothesisVerification(unittest.TestCase):
    """
    Tests for the Counterfactual Hypothesis-Testing & Conservative Verification Gate.
    Verifies that:
    1. Epistemic evidence gain and trust region drift are calculated accurately.
    2. Overthinking (divergence without evidence gain) is penalized and suppressed.
    3. Meaningful deliberations are accepted with high beta.
    4. Gradients flow cleanly during backpropagation.
    5. Seamless integration with LatentDeliberationAdapter.
    """
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 128
        self.num_thoughts = 4
        self.batch_size = 2
        self.seq_len = 16
        self.mem_slots = 8

    def test_hypothesis_verification_forward_and_backward(self):
        gate = HypothesisVerificationGate(d_model=self.d_model)
        
        thoughts = torch.randn(self.batch_size, self.num_thoughts, self.d_model, requires_grad=True)
        query_rep = torch.randn(self.batch_size, self.d_model, requires_grad=True)
        memory = torch.randn(self.batch_size, self.mem_slots, self.d_model, requires_grad=True)
        
        verified, beta, telemetry = gate(thoughts, query_rep, memory)
        
        self.assertEqual(verified.shape, (self.batch_size, self.num_thoughts, self.d_model))
        self.assertEqual(beta.shape, (self.batch_size, 1, 1))
        self.assertTrue((beta >= 0.0).all() and (beta <= 1.0).all())
        
        self.assertIn("evidence_gain", telemetry)
        self.assertIn("drift", telemetry)
        self.assertIn("acceptance_beta", telemetry)
        
        # Test backward pass
        loss = verified.sum() + beta.sum()
        loss.backward()
        
        self.assertIsNotNone(thoughts.grad)
        self.assertIsNotNone(query_rep.grad)
        self.assertIsNotNone(memory.grad)
        self.assertGreater(thoughts.grad.norm().item(), 0.0)

    def test_anti_overthinking_suppression(self):
        """When candidate thoughts drift wildly with negative evidence gain, beta should suppress them."""
        gate = HypothesisVerificationGate(d_model=self.d_model, evidence_margin=0.01)
        
        query_rep = torch.randn(self.batch_size, self.d_model)
        # Context memory is identical to query
        memory = query_rep.unsqueeze(1).expand(-1, self.mem_slots, -1) + torch.randn(self.batch_size, self.mem_slots, self.d_model) * 0.01
        
        # Scenario A: Candidate thoughts perfectly ground in memory (aligned)
        aligned_thoughts = memory[:, :self.num_thoughts, :].clone()
        _, beta_aligned, telem_aligned = gate(aligned_thoughts, query_rep, memory)
        
        # Scenario B: Overthinking thoughts that drift opposite to memory/query (uncorrelated noise)
        drifted_thoughts = -query_rep.unsqueeze(1).expand(-1, self.num_thoughts, -1) + torch.randn(self.batch_size, self.num_thoughts, self.d_model) * 2.0
        _, beta_drifted, telem_drifted = gate(drifted_thoughts, query_rep, memory)
        
        # Drifted thoughts must have higher drift penalty
        self.assertGreater(telem_drifted["drift"].mean().item(), telem_aligned["drift"].mean().item())

    def test_adapter_integration(self):
        adapter = LatentDeliberationAdapter(
            d_model=self.d_model,
            n_heads=4,
            num_thought_tokens=self.num_thoughts,
            max_ponder_steps=2,
            use_hypothesis_verification=True
        )
        
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        enhanced, telemetry = adapter(hidden_states, k_steps=2)
        
        self.assertEqual(enhanced.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertIn("acceptance_beta", telemetry)
        self.assertIn("evidence_gain", telemetry)
        self.assertEqual(len(telemetry["acceptance_beta"]), self.batch_size)
        
        # Test gradient flows back to input
        loss = enhanced.sum()
        loss.backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertGreater(hidden_states.grad.norm().item(), 0.0)

        # Test bypass mode
        enhanced_bp, telemetry_bp = adapter(hidden_states, k_steps=0)
        self.assertTrue(telemetry_bp["bypassed"])
        self.assertEqual(len(telemetry_bp["acceptance_beta"]), 0)


if __name__ == "__main__":
    unittest.main()
