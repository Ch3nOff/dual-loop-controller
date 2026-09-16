import unittest
import torch
import torch.nn as nn
from dual_loop.controller import RecurrentLatentController, LatentCritiqueRefinementUnit
from dual_loop.halting import LearnedHaltingGate
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter

class TestMetacognitiveLoop(unittest.TestCase):
    """
    Comprehensive tests for:
    1. LatentCritiqueRefinementUnit (discrepancy detection & corrective residual injection).
    2. LearnedHaltingGate (multi-signal halting & PonderNet geometric prior KL loss).
    3. Contraction anchor stabilization across deep ponder steps (K=4, 6, 8).
    4. Telemetry and adapter propagation.
    """
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 128
        self.n_heads = 4
        self.num_thoughts = 4
        self.batch_size = 2
        self.seq_len = 16
        self.mem_slots = 8

    def test_latent_critique_refinement_unit(self):
        unit = LatentCritiqueRefinementUnit(d_model=self.d_model)
        thoughts = torch.randn(self.batch_size, self.num_thoughts, self.d_model, requires_grad=True)
        cross_delta = torch.randn(self.batch_size, self.num_thoughts, self.d_model)
        
        delta_correct, error_norm = unit(thoughts, cross_delta)
        
        self.assertEqual(delta_correct.shape, (self.batch_size, self.num_thoughts, self.d_model))
        self.assertEqual(error_norm.shape, (self.batch_size,))
        self.assertTrue((error_norm > 0).all())
        
        # Test gradient propagation
        loss = delta_correct.sum() + error_norm.sum()
        loss.backward()
        self.assertIsNotNone(thoughts.grad)
        self.assertGreater(thoughts.grad.norm().item(), 0.0)

    def test_learned_halting_gate_pondernet_loss(self):
        gate = LearnedHaltingGate(
            d_model=self.d_model,
            lambda_prior=0.5,
            tau_halt=0.75,
            vocab_size=100
        )
        
        # Simulate 4 deliberation steps
        lambdas = []
        step_logits = []
        target = torch.randint(0, 100, (self.batch_size,))
        
        h_prev = None
        H_prev = None
        for step in range(4):
            h_curr = torch.randn(self.batch_size, self.d_model)
            H_curr = torch.randn(self.batch_size, self.num_thoughts, self.d_model)
            logits = torch.randn(self.batch_size, 100)
            step_logits.append(logits)
            
            lam = gate.compute_lambda(
                h_curr=h_curr,
                h_prev=h_prev,
                H_curr=H_curr,
                H_prev=H_prev,
                logits=logits,
                discrepancy_norm=torch.tensor([1.2, 0.8])
            )
            lambdas.append(lam)
            h_prev = h_curr
            H_prev = H_curr
            
            # Check lambda bounds in (0, 1)
            self.assertTrue((lam > 0.0).all())
            self.assertTrue((lam < 1.0).all())

        loss, p_n, kl_loss, rec_loss = gate.compute_pondernet_loss(
            lambdas=lambdas,
            step_logits=step_logits,
            target=target
        )
        
        self.assertEqual(loss.dim(), 0)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(kl_loss.item(), 0.0)
        self.assertGreater(rec_loss.item(), 0.0)
        
        # Verify halting probabilities sum to 1 per sample
        p_total = p_n.sum(dim=1) # [B]
        self.assertTrue(torch.allclose(p_total, torch.ones_like(p_total), atol=1e-4))

    def test_recurrent_controller_with_critique_and_learned_halting(self):
        controller = RecurrentLatentController(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_model * 2,
            num_thought_tokens=self.num_thoughts,
            max_ponder_steps=4,
            vocab_size=100,
            enable_critique=True,
            use_learned_halting=True
        )
        
        query_rep = torch.randn(self.batch_size, self.d_model, requires_grad=True)
        memory = torch.randn(self.batch_size, self.mem_slots, self.d_model)
        
        H, aux_logits, step_entropies = controller(
            query_rep=query_rep,
            memory=memory,
            k_steps=4,
            return_aux=True
        )
        
        self.assertEqual(H.shape, (self.batch_size, self.num_thoughts, self.d_model))
        self.assertEqual(len(controller.last_error_norms), 4)
        self.assertEqual(len(controller.last_lambdas), 4)
        
        # Backprop verification
        loss = H.sum() + sum(l.sum() for l in aux_logits)
        loss.backward()
        self.assertIsNotNone(query_rep.grad)
        self.assertGreater(query_rep.grad.norm().item(), 0.0)

    def test_deep_ponder_stability(self):
        """Verify contractive anchor recurrence at K=4, 6, 8 steps."""
        controller = RecurrentLatentController(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_model * 2,
            num_thought_tokens=self.num_thoughts,
            max_ponder_steps=8,
            enable_critique=True,
            use_learned_halting=False
        )
        
        query_rep = torch.randn(self.batch_size, self.d_model)
        memory = torch.randn(self.batch_size, self.mem_slots, self.d_model)
        
        for k in [4, 6, 8]:
            H, _, _ = controller(query_rep=query_rep, memory=memory, k_steps=k)
            # Norm must stay bounded and finite (clipping max_norm is 50.0)
            norms = H.norm(dim=-1)
            self.assertTrue(torch.isfinite(H).all(), f"NaN/Inf detected at K={k}")
            self.assertTrue((norms <= 50.1).all(), f"Norm explosion detected at K={k}: {norms.max().item()}")

    def test_adapter_telemetry_critique(self):
        adapter = LatentDeliberationAdapter(
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_thought_tokens=self.num_thoughts,
            max_ponder_steps=3,
            enable_critique=True,
            use_learned_halting=True
        )
        
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.d_model)
        enhanced, telemetry = adapter(hidden_states, k_steps=3)
        
        self.assertEqual(enhanced.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertIn("error_norms", telemetry)
        self.assertIn("halting_lambdas", telemetry)
        self.assertEqual(len(telemetry["error_norms"]), 3)
        self.assertEqual(len(telemetry["halting_lambdas"]), 3)
        
        # Test bypass mode
        enhanced_bypass, telemetry_bypass = adapter(hidden_states, k_steps=0)
        self.assertTrue(telemetry_bypass["bypassed"])
        self.assertEqual(len(telemetry_bypass["error_norms"]), 0)
        self.assertEqual(len(telemetry_bypass["halting_lambdas"]), 0)


if __name__ == "__main__":
    unittest.main()
