import unittest
import torch
import torch.nn as nn
from dual_loop.evidential import EvidentialEpistemicGate
from dual_loop.plasticity import PlasticFastWeightUnit
from dual_loop.open_concept import OpenConceptSynthesizer
from dual_loop.controller import RecurrentLatentController
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter

class TestPlasticityAndEvidential(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.L = 4
        self.S = 8
        self.D = 64

    def test_evidential_conservation_law(self):
        """Verify Subjective Logic conservation law: sum(beliefs) + u == 1.0."""
        gate = EvidentialEpistemicGate(d_model=self.D, num_evidence_classes=8)
        h = torch.randn(self.B, self.D)
        
        telem = gate(h)
        u = telem["vacuity_u"] # [B]
        b = telem["beliefs"]   # [B, M]
        
        # Check conservation law: sum_m(b_m) + u == 1.0 (within numerical float precision)
        total_mass = b.sum(dim=-1) + u
        ones = torch.ones_like(total_mass)
        self.assertTrue(torch.allclose(total_mass, ones, atol=1e-5))
        
        # Check bounds: u in [0, 1]
        self.assertTrue((u >= 0.0).all() and (u <= 1.0).all())

    def test_evidential_novelty_detection(self):
        """Verify evidential gate flags zero/flat evidence as high vacuity (OOD)."""
        gate = EvidentialEpistemicGate(d_model=self.D, num_evidence_classes=8, tau_novelty=0.30)
        h = torch.randn(self.B, self.D)
        
        # Scenario A: Flat near-zero logits (zero evidence -> maximum vacuity u ~ 1.0)
        flat_logits = torch.zeros(self.B, 8)
        telem_flat = gate(h, logits_external=flat_logits)
        # alpha = softplus(0) + 1 = 0.693 + 1 = 1.693; S = 8 * 1.693 = 13.54; u = 8 / 13.54 ~ 0.59 > 0.30
        self.assertTrue(telem_flat["is_novel"].all())
        
        # Scenario B: Highly decisive evidence (S >> M -> vacuity u ~ 0.0)
        decisive_logits = torch.zeros(self.B, 8)
        decisive_logits[:, 0] = 50.0 # Extreme evidence on class 0
        telem_decisive = gate(h, logits_external=decisive_logits)
        self.assertFalse(telem_decisive["is_novel"].any())
        self.assertLess(float(telem_decisive["vacuity_u"].mean().item()), 0.20)

    def test_plastic_fast_weight_update(self):
        """Verify in-situ Hebbian update adapts virtual weights without autograd."""
        unit = PlasticFastWeightUnit(d_model=self.D, rank=16, plastic_lr=0.20, decay_rate=0.05)
        thoughts = torch.randn(self.B, self.L, self.D)
        discrepancy = torch.randn(self.B, self.L, self.D)
        u_epistemic = torch.tensor([0.8, 0.9]) # High vacuity -> active plasticity
        
        # Initial step: trace starts at zero
        self.assertIsNone(unit.last_m_fast)
        delta_1, telem_1 = unit(thoughts, discrepancy, u_epistemic=u_epistemic)
        
        self.assertEqual(delta_1.shape, (self.B, self.L, self.D))
        self.assertIsNotNone(unit.last_m_fast)
        trace_norm_1 = telem_1["m_fast_norm"]
        self.assertGreater(trace_norm_1, 0.0)
        
        # Subsequent step: fast weights accumulate memory trace
        delta_2, telem_2 = unit(thoughts, discrepancy, u_epistemic=u_epistemic)
        trace_norm_2 = telem_2["m_fast_norm"]
        self.assertGreater(trace_norm_2, 0.0)
        
        # Reset state: guarantees zero cross-request leakage
        unit.reset_state()
        self.assertIsNone(unit.last_m_fast)
        self.assertEqual(unit.last_update_norm, 0.0)

    def test_open_concept_synthesizer(self):
        """Verify continuous prototype synthesis activates only when u >= tau_unseen."""
        synth = OpenConceptSynthesizer(d_model=self.D, tau_unseen=0.60)
        h_anchor = torch.randn(self.B, self.D)
        discrepancy = torch.randn(self.B, self.D)
        
        # Case A: Low vacuity (familiar concept -> no prototype generated)
        low_u = torch.tensor([0.2, 0.3])
        proto_none, telem_low = synth(h_anchor, discrepancy, low_u)
        self.assertIsNone(proto_none)
        self.assertFalse(telem_low["is_active"])
        self.assertEqual(telem_low["synthesized_count"], 0)
        
        # Case B: High vacuity (unprecedented concept -> prototype synthesized)
        high_u = torch.tensor([0.8, 0.9])
        proto_high, telem_high = synth(h_anchor, discrepancy, high_u)
        self.assertIsNotNone(proto_high)
        self.assertEqual(proto_high.shape, (self.B, 1, self.D))
        self.assertTrue(telem_high["is_active"])
        self.assertEqual(telem_high["synthesized_count"], 2)
        self.assertFalse(torch.isnan(proto_high).any())

    def test_controller_with_plasticity(self):
        """Verify RecurrentLatentController executes with plastic virtual parameters."""
        controller = RecurrentLatentController(
            d_model=self.D,
            n_heads=2,
            d_ff=self.D * 2,
            num_thought_tokens=self.L,
            max_ponder_steps=3,
            enable_plasticity=True,
            plastic_rank=16
        )
        query_rep = torch.randn(self.B, self.D)
        memory = torch.randn(self.B, 4, self.D)
        u_epistemic = torch.tensor([0.7, 0.7])
        
        H, aux, entropies = controller(
            query_rep=query_rep,
            memory=memory,
            k_steps=2,
            u_epistemic=u_epistemic
        )
        self.assertEqual(H.shape, (self.B, self.L, self.D))
        self.assertIn("m_fast_norm", controller.last_plastic_telemetry)
        
        # Check reset
        controller.reset_state()
        self.assertEqual(len(controller.last_plastic_telemetry), 0)

    def test_latent_adapter_end_to_end(self):
        """Verify full LatentDeliberationAdapter forward pass with epistemic and plastic modules."""
        adapter = LatentDeliberationAdapter(
            d_model=self.D,
            n_heads=2,
            num_thought_tokens=self.L,
            max_ponder_steps=2,
            num_cwm_slots=4,
            use_evidential_gate=True,
            use_open_concept=True,
            enable_plasticity=True,
            plastic_rank=16
        )
        hidden_states = torch.randn(self.B, self.S, self.D)
        
        enhanced, telem = adapter(hidden_states=hidden_states, query_idx=-1)
        self.assertEqual(enhanced.shape, (self.B, self.S, self.D))
        
        # Verify new diagnostic telemetry keys
        self.assertIn("epistemic_vacuity", telem)
        self.assertIn("plastic_trace_norm", telem)
        self.assertIn("synthesized_concepts", telem)
        self.assertEqual(len(telem["epistemic_vacuity"]), self.B)
        self.assertFalse(torch.isnan(enhanced).any())

if __name__ == "__main__":
    unittest.main()
