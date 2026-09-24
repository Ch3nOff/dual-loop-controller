import unittest
import torch
import torch.nn as nn
from dual_loop.self_awareness import (
    IntrospectiveSelfDescriptor,
    EgoThoughtProjector,
    EpistemicModestyModulator,
    SelfAwarenessController,
    calculate_active_neurons
)

class TestSelfAwarenessSystem(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.D_inner = 1024
        self.L_thought = 4
        self.k_max = 4

    def test_introspective_self_descriptor_vector(self):
        descriptor = IntrospectiveSelfDescriptor()
        h_k = torch.randn(self.B, self.D_inner)
        h_anchor = torch.randn(self.B, self.D_inner)
        H_k = torch.randn(self.B, self.L_thought, self.D_inner)
        H_prev1 = torch.randn(self.B, self.L_thought, self.D_inner)
        H_prev2 = torch.randn(self.B, self.L_thought, self.D_inner)
        u_vacuity = torch.tensor([0.2, 0.85])
        logits = torch.randn(self.B, 100)

        s_t, metrics = descriptor(
            k=2,
            k_max=self.k_max,
            u_vacuity=u_vacuity,
            h_k=h_k,
            h_anchor=h_anchor,
            H_k=H_k,
            H_prev1=H_prev1,
            H_prev2=H_prev2,
            logits=logits
        )

        # Vector shape must be [B, 5]
        self.assertEqual(s_t.shape, (self.B, 5))
        # s_time = 2 / 4 = 0.5
        self.assertAlmostEqual(s_t[0, 0].item(), 0.5, places=4)
        # s_vacuity must match input
        self.assertAlmostEqual(s_t[0, 1].item(), 0.2, places=4)
        self.assertAlmostEqual(s_t[1, 1].item(), 0.85, places=4)
        # s_drift >= 0
        self.assertGreaterEqual(s_t[:, 2].min().item(), 0.0)
        # s_lipschitz >= 0
        self.assertGreaterEqual(s_t[:, 3].min().item(), 0.0)
        # Metrics populated
        self.assertIn("s_time", metrics)
        self.assertIn("s_vacuity", metrics)
        self.assertIn("s_drift", metrics)
        self.assertIn("s_lipschitz", metrics)
        self.assertIn("s_margin", metrics)

    def test_ego_thought_projector_and_slot0_injection(self):
        controller = SelfAwarenessController(d_inner=self.D_inner)
        H_k = torch.randn(self.B, self.L_thought, self.D_inner)
        H_anchor = torch.randn(self.B, self.L_thought, self.D_inner)

        H_aware, t_ego, telem = controller.step_self_awareness(
            k=1,
            k_max=self.k_max,
            H_k=H_k,
            H_anchor=H_anchor,
            u_vacuity=torch.tensor([0.1, 0.2])
        )

        # t_ego shape must be [B, 1, D_inner]
        self.assertEqual(t_ego.shape, (self.B, 1, self.D_inner))
        # H_aware shape must be preserved [B, L_thought, D_inner]
        self.assertEqual(H_aware.shape, (self.B, self.L_thought, self.D_inner))
        # Slot 0 must equal t_ego
        self.assertTrue(torch.allclose(H_aware[:, 0:1, :], t_ego, atol=1e-5))
        # Task slots (1, 2, 3) must be preserved from H_k
        self.assertTrue(torch.allclose(H_aware[:, 1:, :], H_k[:, 1:, :], atol=1e-5))
        # Telemetry check
        self.assertEqual(telem["slot_0_type"], "ego_thought_token")
        self.assertEqual(telem["task_slots_count"], 3)

    def test_epistemic_modesty_trigger(self):
        modulator = EpistemicModestyModulator()
        
        # Case A: Normal bounded state (vacuity 0.2, lipschitz 0.6)
        s_normal = torch.tensor([[0.5, 0.2, 0.1, 0.6, 2.0]])
        factor_norm, is_active_norm = modulator(s_normal)
        self.assertAlmostEqual(factor_norm.item(), 1.0, places=4)
        self.assertFalse(is_active_norm)

        # Case B: High ignorance (vacuity 0.9 > 0.7) and chaotic divergence (L_k 1.4 > 1.0)
        s_extreme = torch.tensor([[0.5, 0.9, 0.5, 1.4, -0.5]])
        factor_ext, is_active_ext = modulator(s_extreme)
        self.assertTrue(is_active_ext)
        self.assertLess(factor_ext.item(), 0.5) # Strongly damped

    def test_active_neurons_calculation(self):
        neurons = calculate_active_neurons(d_model=2048, d_inner=1024, k_steps=2)
        self.assertGreater(neurons["active_neurons_system2"], 100000)
        self.assertGreater(neurons["active_neurons_total"], 300000)
        self.assertIn("formatted", neurons)
        self.assertEqual(neurons["synapses_total"], 2310000000)

if __name__ == "__main__":
    unittest.main()
