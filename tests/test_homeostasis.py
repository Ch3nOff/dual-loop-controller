import unittest
import torch
from dual_loop.homeostasis import HomeostaticDriveEngine, ActiveInferencePolicyRouter

class TestHomeostasis(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.engine = HomeostaticDriveEngine(p_norm=2.0)
        self.router = ActiveInferencePolicyRouter(d_model=128)

    def test_ideal_setpoint_zero_drive(self):
        """At ideal setpoint S*, drive penalty should be exactly 0.0."""
        s_star = self.engine.S_star.unsqueeze(0)
        drive = self.engine.compute_drive(s_star)
        self.assertAlmostEqual(drive.item(), 0.0, places=5)

    def test_drive_increases_with_perturbation(self):
        """Deviating from S* must strictly increase drive penalty."""
        s_star = self.engine.S_star.unsqueeze(0)
        perturbed = s_star.clone()
        perturbed[0, 1] = 0.90 # high epistemic entropy
        drive = self.engine.compute_drive(perturbed)
        self.assertGreater(drive.item(), 0.0)

    def test_streaming_syntax_bypass_rule(self):
        """During single-token streaming with low uncertainty, router MUST select k=0 bypass."""
        h_stream = torch.randn(1, 1, 128)
        k, telem = self.router.select_policy(
            h_current=h_stream,
            vacuity_u=0.05, # low uncertainty (e.g. typing ';')
            is_token_streaming=True,
            base_k_steps=2
        )
        self.assertEqual(k, 0)
        self.assertEqual(telem["policy"], "pi_0_syntax_bypass")

    def test_high_uncertainty_pondering(self):
        """With high uncertainty at prompt stage, router MUST select k > 0 to resolve entropy."""
        h_prompt = torch.randn(1, 10, 128)
        k, telem = self.router.select_policy(
            h_current=h_prompt,
            vacuity_u=0.85, # high uncertainty
            is_token_streaming=False,
            base_k_steps=2
        )
        self.assertGreater(k, 0)
        self.assertIn(telem["policy"], ["pi_1_focused_deliberation", "pi_2_brain_sandbox"])

    def test_energy_depletion_and_recovery(self):
        """Pondering consumes energy; bypass recovers energy."""
        self.engine.reset_physiological_state()
        self.assertEqual(self.engine.current_energy, 1.0)
        
        # Take 2 steps
        self.engine.update_state_after_action(2)
        self.assertLess(self.engine.current_energy, 1.0)
        
        # Bypass 1 step
        prev_energy = self.engine.current_energy
        self.engine.update_state_after_action(0)
        self.assertGreater(self.engine.current_energy, prev_energy)

if __name__ == "__main__":
    unittest.main()
