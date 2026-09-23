import unittest
import torch
from dual_loop.curiosity_daemon import (
    EpistemicHumilityModule,
    IntrinsicCuriosityModule,
    PopperianSelfPlayEngine,
    AutonomousDaemonController
)

class TestCuriosityDaemon(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 128
        self.humility = EpistemicHumilityModule(d_model=self.d_model, max_confidence=0.95, min_vacuity=0.05)
        self.curiosity_icm = IntrinsicCuriosityModule(d_model=self.d_model, d_feature=64, d_action=32)
        self.self_play = PopperianSelfPlayEngine(d_model=self.d_model)
        self.daemon = AutonomousDaemonController(d_model=self.d_model)

    def test_bounded_confidence_and_vacuity(self):
        h = torch.randn(10, self.d_model)
        out = self.humility(h)
        
        conf = out["confidence"]
        vac = out["vacuity"]
        
        # Bounded confidence: model can NEVER be 100% confident
        self.assertTrue(torch.all(conf <= 0.95 + 1e-5), f"Max conf exceeded: {conf.max()}")
        # Minimum vacuity: perpetual curiosity opening
        self.assertTrue(torch.all(vac >= 0.05 - 1e-5), f"Min vacuity violated: {vac.min()}")

    def test_asymmetric_overconfidence_penalty(self):
        """
        Tests the hyperbolic arrogance penalty:
        Arrogant model (c=0.95) that is wrong should suffer > 100x penalty compared
        to a modest model (c=0.25).
        """
        c_arrogant = torch.tensor([0.95])
        c_modest = torch.tensor([0.25])
        error = torch.tensor([1.0])
        
        loss_arrogant = self.humility.compute_humility_loss(c_arrogant, error)
        loss_modest = self.humility.compute_humility_loss(c_modest, error)
        
        self.assertGreater(loss_arrogant.item(), 300.0) # (0.95 / 0.05)^2 = 361
        self.assertLess(loss_modest.item(), 0.20)       # (0.25 / 0.75)^2 = 0.11
        
        # Zero loss if prediction was correct
        correct = torch.tensor([0.0])
        self.assertEqual(self.humility.compute_humility_loss(c_arrogant, correct).item(), 0.0)

    def test_intrinsic_curiosity_module(self):
        s_t = torch.randn(4, self.d_model)
        a_t = torch.randn(4, 32)
        s_next = torch.randn(4, self.d_model)
        
        r_c, inv_loss, fwd_loss = self.curiosity_icm(s_t, a_t, s_next)
        
        self.assertEqual(r_c.shape, (4,))
        self.assertTrue(torch.all(r_c >= 0.0))
        self.assertGreater(inv_loss.item(), 0.0)
        self.assertGreater(fwd_loss.item(), 0.0)

    def test_popperian_self_play_and_sandbox(self):
        mem_a = torch.randn(1, self.d_model)
        mem_b = torch.randn(1, self.d_model)
        
        hyp = self.self_play.propose_hypothesis(mem_a, mem_b)
        self.assertEqual(hyp.shape, (1, self.d_model))
        
        counter_ex = self.self_play.generate_counter_example(hyp, mem_a)
        self.assertEqual(counter_ex.shape, (1, self.d_model))
        
        # Deterministic sandbox verification
        valid_code = "x = 10 + 20\ny = x * 2"
        is_ok, msg = PopperianSelfPlayEngine.verify_sandbox(valid_code, "syntax_check")
        self.assertTrue(is_ok)
        
        invalid_code = "def broken(:\n    pass"
        is_bad, bad_msg = PopperianSelfPlayEngine.verify_sandbox(invalid_code, "syntax_check")
        self.assertFalse(is_bad)
        self.assertIn("SyntaxError", bad_msg)

    def test_autonomous_daemon_step(self):
        # Create synthetic memory slots with deliberate opposition (contradiction)
        slots = torch.randn(6, self.d_model)
        # Create an anomaly: slot 1 is strongly opposed to slot 0
        slots[1] = -slots[0] * 1.5
        
        step_res = self.daemon.run_daemon_step(slots)
        
        self.assertIn("contradictions_detected", step_res)
        self.assertGreaterEqual(step_res["contradictions_detected"], 1)
        self.assertIn("anomalies_resolved", step_res)
        self.assertEqual(step_res["anomalies_resolved"], 1)
        self.assertIn("curiosity_reward", step_res)
        self.assertIn("humility_loss", step_res)
        self.assertEqual(self.daemon.resolved_anomalies_count, 1)

if __name__ == "__main__":
    unittest.main()
