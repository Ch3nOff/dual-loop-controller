import unittest
import torch
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter

class TestBrainSandboxIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.adapter = LatentDeliberationAdapter(
            d_model=self.d_model,
            n_heads=4,
            num_thought_tokens=4,
            max_ponder_steps=2,
            enable_homeostasis=True,
            enable_nullspace_projection=True,
            enable_mdl_selection=True,
            enable_functorial_mapping=True,
            enable_brain_sandbox=True
        )

    def test_streaming_syntax_automatic_bypass(self):
        """Streaming single tokens (S=1) must trigger automatic homeostatic bypass (k=0)."""
        h_token = torch.randn(1, 1, self.d_model)
        enhanced, telem = self.adapter(h_token, k_steps=2)
        
        # Must be exact identity bypass
        self.assertTrue(torch.equal(enhanced, h_token))
        self.assertTrue(telem["bypassed"])
        self.assertEqual(telem["homeostasis"]["optimal_k"], 0)
        self.assertEqual(telem["homeostasis"]["policy"], "pi_0_syntax_bypass")

    def test_prompt_pondering_active(self):
        """Multi-token prompts (S > 1) with uncertainty must deliberate and populate MDL/telemetry."""
        h_prompt = torch.randn(1, 12, self.d_model)
        enhanced, telem = self.adapter(h_prompt, k_steps=2)
        
        self.assertFalse(telem["bypassed"])
        self.assertGreater(telem["ponder_steps"], 0)
        self.assertIn("homeostasis", telem)
        self.assertIn("mdl_telemetry", telem)
        self.assertIn("functorial_telemetry", telem)
        self.assertIn("total_mdl", telem["mdl_telemetry"])
        self.assertTrue(telem["functorial_telemetry"]["functor_applied"])

    def test_strict_backward_compatibility(self):
        """Adapter instantiated with legacy defaults (all HADL flags False) maintains exact legacy behavior."""
        legacy_adapter = LatentDeliberationAdapter(
            d_model=self.d_model,
            n_heads=4,
            num_thought_tokens=4,
            max_ponder_steps=2
        )
        h_prompt = torch.randn(1, 8, self.d_model)
        enhanced, telem = legacy_adapter(h_prompt, k_steps=2)
        
        self.assertFalse(telem["bypassed"])
        self.assertEqual(telem["homeostasis"], {})
        self.assertEqual(telem["mdl_telemetry"], {})
        self.assertEqual(telem["functorial_telemetry"], {})

if __name__ == "__main__":
    unittest.main()
