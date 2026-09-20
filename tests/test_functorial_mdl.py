import unittest
import torch
from dual_loop.functorial_engine import FunctorialCrossDomainMapper
from dual_loop.mdl_selector import NeuroSymbolicMDLSelector

class TestFunctorialAndMDL(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.num_slots = 8
        self.functor = FunctorialCrossDomainMapper(d_model=self.d_model, num_slots=self.num_slots)
        self.mdl = NeuroSymbolicMDLSelector(d_model=self.d_model)

    def test_functorial_mapping_shape_and_commutativity(self):
        """Functorial mapper must compute relational morphisms and output structured analog."""
        B = 2
        source_memory = torch.randn(B, self.num_slots, self.d_model)
        target_anchor = torch.randn(B, self.d_model)
        
        analog, telem = self.functor(source_memory, target_anchor)
        
        self.assertEqual(analog.shape, (B, self.num_slots, self.d_model))
        self.assertIn("commutativity_error", telem)
        self.assertGreaterEqual(telem["commutativity_error"], 0.0)
        self.assertTrue(telem["functor_applied"])

    def test_mdl_selector_prefers_parsimonious_plan(self):
        """MDL must penalize noisy/over-engineered plans and select the compact, grounded plan."""
        B = 1
        context_memory = torch.randn(B, self.num_slots, self.d_model)
        
        # Plan A: compact, well-grounded (close to context_memory)
        plan_A = context_memory[:, :4, :].clone() # 4 tokens, zero constraint error
        
        # Plan B: bloated, noisy, high variance
        plan_B = torch.randn(B, 4, self.d_model) * 10.0 # high energy, massive error
        
        best_plan, best_idx, telem = self.mdl.select_best_candidate([plan_A, plan_B], context_memory)
        
        self.assertEqual(best_idx, 0)
        self.assertLess(telem["all_scores"][0], telem["all_scores"][1])

if __name__ == "__main__":
    unittest.main()
