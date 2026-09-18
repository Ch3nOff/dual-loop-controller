import unittest
import numpy as np
from dual_loop import ContextDirectionalRouter, CompactCommonSenseReservoir, ProbabilisticCognitiveJudge

class TestDirectionalReservoir(unittest.TestCase):
    def setUp(self):
        self.router = ContextDirectionalRouter(latent_dim=64)
        self.reservoir = CompactCommonSenseReservoir(latent_dim=64)
        self.judge = ProbabilisticCognitiveJudge()

    def test_directional_routing_scientific_vs_commonsense(self):
        sci_prompt = "What is the chemical reaction and molecular equation for cellular respiration in mitochondria?"
        route_sci = self.router.route_context(sci_prompt)
        self.assertEqual(route_sci["direction"], "UP_SCIENTIFIC")
        self.assertGreater(route_sci["rho"], 0.0)
        self.assertLess(route_sci["alpha_cs"], 0.5)

        cs_prompt = "Which requires energy to move?"
        route_cs = self.router.route_context(cs_prompt)
        self.assertEqual(route_cs["direction"], "DOWN_COMMONSENSE")
        self.assertLessEqual(route_cs["rho"], 0.0)
        self.assertGreaterEqual(route_cs["alpha_cs"], 0.5)

    def test_compact_reservoir_locomotion_grounding(self):
        prompt = "Which requires energy to move?"
        choices = ["weasel", "willow", "mango", "poison ivy"]
        deltas = self.reservoir.compute_grounding_deltas(prompt, choices)
        
        # Weasel (animal active locomotion) must receive positive boost
        self.assertGreater(deltas[0], 1.5)
        # Willow, mango, poison ivy (plants) must not receive active locomotion boost
        self.assertLessEqual(deltas[1], 0.0)
        self.assertLessEqual(deltas[2], 0.0)
        self.assertLessEqual(deltas[3], 0.0)

    def test_memory_footprint_compactness(self):
        # Memory of M_cs prototype matrix must be under 50 KB
        n_bytes = self.reservoir.M_cs.nbytes
        self.assertLess(n_bytes, 50 * 1024)

    def test_judge_and_fuse_with_directional_grounding(self):
        # Simulation of OpenBookQA #18:
        # Base scores: A=-8.40759, B=-8.40907, C=-14.929, D=-5.713 (banned)
        # Deliberation scores: A=-7.5420, B=-5.9615, C=-13.826, D=-5.317
        prompt = "Which requires energy to move?"
        choices = ["weasel", "willow", "mango", "poison ivy"]
        labels = ["A", "B", "C", "D"]
        scores_base = [-8.40759, -8.40907, -14.929, -5.713]
        scores_delib = [-7.5420, -5.9615, -13.826, -5.317]
        
        # With directional reservoir active:
        res = self.judge.judge_and_fuse(
            scores_base=scores_base,
            scores_delib=scores_delib,
            labels=labels,
            banned_labels=["D"],
            prompt=prompt,
            choices=choices
        )
        # 'A' (weasel) should correctly win because common-sense locomotion grounds A!
        self.assertEqual(res["pred_label"], "A")
        self.assertEqual(res["direction"], "DOWN_COMMONSENSE")

if __name__ == "__main__":
    unittest.main()
