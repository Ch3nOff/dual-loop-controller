import unittest
import numpy as np
from dual_loop import ProbabilisticCognitiveJudge

class TestProbabilisticCognitiveJudge(unittest.TestCase):
    def setUp(self):
        self.judge = ProbabilisticCognitiveJudge(
            cs_margin_threshold=0.35,
            base_lambda=0.85,
            intuitive_lambda=0.20,
            soft_penalty_weight=4.5,
            allow_belief_revision=True
        )

    def test_common_sense_decisive_margin(self):
        # Base clearly prefers C: C=-7.7, A=-8.15, B=-12.0
        scores_base = [-8.15, -12.0, -7.7, -14.0]
        # Margin between top unbanned (-7.7) and second (-8.15) is 0.45 >= 0.35
        margin, top_idx, is_decisive = self.judge.assess_common_sense_margin(scores_base, banned_indices=[1])
        self.assertTrue(is_decisive)
        self.assertEqual(top_idx, 2)
        self.assertAlmostEqual(margin, 0.45, places=2)

    def test_adaptive_lambda_scaling(self):
        # Decisive -> intuitive lambda
        lam_decisive = self.judge.compute_adaptive_lambda(margin=0.50, is_decisive=True)
        self.assertAlmostEqual(lam_decisive, 0.20, places=3)

        # Ambiguous -> base lambda (escalated)
        lam_ambiguous = self.judge.compute_adaptive_lambda(margin=0.0, is_decisive=False)
        self.assertAlmostEqual(lam_ambiguous, 0.85, places=3)

    def test_soft_penalty_allows_belief_revision(self):
        # Choice B is banned with soft penalty (-4.5)
        # But deliberation discovers massive evidence for B: +10.0
        scores_base = [-5.0, -5.0, -5.0, -5.0]
        scores_delib = [0.0, 10.0, 0.0, 0.0]
        res = self.judge.judge_and_fuse(
            scores_base=scores_base,
            scores_delib=scores_delib,
            labels=["A", "B", "C", "D"],
            banned_labels=["B"]
        )
        # Even with penalty, B should win due to overwhelming evidence!
        self.assertEqual(res["pred_label"], "B")
        self.assertTrue(res["is_belief_revision"])

if __name__ == "__main__":
    unittest.main()
