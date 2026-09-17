import unittest
import numpy as np
import torch
from dual_loop import CognitiveMatrixHelper

class TestCognitiveMatrixHelper(unittest.TestCase):
    def setUp(self):
        self.helper = CognitiveMatrixHelper(
            default_temperature=0.5,
            elimination_threshold=0.10,
            min_survivors=2
        )

    def test_build_evidence_matrix_pruning(self):
        # 5 candidates: 2 clear leaders, 3 distant distractors
        scores = [-2.0, -2.1, -8.0, -9.0, -10.0]
        labels = ["A", "B", "C", "D", "E"]
        
        matrix = self.helper.build_evidence_matrix(scores, labels=labels)
        
        self.assertEqual(len(matrix["survivors"]), 2)
        self.assertListEqual(matrix["survivor_labels"], ["A", "B"])
        self.assertListEqual(matrix["eliminated_labels"], ["C", "D", "E"])
        self.assertEqual(matrix["top1_idx"], 0)
        self.assertGreater(matrix["probs"][0], 0.40)
        self.assertLess(matrix["probs"][2], 0.05)

    def test_min_survivors_guarantee(self):
        # 4 candidates where 1 has overwhelming probability (e.g. 99%)
        scores = [0.0, -10.0, -10.0, -10.0]
        labels = ["A", "B", "C", "D"]
        
        matrix = self.helper.build_evidence_matrix(scores, labels=labels)
        
        # Even with one dominant choice, min_survivors=2 must be preserved
        self.assertGreaterEqual(len(matrix["survivors"]), 2)
        self.assertIn(0, matrix["survivors"])

    def test_filter_candidate_embeddings(self):
        B, n_cands, L_c, D = 1, 5, 4, 128
        embeds = torch.randn(B, n_cands, L_c, D)
        survivors = np.array([0, 1])
        
        filtered = self.helper.filter_candidate_embeddings(embeds, survivors)
        self.assertEqual(filtered.shape, (B, 2, L_c, D))
        self.assertTrue(torch.equal(filtered[:, 0], embeds[:, 0]))
        self.assertTrue(torch.equal(filtered[:, 1], embeds[:, 1]))

    def test_fuse_scores(self):
        scores_base = [-2.0, -2.5, -8.0, -9.0]
        survivors = np.array([0, 1])
        scores_delib_surv = [-1.5, -1.0] # Deliberation favors index 1 (Choice B)
        
        fused = self.helper.fuse_scores(
            scores_base=scores_base,
            scores_delib_survivors=scores_delib_surv,
            survivor_indices=survivors,
            lambda_delib=0.80,
            eliminated_fill_value=-1e9
        )
        
        self.assertEqual(len(fused), 4)
        self.assertLess(fused[2], -1e8) # Eliminated
        self.assertLess(fused[3], -1e8) # Eliminated
        # Survivor 1 (Choice B) should now have highest score
        self.assertEqual(np.argmax(fused), 1)

if __name__ == "__main__":
    unittest.main()
