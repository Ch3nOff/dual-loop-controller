import unittest
import torch
from dual_loop.memory import EpisodicMemoryBuffer
from dual_loop.plasticity import PlasticFastWeightUnit
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter

class TestEpisodicSelfCorrection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.rank = 8

    def test_episodic_memory_store_and_recall(self):
        buf = EpisodicMemoryBuffer(d_model=self.d_model, capacity=10, sim_threshold=0.8)
        
        k1 = torch.randn(self.d_model)
        t1 = torch.randn(self.d_model)
        m_fast1 = torch.randn(self.rank, self.rank)
        
        buf.store(key=k1, thought=t1, vacuity_u=0.62, margin=0.08, m_fast=m_fast1, meta={"task": "openbook"})
        self.assertEqual(len(buf), 1)
        
        # Test recall with very similar query (identical + slight noise)
        q_similar = k1 + 0.01 * torch.randn_like(k1)
        recalled = buf.recall(q_similar, top_k=1)
        self.assertEqual(len(recalled), 1)
        self.assertGreater(recalled[0]["similarity"], 0.95)
        self.assertEqual(recalled[0]["metadata"]["task"], "openbook")
        self.assertAlmostEqual(recalled[0]["vacuity_u"], 0.62, places=4)
        
        # Test recall with orthogonal query (should return empty)
        q_diff = -k1 + torch.randn_like(k1) * 0.5
        recalled_diff = buf.recall(q_diff, top_k=1)
        self.assertEqual(len(recalled_diff), 0)

    def test_plastic_continual_mode_and_critique(self):
        unit = PlasticFastWeightUnit(d_model=self.d_model, rank=self.rank, decay_rate=0.1)
        unit.set_continual_mode(True, decay=0.8)
        
        thoughts = torch.randn(2, 4, self.d_model)
        discrepancy = torch.randn(2, self.d_model)
        
        # Forward pass creates last_m_fast
        _, telem1 = unit(thoughts, discrepancy)
        initial_norm = telem1["m_fast_norm"]
        self.assertGreater(initial_norm, 0.0)
        
        # Reset state with continual_mode=True should softly decay rather than wipe to None
        unit.reset_state(force=False)
        self.assertIsNotNone(unit.last_m_fast)
        decayed_norm = float(unit.last_m_fast.norm().item())
        self.assertAlmostEqual(decayed_norm, initial_norm * 0.8, places=4)
        
        # Apply critique falsification
        critique_vec = torch.randn(2, self.d_model)
        critique_telem = unit.apply_critique_falsification(thoughts, critique_vec, anti_lr=0.3)
        self.assertIn("anti_update_norm", critique_telem)
        self.assertGreater(critique_telem["anti_update_norm"], 0.0)

    def test_latent_adapter_joint_candidates_and_critique(self):
        adapter = LatentDeliberationAdapter(d_model=self.d_model, n_heads=2, num_thought_tokens=2, max_ponder_steps=2)
        adapter.set_continual_mode(True)
        
        hidden_states = torch.randn(2, 10, self.d_model)
        cand_embeds = torch.randn(2, 4, self.d_model) # 4 candidate choices
        critique_vec = torch.randn(2, self.d_model)
        
        enhanced, telem = adapter(
            hidden_states=hidden_states,
            k_steps=2,
            candidate_embeds=cand_embeds,
            critique_vector=critique_vec
        )
        self.assertEqual(enhanced.shape, hidden_states.shape)
        self.assertIn("contrastive_scores", telem)
        self.assertEqual(len(telem["contrastive_scores"]), 2) # Batch size 2
        self.assertEqual(len(telem["contrastive_scores"][0]), 4) # 4 candidates

if __name__ == "__main__":
    unittest.main()
