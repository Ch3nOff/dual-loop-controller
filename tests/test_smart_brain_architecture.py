import unittest
import torch
from dual_loop.memory import EpisodicMemoryBuffer
from dual_loop.verification import CognitiveConflictMonitor, AntiHyperSkepticismFilter
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter

class TestSmartBrainArchitecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.rank = 8

    def test_settled_logic_consolidation(self):
        buf = EpisodicMemoryBuffer(d_model=self.d_model, capacity=10)
        
        q1 = torch.randn(self.d_model)
        t1 = torch.randn(self.d_model)
        m_fast1 = torch.randn(self.rank, self.rank)
        
        # Store as settled logic anchor
        buf.store(
            key=q1,
            thought=t1,
            vacuity_u=0.60,
            margin=0.85, # Solid margin
            m_fast=m_fast1,
            is_settled=True,
            confidence=0.92,
            meta={"task": "arc_easy", "pred": "A"}
        )
        self.assertEqual(len(buf), 1)
        
        # Test recall_settled
        settled = buf.recall_settled(q1 + 0.02 * torch.randn_like(q1), sim_threshold=0.85)
        self.assertIsNotNone(settled)
        self.assertTrue(settled["is_settled"])
        self.assertAlmostEqual(settled["confidence"], 0.92, places=2)
        
        # Non-settled item
        q2 = torch.randn(self.d_model)
        buf.store(key=q2, thought=t1, vacuity_u=0.59, margin=0.05, is_settled=False)
        settled2 = buf.recall_settled(q2, sim_threshold=0.85)
        self.assertIsNone(settled2)

    def test_cognitive_conflict_monitor_effort_index(self):
        monitor = CognitiveConflictMonitor(tau_margin=0.30, tau_jsd=0.05, tau_effort=0.35)
        
        # Case A: Intuitive Physical Commonsense (High margin, near zero JSD)
        # Should yield low cognitive effort -> recommended_k = 0 (System 1 fast path)
        effort_intuitive, k_intuitive, should_delib_intuitive = monitor.compute_effort_index(
            margin=0.80,
            jsd=0.001,
            vacuity_u=0.55
        )
        self.assertLess(effort_intuitive, 0.35)
        self.assertEqual(k_intuitive, 0)
        self.assertFalse(should_delib_intuitive)
        
        # Case B: Multi-Hop Conflict (Razor-thin margin, high JSD)
        # Should yield high cognitive effort -> recommended_k = 2 (System 2 deliberation)
        effort_conflict, k_conflict, should_delib_conflict = monitor.compute_effort_index(
            margin=0.05,
            jsd=0.15,
            vacuity_u=0.60
        )
        self.assertGreater(effort_conflict, 0.50)
        self.assertEqual(k_conflict, 2)
        self.assertTrue(should_delib_conflict)

    def test_anti_hyper_skepticism_filter(self):
        filter_gate = AntiHyperSkepticismFilter(tau_protect=0.25)
        
        # Settled answer: critique MUST be blocked
        self.assertFalse(filter_gate.should_apply_critique(is_settled=True, pass1_margin=0.05))
        
        # High confidence answer: critique MUST be blocked
        self.assertFalse(filter_gate.should_apply_critique(is_settled=False, pass1_margin=0.10, pass1_confidence=0.80))
        
        # Solid margin answer: critique MUST be blocked (prevents second-guessing correct choices)
        self.assertFalse(filter_gate.should_apply_critique(is_settled=False, pass1_margin=0.50))
        
        # Razor-thin tie / unresolved conflict: critique permitted
        self.assertTrue(filter_gate.should_apply_critique(is_settled=False, pass1_margin=0.04, pass1_confidence=0.30))

    def test_latent_adapter_blocks_critique_on_settled_item(self):
        adapter = LatentDeliberationAdapter(d_model=self.d_model, n_heads=2, num_thought_tokens=2, max_ponder_steps=2)
        
        hidden_states = torch.randn(1, 6, self.d_model)
        query_rep = hidden_states[:, -1, :]
        critique_vec = torch.randn(1, self.d_model)
        
        # Store settled entry in adapter's episodic memory
        adapter.episodic_memory.store(
            key=query_rep,
            thought=torch.randn(self.d_model),
            vacuity_u=0.60,
            margin=0.65,
            is_settled=True,
            confidence=0.90
        )
        
        enhanced, telem = adapter(
            hidden_states=hidden_states,
            k_steps=2,
            critique_vector=critique_vec
        )
        self.assertEqual(enhanced.shape, hidden_states.shape)

if __name__ == "__main__":
    unittest.main()
