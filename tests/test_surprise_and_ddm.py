import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dual_loop.halting import DriftDiffusionHalting
from dual_loop.verification import UncertaintySurpriseGate, ContrastiveEvidenceAccumulator
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter
from dual_loop.adapters.qwen_adapter import DualLoopQwenModel


class TestDriftDiffusionHalting(unittest.TestCase):
    def setUp(self):
        self.ddm = DriftDiffusionHalting(theta_0=3.0, k_max=4, gamma=0.5, min_theta=0.5)

    def test_evidence_computation(self):
        logits = torch.tensor([
            [1.0, 10.0, 0.5, 2.0, -1.0],
            [4.0, 1.0, 4.1, 2.0, 0.0]
        ])
        evidence = self.ddm.compute_evidence(logits)
        self.assertAlmostEqual(evidence[0].item(), 8.0, places=4)
        self.assertAlmostEqual(evidence[1].item(), 0.1, places=4)

    def test_collapsing_boundary(self):
        b1 = self.ddm.decision_boundary(1).item()
        b2 = self.ddm.decision_boundary(2).item()
        b3 = self.ddm.decision_boundary(3).item()
        b4 = self.ddm.decision_boundary(4).item()

        self.assertGreater(b1, b2)
        self.assertGreater(b2, b3)
        self.assertGreaterEqual(b3, b4)
        self.assertGreaterEqual(b4, 0.5)

    def test_halting_decision(self):
        logits = torch.tensor([
            [1.0, 10.0, 0.5, 2.0, -1.0],
            [4.0, 1.0, 4.1, 2.0, 0.0]
        ])
        halt_mask, evidence, bnd = self.ddm.should_halt(logits, k=1)
        self.assertTrue(halt_mask[0].item())
        self.assertFalse(halt_mask[1].item())


class TestUncertaintySurpriseGate(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.vocab_size = 128
        self.gate = UncertaintySurpriseGate(
            d_model=self.d_model,
            tau_surprise=0.01,
            temperature=0.05,
            vocab_size=self.vocab_size
        )

    def test_zero_delta_yields_zero_jsd_and_low_gate(self):
        h_base = torch.randn(2, self.d_model)
        h_delib = h_base.clone()
        gate_val, jsd = self.gate.compute_surprise_gate(h_base, h_delib)
        
        self.assertTrue((jsd < 1e-5).all())
        self.assertTrue((gate_val < 0.5).all())

    def test_significant_divergence_opens_gate(self):
        h_base = torch.randn(2, self.d_model)
        h_delib = h_base + torch.randn(2, self.d_model) * 10.0
        gate_val, jsd = self.gate.compute_surprise_gate(h_base, h_delib)
        
        self.assertTrue((jsd > 0.05).all())
        self.assertTrue((gate_val > 0.65).all())

    def test_external_lm_head_binding(self):
        lm_head = nn.Linear(self.d_model, 1000)
        self.gate.set_lm_head(lm_head)
        self.assertIs(self.gate.probe, lm_head)
        
        h_base = torch.randn(3, self.d_model)
        h_delib = h_base + torch.randn(3, self.d_model) * 0.1
        gate_val, jsd = self.gate.compute_surprise_gate(h_base, h_delib)
        self.assertEqual(gate_val.shape, (3, 1))
        self.assertEqual(jsd.shape, (3,))


class TestContrastiveEvidenceAccumulator(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.accumulator = ContrastiveEvidenceAccumulator(
            d_model=self.d_model,
            n_heads=4,
            tau_contrast=0.1
        )

    def test_directed_delta_and_scores_tensor(self):
        B = 2
        n_cands = 4
        L_c = 6
        thought = torch.randn(B, self.d_model)
        candidate_embeds = torch.randn(B, n_cands, L_c, self.d_model)
        
        delta, scores = self.accumulator(thought, candidate_embeds)
        self.assertEqual(delta.shape, (B, self.d_model))
        self.assertEqual(scores.shape, (B, n_cands))
        sums = scores.sum(dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_directed_delta_and_scores_list(self):
        B = 2
        thought = torch.randn(B, 1, self.d_model)
        cands_list = [
            torch.randn(B, 3, self.d_model),
            torch.randn(B, 5, self.d_model),
            torch.randn(B, 2, self.d_model)
        ]
        delta, scores = self.accumulator(thought, cands_list)
        self.assertEqual(delta.shape, (B, self.d_model))
        self.assertEqual(scores.shape, (B, 3))


class TestAdapterEndToEndIntegration(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.adapter = LatentDeliberationAdapter(
            d_model=self.d_model,
            n_heads=4,
            num_thought_tokens=2,
            max_ponder_steps=2,
            use_surprise_gate=True,
            use_contrastive_evidence=True,
            use_ddm_halting=True
        )

    def test_forward_with_candidate_embeds(self):
        B, S = 2, 10
        hidden_states = torch.randn(B, S, self.d_model)
        candidate_embeds = torch.randn(B, 4, 3, self.d_model)
        
        enhanced, telem = self.adapter(
            hidden_states=hidden_states,
            k_steps=2,
            candidate_embeds=candidate_embeds
        )
        self.assertEqual(enhanced.shape, hidden_states.shape)
        self.assertIn('surprise_gate', telem)
        self.assertIn('surprise_jsd', telem)
        self.assertIn('contrastive_scores', telem)
        self.assertEqual(len(telem['contrastive_scores']), B)
        self.assertEqual(len(telem['contrastive_scores'][0]), 4)

    def test_forward_open_ended_without_candidates(self):
        B, S = 2, 10
        hidden_states = torch.randn(B, S, self.d_model)
        
        enhanced, telem = self.adapter(
            hidden_states=hidden_states,
            k_steps=2,
            candidate_embeds=None
        )
        self.assertEqual(enhanced.shape, hidden_states.shape)
        self.assertEqual(len(telem['contrastive_scores']), 0)
        self.assertEqual(len(telem['surprise_gate']), B)


class DummyQwen(nn.Module):
    def __init__(self, d_model=64, vocab_size=128):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, 4, 128, batch_first=True)
            for _ in range(4)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        x = inputs_embeds if inputs_embeds is not None else torch.randn(2, 8, 64)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return logits


class TestDualLoopQwenModelWithNewCapabilities(unittest.TestCase):
    def test_qwen_wrapper_with_candidates(self):
        qwen = DummyQwen(d_model=64, vocab_size=128)
        model = DualLoopQwenModel(
            qwen_model=qwen,
            layer_idx=2,
            k_steps=2,
            num_thought_tokens=2,
            max_ponder_steps=2,
            use_surprise_gate=True,
            use_contrastive_evidence=True
        )
        
        self.assertIs(model.adapter.surprise_gate.probe, qwen.lm_head)

        B = 2
        x = torch.randn(B, 8, 64)
        candidate_embeds = torch.randn(B, 4, 3, 64)
        
        out = model(inputs_embeds=x, candidate_embeds=candidate_embeds)
        self.assertEqual(out.shape, (B, 8, 128))
        self.assertIn('contrastive_scores', model.last_telemetry)
        self.assertIn('surprise_gate', model.last_telemetry)
        model.remove_hook()


if __name__ == '__main__':
    unittest.main()
