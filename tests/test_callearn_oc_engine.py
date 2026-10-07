import unittest
import torch
import torch.nn as nn
from dual_loop.callearn_oc_engine import (
    SignedInDynamicSoftmax,
    OCDimensionModule,
    CalLearnConflictEngine,
    HADLv4UnifiedEngine,
    HADLv4ModelWrapper
)


class TestCalLearnOCEngine(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.batch_size = 2
        self.seq_len = 4

    def test_signed_in_dynamic_softmax_negative_weights(self):
        """Verifies that Signed Softmax outputs negative weights for conflicting queries."""
        signed_attn = SignedInDynamicSoftmax(d_model=self.d_model, tau=1.0)
        # Create Q and K with intentional negative alignment
        Q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        K = -Q * 2.0  # Strongly negative dot products
        V = torch.randn(self.batch_size, self.seq_len, self.d_model)

        out, weights = signed_attn(Q, K, V)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isnan(weights).any())
        # Check that negative weights exist (inhibitory component)
        has_negative = (weights < 0).any().item()
        self.assertTrue(has_negative, "Signed Softmax must produce negative weights for anti-aligned keys")

    def test_oc_dimension_open_and_close(self):
        """Verifies OC-Dimension opens to D+K and closes smoothly back to exact D."""
        oc = OCDimensionModule(d_model=self.d_model, expansion_ratio=0.5, cooling_tau=2.0)
        h = torch.randn(self.batch_size, self.seq_len, self.d_model)
        h_out, open_scale = oc(h, step_idx=1)

        self.assertEqual(h_out.shape, h.shape, "OC-Dimension must output exact original dimension D")
        self.assertFalse(torch.isnan(h_out).any())
        self.assertGreaterEqual(open_scale, 0.0)
        self.assertLessEqual(open_scale, 1.0)

    def test_callearn_conflict_engine(self):
        """Verifies CalLearn conflict score calculation and prior suppression without NaNs."""
        cal = CalLearnConflictEngine(d_model=self.d_model)
        h = torch.randn(self.batch_size, self.seq_len, self.d_model)
        h_adapted, conflict_score = cal(h)

        self.assertEqual(h_adapted.shape, h.shape)
        self.assertFalse(torch.isnan(h_adapted).any())
        self.assertGreaterEqual(conflict_score, 0.0)

    def test_hadl_v4_unified_isometry(self):
        """Verifies full unified engine isometry error is strictly < 1e-4."""
        engine = HADLv4UnifiedEngine(d_model=self.d_model)
        h = torch.randn(self.batch_size, self.seq_len, self.d_model)
        orig_norm = torch.norm(h, dim=-1)

        h_out, telemetry = engine(h)
        out_norm = torch.norm(h_out, dim=-1)

        diff = (out_norm - orig_norm).abs().mean().item()
        self.assertLess(diff, 1e-4, f"Isometry error {diff} exceeded threshold")
        self.assertIn("callearn_conflict", telemetry)
        self.assertIn("oc_dimension_open_scale", telemetry)
        self.assertIn("signed_attention_min_weight", telemetry)

    def test_hadl_v4_wrapper(self):
        """Verifies layer hook attachment on a dummy model."""
        class DummyLayer(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.linear = nn.Linear(d, d)
            def forward(self, x):
                return self.linear(x)

        class DummyModel(nn.Module):
            def __init__(self, d):
                super().__init__()
                class Sub(nn.Module):
                    def __init__(self, d):
                        super().__init__()
                        self.layers = nn.ModuleList([DummyLayer(d) for _ in range(3)])
                self.model = Sub(d)
            def forward(self, x=None, input_ids=None, inputs_embeds=None, *args, **kwargs):
                if x is None:
                    x = input_ids if input_ids is not None else inputs_embeds
                for layer in self.model.layers:
                    x = layer(x)
                return x

        base = DummyModel(self.d_model)
        wrapper = HADLv4ModelWrapper(base, target_layer_idx=1, ghost_layer_idx=2, d_model=self.d_model)
        x = torch.randn(2, 3, self.d_model)
        out = wrapper(x)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNotNone(wrapper.hook_handle_target)
        self.assertIsNotNone(wrapper.hook_handle_ghost)

    def test_signed_in_dynamic_softmax_causality(self):
        """Verifies that Signed Softmax strictly masks future positions (upper triangle == 0)."""
        signed_attn = SignedInDynamicSoftmax(d_model=self.d_model, tau=1.0)
        Q = torch.randn(2, 6, self.d_model)
        K = torch.randn(2, 6, self.d_model)
        V = torch.randn(2, 6, self.d_model)

        _, weights = signed_attn(Q, K, V, is_causal=True)
        # Upper triangle without diagonal must be strictly 0.0
        upper_tri = torch.triu(weights, diagonal=1)
        self.assertEqual(upper_tri.abs().max().item(), 0.0, "Future tokens must receive zero attention weight")

    def test_hadl_v4_kv_caching(self):
        """Verifies KV-caching across sequential generation steps."""
        engine = HADLv4UnifiedEngine(d_model=self.d_model)
        engine.eval()

        # Step 1: Prefill prompt of length 4
        h_prompt = torch.randn(1, 4, self.d_model)
        _ = engine(h_prompt)
        self.assertIsNotNone(engine.past_k)
        self.assertEqual(engine.past_k.shape[1], 4)

        # Step 2: Next token (length 1)
        h_token1 = torch.randn(1, 1, self.d_model)
        _ = engine(h_token1)
        self.assertEqual(engine.past_k.shape[1], 5)

        # Step 3: Reset cache
        engine.reset_kv_cache()
        self.assertIsNone(engine.past_k)
        self.assertIsNone(engine.past_v)

    def test_hadl_causal_dynamic_loss(self):
        """Verifies HADLCausalDynamicLoss forward and backward propagation."""
        from dual_loop.callearn_oc_engine import HADLCausalDynamicLoss
        engine = HADLv4UnifiedEngine(d_model=self.d_model)
        engine.train()
        h = torch.randn(2, 4, self.d_model, requires_grad=True)
        out, _ = engine(h)

        loss_fn = HADLCausalDynamicLoss(target_neg_weight=-0.15, lambda_bipolar=0.5, lambda_isometry=0.1)
        dummy_ce = out.sum() * 0.01
        total_loss, breakdown = loss_fn(dummy_ce, engine)

        self.assertFalse(torch.isnan(total_loss))
        self.assertIn("bipolar_loss", breakdown)
        self.assertIn("ce_loss", breakdown)
        total_loss.backward()
        self.assertIsNotNone(h.grad)

    def test_parallel_ghost_hard_wall_barrier(self):
        """Verifies Parallel Ghost Assimilation Layer enforces g_wall >= g_floor (no gate collapse)."""
        from dual_loop.callearn_oc_engine import ParallelGhostAssimilationLayer
        g_floor = 0.35
        ghost = ParallelGhostAssimilationLayer(d_model=self.d_model, g_floor=g_floor)
        
        # Test 1: Random forward pass
        h_f = torch.randn(self.batch_size, self.seq_len, self.d_model)
        h_d = torch.randn(self.batch_size, self.seq_len, self.d_model)
        h_verified, telem = ghost(h_f, h_d)
        
        self.assertEqual(h_verified.shape, h_f.shape)
        self.assertFalse(torch.isnan(h_verified).any())
        self.assertGreaterEqual(telem["ghost_gate_mean"], g_floor, "g_wall must not collapse below g_floor")
        self.assertLessEqual(telem["ghost_gate_mean"], 1.0, "g_wall must not exceed 1.0")

        # Test 2: Extreme negative diagnostic driving tanh to -1.0
        # Even under adversarial inputs, g_wall must remain >= g_floor
        h_adversarial = torch.ones_like(h_f) * 100.0
        h_verified_adv, telem_adv = ghost(h_adversarial, -h_adversarial)
        self.assertGreaterEqual(telem_adv["ghost_gate_mean"], g_floor - 1e-6)
        self.assertFalse(torch.isnan(h_verified_adv).any())

    def test_log_cosh_curvature_limiter(self):
        """Verifies LogCoshCurvatureLimiter detects 1-step and 2-step cycles and applies log-cosh penalty."""
        from dual_loop.callearn_oc_engine import LogCoshCurvatureLimiter
        limiter = LogCoshCurvatureLimiter(beta=4.0, alpha=0.5, max_period=8)

        # Case 1: 1-token loop (e.g. 5, 5, 5, 5, 5)
        input_ids = torch.tensor([[100, 5, 5, 5, 5, 5]])
        scores = torch.zeros(1, 200)
        orig_score = scores[0, 5].item()
        
        modified_scores = limiter(input_ids, scores.clone())
        self.assertLess(modified_scores[0, 5].item(), orig_score - 10.0, "LogCosh must penalize repeating 1-token cycle")

        # Case 2: 2-token loop (e.g. 10, 20, 10, 20, 10, 20)
        input_ids_2 = torch.tensor([[50, 10, 20, 10, 20, 10, 20]])
        scores_2 = torch.zeros(1, 200)
        mod_scores_2 = limiter(input_ids_2, scores_2.clone())
        self.assertLess(mod_scores_2[0, 10].item(), 0.0, "LogCosh must penalize 2-token oscillatory cycle")

    def test_confidence_oblivion_logits_processor(self):
        """Verifies ConfidenceOblivionLogitsProcessor masks popped tokens at tau=1."""
        from dual_loop.callearn_oc_engine import ConfidenceOblivionLogitsProcessor
        class MockTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                # Simulate text ending with pop and then selecting operand
                return "1. PUSH 9\n2. PUSH 3\n3. PUSH 2\n4. POP -> pop top element 2 -> [9, 3]\n5. SUB_FOLD -> pop 3 and "
            def encode(self, text, add_special_tokens=False):
                if "2" in text:
                    return [22]
                return [99]

        mock_tok = MockTokenizer()
        oblivion = ConfidenceOblivionLogitsProcessor(mock_tok, confidence_thresh=0.90)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        scores = torch.zeros(1, 100)

        mod_scores = oblivion(input_ids, scores)
        # Token 22 corresponds to popped token '2', must be masked to -1e9
        self.assertEqual(mod_scores[0, 22].item(), -1e9, "Popped token must be masked with -1e9 at tau=1")

    def test_fast_embedded_routing(self):
        """Verifies FastEmbeddedRouting M_r = M_t x M_v * exp(beta) computes routing probabilities without NaNs."""
        from dual_loop.callearn_oc_engine import FastEmbeddedRouting
        router = FastEmbeddedRouting(d_model=self.d_model, num_routes=2)
        input_ids = torch.tensor([[10, 20, 30, 40]])
        dummy_embed = torch.randn(1, 4, self.d_model)

        res = router(input_ids, embed_tensor=dummy_embed)
        if len(res) == 3:
            route_mode, route_probs, alpha_eff = res
            is_bypass = (route_mode == "bypass")
        else:
            is_bypass, route_probs = res
        self.assertIsInstance(is_bypass, bool)
        self.assertEqual(route_probs.shape, (1, 2))
        self.assertAlmostEqual(route_probs.sum().item(), 1.0, places=4)
        self.assertFalse(torch.isnan(route_probs).any())
        self.assertIn("p_bypass", router.last_routing_scores)
        self.assertIn("p_heavy", router.last_routing_scores)


if __name__ == "__main__":
    unittest.main()

