"""
Unit Tests for HADL v3.0: Universal Model-Agnostic Adapter & 5 Brain Organs
===========================================================================
"""

import unittest
import torch
import torch.nn as nn

from dual_loop import (
    UniversalDualLoopAdapter,
    UniversalDualLoopModelWrapper,
    DynamicGraphIntrospector,
    attach_universal_dual_loop,
    attach,
    CANONICAL_DIM,
    SheafInvariantFirewall,
    BoundedNormInvariant,
    DirichletVacuityInvariant,
    CodeExecutionIntegrityInvariant,
    SleepPhaseConsolidationEngine,
    UnifiedCognitiveOS,
    CognitiveConflictEvaluator,
    VitalStateVector
)


# Dummy Model Stubs representing different model families
class DummyQwenLayer(nn.Module):
    def __init__(self, d_model=1536):
        super().__init__()
        self.attn = nn.Linear(d_model, d_model)
    def forward(self, x):
        return x + self.attn(x)

class DummyQwenModel(nn.Module):
    def __init__(self, num_layers=8, d_model=1536):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyQwenLayer(d_model) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model, 1000, bias=False)
    def forward(self, x):
        h = x
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(h)


class DummyGPTModel(nn.Module):
    """Uses transformer.h container with D=2048."""
    def __init__(self, num_layers=6, d_model=2048):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model, 500, bias=False)
    def forward(self, x):
        h = x
        for block in self.transformer.h:
            h = block(h)
        return self.lm_head(h)


class TestUniversalV3(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_dynamic_graph_introspector(self):
        # 1. Qwen-style (model.layers, D=1536)
        qwen = DummyQwenModel(num_layers=10, d_model=1536)
        path, layer_list, total_layers = DynamicGraphIntrospector.discover_layer_container(qwen)
        self.assertEqual(path, "model.layers")
        self.assertEqual(total_layers, 10)
        d_native = DynamicGraphIntrospector.probe_native_dimension(qwen)
        self.assertEqual(d_native, 1536)

        # 2. GPT-style (transformer.h, D=2048)
        gpt = DummyGPTModel(num_layers=6, d_model=2048)
        path2, layer_list2, total_layers2 = DynamicGraphIntrospector.discover_layer_container(gpt)
        self.assertEqual(path2, "transformer.h")
        self.assertEqual(total_layers2, 6)
        d_native2 = DynamicGraphIntrospector.probe_native_dimension(gpt)
        self.assertEqual(d_native2, 2048)

    def test_universal_adapter_canonical_projection_rezero(self):
        # Base model with arbitrary dimension D=3584 (like Qwen-2.5-7B or Gemma)
        d_native = 3584
        adapter = UniversalDualLoopAdapter(d_native=d_native, d_canonical=1024, max_ponder_steps=2)
        
        # Test 3D input: [B=2, S=8, D=3584]
        h_in = torch.randn(2, 8, d_native)
        enhanced_h, telem = adapter(h_in)
        
        self.assertEqual(enhanced_h.shape, h_in.shape)
        self.assertEqual(telem["d_native"], 3584)
        self.assertEqual(telem["d_canonical"], 1024)
        
        # ReZero preservation theorem: because up_proj and alpha are initialized to 0,
        # enhanced_h MUST identically match h_in at initialization
        diff = torch.norm(enhanced_h - h_in).item()
        self.assertAlmostEqual(diff, 0.0, places=5)
        self.assertAlmostEqual(telem["delta_norm"], 0.0, places=5)

    def test_vllm_flat_2d_tensor_compatibility(self):
        # vLLM passes 2D flat hidden states: [NumTokens=16, D=2048]
        d_native = 2048
        adapter = UniversalDualLoopAdapter(d_native=d_native, d_canonical=1024, max_ponder_steps=2)
        
        flat_h = torch.randn(16, d_native)
        out_h, telem = adapter(flat_h)
        self.assertEqual(out_h.shape, (16, d_native))
        self.assertFalse(telem["bypassed"])

        # Test k_steps=0 streaming fast bypass
        bypass_h, bypass_telem = adapter(flat_h, k_steps=0)
        self.assertEqual(bypass_h.shape, flat_h.shape)
        self.assertTrue(bypass_telem["bypassed"])
        self.assertLess(bypass_telem["latency_us"], 1000.0)  # Sub-millisecond bypass

    def test_sheaf_invariant_firewall(self):
        firewall = SheafInvariantFirewall(max_norm=10.0, tau_obstruction=0.50)
        
        # 1. Compliant candidate delta
        small_delta = torch.randn(1, 1, 1024) * 0.10
        _ = firewall(small_delta)  # Warm up cold-start allocations
        safe_delta, telem = firewall(small_delta)
        self.assertFalse(telem["firewall_tripped"])
        self.assertEqual(len(telem["violations"]), 0)
        self.assertLess(telem["inhibition_latency_ms"], 1.0)  # Sub-millisecond

        # 2. Explosive delta (||delta|| >> 10.0)
        huge_delta = torch.randn(1, 1, 1024) * 100.0
        safe_delta2, telem2 = firewall(huge_delta)
        self.assertTrue(telem2["firewall_tripped"])
        self.assertIn("bounded_norm", telem2["violations"])
        # Should be clamped to safe norm
        self.assertLessEqual(torch.norm(safe_delta2).item(), 10.5)

        # 3. Critical Dirichlet Vacuity (u = 0.95 > 0.80)
        vacuous_delta = torch.randn(1, 1, 1024)
        safe_delta3, telem3 = firewall(vacuous_delta, context={"vacuity": 0.95})
        self.assertTrue(telem3["firewall_tripped"])
        self.assertIn("dirichlet_vacuity", telem3["violations"])

        # 4. Prohibited code instruction
        code_delta = torch.randn(1, 1, 1024)
        safe_delta4, telem4 = firewall(code_delta, context={"candidate_text": "rm -rf /tests/test_core.py"})
        self.assertTrue(telem4["firewall_tripped"])
        self.assertIn("code_integrity", telem4["violations"])
        self.assertAlmostEqual(torch.norm(safe_delta4).item(), 0.0, places=5)

    def test_sleep_phase_consolidation_engine(self):
        engine = SleepPhaseConsolidationEngine(d_model=1024, rank=16)
        
        # Simulate waking episodes
        for i in range(5):
            v = torch.randn(1, 1024)
            u = torch.randn(1, 1024)
            engine.record_episode(v, u, surprise_score=0.80 + i * 0.05)
            
        self.assertEqual(len(engine.episodes), 5)
        
        # Consolidate
        telem = engine.consolidate()
        self.assertEqual(telem["status"], "CONSOLIDATED_SUCCESS")
        self.assertEqual(telem["episodes_processed"], 5)
        self.assertGreater(telem["singular_energy_retained"], 0.90)
        self.assertEqual(telem["nullspace_leakage_overlap"], 0.0)
        self.assertEqual(len(engine.episodes), 0)  # Flushed after consolidation
        self.assertEqual(engine.total_sleep_cycles, 1)

    def test_unified_cognitive_os_step(self):
        os_engine = UnifiedCognitiveOS(d_native=2048, d_canonical=1024, tau_conflict_margin=1.0)
        h_native = torch.randn(1, 4, 2048)
        
        # 1. High-confidence logits -> Fast System 1 bypass
        confident_logits = torch.tensor([[10.0, 2.0, 1.0, 0.5]])
        out_fast, telem_fast = os_engine.execute_cognitive_step(h_native, current_logits=confident_logits)
        self.assertEqual(telem_fast["mode"], "FAST_SYSTEM_1_BYPASS")
        self.assertEqual(telem_fast["k_steps"], 0)

        # 2. Conflicting logits (dilemma: top-1 and top-2 are nearly identical) -> Slow System 2
        dilemma_logits = torch.tensor([[5.01, 5.00, 1.0, 0.5]])
        out_slow, telem_slow = os_engine.execute_cognitive_step(h_native, current_logits=dilemma_logits)
        self.assertEqual(telem_slow["mode"], "SLOW_SYSTEM_2")
        self.assertGreater(telem_slow["k_steps"], 0)
        self.assertIn("energy", telem_slow["vital_state"])

    def test_universal_attach_model(self):
        # Attach to Qwen-like model
        qwen = DummyQwenModel(num_layers=6, d_model=1536)
        wrapped_qwen = attach(qwen)
        
        self.assertIsInstance(wrapped_qwen, UniversalDualLoopModelWrapper)
        self.assertEqual(wrapped_qwen.target_layer_idx, 3)  # Midpoint of 6 layers
        self.assertEqual(wrapped_qwen.adapter.d_native, 1536)
        self.assertEqual(wrapped_qwen.adapter.d_canonical, CANONICAL_DIM)

        # Forward pass on wrapped model
        dummy_input = torch.randn(1, 1536)
        out = wrapped_qwen(dummy_input)
        self.assertEqual(out.shape, (1, 1000))

        # Sleep cycle trigger through model wrapper
        telem = wrapped_qwen.trigger_sleep_cycle()
        self.assertIn("status", telem)

        # Hook removal clean-up
        wrapped_qwen.remove_hook()


if __name__ == "__main__":
    unittest.main()
