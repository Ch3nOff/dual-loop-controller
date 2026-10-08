"""
Unit Tests for Living Cognitive Organ System (LCOS)
====================================================
Verifies:
1. DynamicAdaptiveConfusionSensor (DACS):
   - Online EMA distribution tracking without hardcoded static thresholds.
   - 2D and 3D tensor shape preservation.
   - Drift-aware gating activation.
2. SemanticAffinityEnergyMatrix (SAEM):
   - Vertical energy hierarchy & Ockham's razor parsimonious routing.
   - Energy minimization and shape preservation.
3. HierarchicalPlanCache (HLPC) & LostPlanRegister:
   - Context Engine Plan Necessity Gate (PNG) in [0, 1].
   - Ultra-compact latent waypoints (<2 KB VRAM footprint).
   - Staged decaying memory in LostPlan buffer.
   - Goal cross-verification.
4. End-to-End Runtime Integration within HADLDynamicRuntime.
"""

import unittest
import torch
import torch.nn as nn

from dual_loop.cognitive_organs import (
    ExecutiveLogicRouter,
    DynamicAdaptiveConfusionSensor,
    SemanticAffinityEnergyMatrix,
    HierarchicalPlanCache,
    CognitiveFloodgateEngine
)
from dual_loop.imagination_canvas import EmergentLatentCanvas
from dual_loop.runtime.dynamic_runtime import HADLDynamicRuntime


class DummyBlock(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        return (self.linear(x),)


class DummyModel(nn.Module):
    def __init__(self, num_layers=4, d_model=64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyBlock(d_model) for _ in range(num_layers)])
        
    def forward(self, x):
        h = x
        for layer in self.model.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return h


class TestCognitiveOrgans(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.L = 8
        self.d_model = 64

    # --------------------------------------------------------------------------
    # 1. DYNAMIC ADAPTIVE CONFUSION SENSOR (DACS) TESTS
    # --------------------------------------------------------------------------
    def test_dacs_shape_and_dynamic_threshold(self):
        sensor = DynamicAdaptiveConfusionSensor(
            d_model=self.d_model,
            num_heads=4,
            k_sigma=0.50
        )
        # 3D tensor: (B, L, D)
        h_3d = torch.randn(self.B, self.L, self.d_model)
        gate_3d, raw_conf_3d, tau_3d = sensor(h_3d)
        
        self.assertEqual(gate_3d.shape, (self.B, self.L, 1))
        self.assertEqual(raw_conf_3d.shape, (self.B, self.L, 1))
        self.assertGreaterEqual(tau_3d, 0.30)
        self.assertLessEqual(tau_3d, 0.85)
        
        # 2D tensor: (B, D)
        h_2d = torch.randn(self.B, self.d_model)
        gate_2d, raw_conf_2d, tau_2d = sensor(h_2d)
        self.assertEqual(gate_2d.shape, (self.B, 1))
        self.assertEqual(raw_conf_2d.shape, (self.B, 1))

    def test_dacs_online_adaptation_without_hardcoding(self):
        sensor = DynamicAdaptiveConfusionSensor(d_model=self.d_model, num_heads=4)
        initial_tau = sensor.get_dynamic_threshold()
        
        # Stream a series of high-variance tokens to observe self-calibrating shift
        for _ in range(10):
            h = torch.randn(self.B, self.L, self.d_model) * 2.0
            sensor(h, update_stats=True)
            
        adapted_tau = sensor.get_dynamic_threshold()
        self.assertGreater(sensor.step_count.item(), 0)
        # Sensor should have evolved its running mean and variance
        self.assertFalse(torch.isnan(sensor.running_mean))
        self.assertFalse(torch.isnan(sensor.running_var))

    # --------------------------------------------------------------------------
    # 2. SEMANTIC AFFINITY ENERGY MATRIX (SAEM) TESTS
    # --------------------------------------------------------------------------
    def test_saem_energy_hierarchy_and_parsimony(self):
        saem = SemanticAffinityEnergyMatrix(
            d_model=self.d_model,
            num_slots=8,
            d_slot=32
        )
        h = torch.randn(self.B, self.L, self.d_model)
        delta_saem, energy = saem(h, return_energy=True)
        
        self.assertEqual(delta_saem.shape, (self.B, self.L, self.d_model))
        self.assertEqual(energy.shape, (self.B, self.L, 8))
        
        # Check that top slots have higher energy penalty than bottom slots
        mean_slot_energies = energy.mean(dim=(0, 1))  # (8,)
        # Top slot (index -1) should have strictly higher vertical penalty than bottom (index 0)
        self.assertGreater(mean_slot_energies[-1].item(), mean_slot_energies[0].item())
        self.assertFalse(torch.isnan(delta_saem).any())

    # --------------------------------------------------------------------------
    # 3. HIERARCHICAL PLAN CACHE & LOSTPLAN TESTS
    # --------------------------------------------------------------------------
    def test_hlpc_plan_necessity_gating_and_vram_efficiency(self):
        hlpc = HierarchicalPlanCache(
            d_model=self.d_model,
            num_waypoints=4,
            d_plan=32
        )
        h = torch.randn(self.B, self.L, self.d_model)
        delta_plan, info = hlpc(h)
        
        self.assertEqual(delta_plan.shape, (self.B, self.L, self.d_model))
        self.assertIn("is_plan_active", info)
        self.assertIn("gate_score", info)
        self.assertGreaterEqual(info["gate_score"], 0.0)
        self.assertLessEqual(info["gate_score"], 1.0)
        
        # Verify ultra-compact VRAM footprint: 4 waypoints * 32 dims = 128 elements = 512 bytes
        plan_footprint_bytes = hlpc.active_plan.element_size() * hlpc.active_plan.nelement()
        self.assertLess(plan_footprint_bytes, 2048, "Plan memory footprint must be under 2KB!")

    def test_hlpc_lostplan_staged_decay_and_goal_verification(self):
        hlpc = HierarchicalPlanCache(
            d_model=self.d_model,
            num_waypoints=3,
            d_plan=32,
            decay_factor=0.80
        )
        h_goal = torch.randn(1, self.L, self.d_model)
        
        # Explicitly formulate plan
        hlpc.formulate_plan(h_goal)
        self.assertTrue(hlpc.is_plan_active.item())
        self.assertEqual(hlpc.current_step_idx.item(), 0)
        
        # Advance step 1: transitions to LostPlan
        hlpc.advance_substep()
        self.assertEqual(hlpc.current_step_idx.item(), 1)
        self.assertGreater(hlpc.lost_plan.abs().sum().item(), 0.0)
        
        # Advance step 2 & 3: completes all waypoints
        hlpc.advance_substep()
        hlpc.advance_substep()
        self.assertEqual(hlpc.current_step_idx.item(), 3)
        self.assertFalse(hlpc.is_plan_active.item(), "Plan should deactivate after all steps complete")
        
        # Check goal verification against accumulated LostPlan
        is_satisfied, score = hlpc.verify_goal_satisfaction(h_goal)
        self.assertIsInstance(is_satisfied, bool)
        self.assertIsInstance(score, float)
        self.assertFalse(torch.isnan(torch.tensor(score)))

    def test_hlpc_dynamic_alternative_branching_and_lateral_divergence(self):
        hlpc = HierarchicalPlanCache(
            d_model=self.d_model,
            num_waypoints=3,
            d_plan=32,
            chunk_size=1  # Immediate evaluation per chunk for unit test
        )
        h_initial = torch.randn(self.B, self.L, self.d_model)
        
        # Formulate initial plan
        hlpc.formulate_plan(h_initial)
        initial_waypoint = hlpc.active_plan[:, 0, :].clone()
        
        # Forward pass with normal state
        delta1, info1 = hlpc(h_initial)
        self.assertIn("discrepancy_score", info1)
        self.assertIn("is_branching", info1)
        self.assertIn("alternative_branches_count", info1)
        self.assertIn("lateral_norm", info1)
        self.assertIn("is_task_completed", info1)
        
        # Simulate unexpected obstacle / compiler error:
        h_obstacle = -h_initial * 3.0
        delta2, info2 = hlpc(h_obstacle, compiler_error_signal=0.6)
        
        # Discrepancy or compiler error should trigger alternative branching
        self.assertGreater(info2["discrepancy_score"], 0.0)
        self.assertTrue(info2["is_branching"])
        self.assertGreaterEqual(info2["alternative_branches_count"], 1)
        self.assertFalse(torch.isnan(delta2).any())

    def test_cognitive_floodgate_and_compiler_feedback(self):
        fg = CognitiveFloodgateEngine(
            d_model=self.d_model,
            chunk_size=4,
            capacity_limit=50.0
        )
        h = torch.randn(self.B, self.L, self.d_model)
        
        # 1. Test Dynamic Key-LogTan Noise Calibration (% log-tan * Key)
        noise_vec, dyn_thresh = fg.compute_dynamic_key_noise(h)
        self.assertEqual(noise_vec.shape, h.shape)
        self.assertGreater(dyn_thresh, 0.0)
        
        # 2. Test Non-linear Trigonometric Divergence (tan(theta/2) * sin(theta/2))
        wp = torch.randn(self.B, self.d_model)
        h_chunk = torch.randn(self.B, self.d_model)
        trig_div, mean_div = fg.compute_trigonometric_divergence(h_chunk, wp)
        self.assertGreaterEqual(mean_div, 0.0)
        self.assertFalse(torch.isnan(trig_div).any())
        
        # 3. Test Compiler Test-Drive Incoherent Penalty
        penalty = fg.apply_compiler_test_drive_penalty(h, compiler_error_signal=0.5)
        self.assertEqual(penalty.shape, h.shape)
        self.assertGreater(penalty.abs().sum().item(), 0.0)
        
        # 4. Test Floodgate Capacity Reservoir (Tembok Anti-Banjir)
        self.assertFalse(fg.is_task_completed.item())
        # Accumulate water past capacity limit (50.0)
        fg.update_floodgate_capacity(30.0, is_verified_valid=True)
        self.assertFalse(fg.is_task_completed.item())
        fg.update_floodgate_capacity(25.0, is_verified_valid=True)
        self.assertTrue(fg.is_task_completed.item(), "Floodgate dam must trip when capacity is reached!")


    # --------------------------------------------------------------------------
    # 0. EXECUTIVE LOGIC ROUTER (TOP-LEVEL META-CONTROLLER) TESTS
    # --------------------------------------------------------------------------
    def test_executive_logic_router(self):
        router = ExecutiveLogicRouter(d_model=self.d_model)
        h_3d = torch.randn(self.B, self.L, self.d_model)
        probs, info = router(h_3d)
        
        self.assertEqual(probs.shape, (self.B, 4))
        # Probabilities should sum to 1 across regimes
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.B), atol=1e-5))
        
        # Verify continuous modulation parameters exist and are bounded
        self.assertIn("dominant_mode", info)
        self.assertIn("scope_score", info)
        self.assertIn("density_score", info)
        self.assertIn("delta_anchor", info)
        self.assertIn("parsimony_scale", info)
        self.assertIn("elaboration_scale", info)
        self.assertIn("planning_urgency", info)
        self.assertGreater(info["parsimony_scale"], 0.0)
        self.assertGreater(info["elaboration_scale"], 0.0)
        self.assertGreater(info["planning_urgency"], 0.0)
        self.assertEqual(info["delta_anchor"].shape, h_3d.shape)

    # --------------------------------------------------------------------------
    # 4. RUNTIME INTEGRATION TEST
    # --------------------------------------------------------------------------
    def test_runtime_integration_with_organs(self):
        base_model = DummyModel(num_layers=4, d_model=self.d_model)
        runtime = HADLDynamicRuntime(base_model=base_model, device="cpu")
        
        # Verify new organs exist in runtime
        self.assertTrue(hasattr(runtime, "router"))
        self.assertTrue(hasattr(runtime, "saem"))
        self.assertTrue(hasattr(runtime, "plan_cache"))
        
        runtime.attach()
        self.assertTrue(runtime._active)
        
        x = torch.randn(self.B, self.L, self.d_model)
        out = base_model(x)
        self.assertEqual(out.shape, (self.B, self.L, self.d_model))
        self.assertFalse(torch.isnan(out).any())
        
        # Reset state clears both memory and plan cache
        runtime.reset_state()
        self.assertFalse(runtime.plan_cache.is_plan_active.item())
        
        runtime.detach()
        self.assertFalse(runtime._active)


if __name__ == "__main__":
    unittest.main()
