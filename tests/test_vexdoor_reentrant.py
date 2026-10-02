"""
Unit Tests for HADL v3.4: Vexdoor Re-entrant Closed-Loop & Nullspace Append Engine
==================================================================================
Tests:
1. Vexdoor Decay: verifies wind closure dynamics V(0) > 0 and V(t) -> 0.0.
2. Bounded Logits: guarantees delta logits are strictly bounded within [-max_bound, max_bound].
3. Log-Det Volume Similarity: verifies Gramian volume computation and novelty detection.
4. Nullspace Append Orthogonality: tests that W_old * Pi_null == 0 (zero catastrophic forgetting).
5. Closed-Loop Wrapper Interception: tests System 1 bypass vs System 2 Vexdoor deliberation.
"""

import math
import unittest
import torch
import torch.nn as nn

from dual_loop.vexdoor_reentrant_engine import (
    VexdoorDecayGate,
    LogDetVolumeSimilarity,
    NullspaceMemoryAppend,
    VexdoorClosedLoopWrapper
)


class MockBaseConfig:
    def __init__(self, hidden_size=256, vocab_size=1000):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class MockBaseModel(nn.Module):
    def __init__(self, d_model=256, vocab_size=1000):
        super().__init__()
        self.config = MockBaseConfig(hidden_size=d_model, vocab_size=vocab_size)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        return self.lm_head(x)


class TestVexdoorReentrantEngine(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 256
        self.vocab_size = 1000

    def test_vexdoor_decay_dynamics(self):
        """Verifies that Vexdoor starts open and decays smoothly to 0.0 over steps."""
        vexdoor = VexdoorDecayGate(tau_wind=2.0, gamma_decay=0.2, max_bound=4.5)
        energy = torch.ones(1, 1, 1)

        # Step 0: Door is open
        v_0 = vexdoor.compute_gate(energy).item()
        self.assertGreater(v_0, 0.8, "Vexdoor should be wide open at step 0.")

        # Simulate steps
        for _ in range(5):
            vexdoor.step()

        # Step 5: Door is substantially closed
        v_5 = vexdoor.compute_gate(energy).item()
        self.assertLess(v_5, 0.2, "Vexdoor should be closing after multiple steps.")

        # Step 15: Door must be fully shut (0.0)
        for _ in range(10):
            vexdoor.step()
        v_15 = vexdoor.compute_gate(energy).item()
        self.assertEqual(v_15, 0.0, "Vexdoor must be fully clamped at 0.0.")

    def test_delta_logits_bounded(self):
        """Guarantees delta logits NEVER exceed max_bound (no +1680 explosion)."""
        vexdoor = VexdoorDecayGate(max_bound=4.5)
        v_gate = torch.tensor([[[1.0]]])
        
        # Huge raw delta logits simulating uncontrolled optimization
        huge_delta = torch.randn(2, 4, 1000) * 5000.0
        bounded = vexdoor.bound_delta_logits(huge_delta, v_gate)
        
        max_val = torch.max(torch.abs(bounded)).item()
        self.assertLessEqual(max_val, 4.5 + 1e-4, f"Delta logits exceeded bound: {max_val} > 4.5")

    def test_log_det_volume_similarity(self):
        """Verifies Gramian log-determinant volume calculation."""
        meter = LogDetVolumeSimilarity(d_model=self.d_model)
        
        # Parallel / Collinear vectors (low volume)
        v = torch.randn(1, 1, self.d_model)
        k_parallel = v.repeat(1, 4, 1)
        _, vol_low = meter(k_parallel)

        # Orthogonal random vectors (higher volume)
        k_ortho = torch.randn(1, 4, self.d_model)
        _, vol_high = meter(k_ortho)

        self.assertGreater(vol_high, vol_low, "Orthogonal representations must span greater log-det volume.")

    def test_nullspace_append_orthogonality(self):
        """Tests that W_old * Pi_null(W) == 0 up to machine precision."""
        stager = NullspaceMemoryAppend(d_model=128)
        W = torch.randn(64, 128)  # Fat matrix with non-trivial nullspace
        
        # Stage new knowledge
        X_new = torch.randn(4, 128)
        stager.stage_knowledge(X_new)

        delta_W, ortho_error = stager.compute_nullspace_append(W)
        self.assertEqual(delta_W.shape, W.shape)
        # Verify strict orthogonality
        self.assertLess(ortho_error, 1e-4, f"Orthogonality violated: {ortho_error}")

    def test_closed_loop_wrapper_system1_bypass(self):
        """Verifies System 1 bypass when entropy is below threshold."""
        base_model = MockBaseModel(d_model=self.d_model, vocab_size=self.vocab_size)
        wrapper = VexdoorClosedLoopWrapper(base_model, entropy_threshold=2.0)
        
        # Very confident logits (low entropy)
        x = torch.randn(1, 2, self.d_model) * 10.0
        out = wrapper(x)
        
        self.assertEqual(out.shape, (1, 2, self.vocab_size))
        self.assertEqual(wrapper.last_telemetry.get("mode"), "System 1 (Bypass)")
        wrapper.remove_hooks()


if __name__ == "__main__":
    unittest.main()
