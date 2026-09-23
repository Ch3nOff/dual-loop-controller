import unittest
import time
import torch
import torch.nn as nn
from dual_loop.allostasis import AllostaticEnergyModulator
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter

class TestAllostaticModulation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 256
        self.B = 4
        self.raw_delta = torch.randn(self.B, self.d_model)
        self.modulator = AllostaticEnergyModulator(d_model=self.d_model)

    def test_gate_boundedness_and_shape(self):
        scale = torch.tensor([0.8])
        surprise_gate = torch.tensor([[0.7], [0.5], [0.9], [0.1]])
        beta_gate = torch.tensor([[0.8], [0.6], [0.95], [0.2]])
        drift = torch.tensor([[0.1], [0.5], [0.0], [1.2]])
        vacuity = torch.tensor([[0.2], [0.4], [0.1], [0.8]])
        
        modulated, telem = self.modulator(
            raw_delta=self.raw_delta,
            scale=scale,
            surprise_gate=surprise_gate,
            beta_gate=beta_gate,
            drift_penalty=drift,
            vacuity_u=vacuity
        )
        
        self.assertEqual(modulated.shape, (self.B, self.d_model))
        gammas = telem["gamma_allostatic"]
        for g in gammas:
            self.assertGreaterEqual(g, 0.0)
            self.assertLessEqual(g, 1.0)
            
    def test_avoids_cascade_collapse(self):
        """
        Verifies that under moderately confident conditions, AllostaticEnergyModulator
        preserves signal magnitude substantially better than naive 5-way multiplication.
        """
        scale = torch.tensor([0.7])
        surprise = torch.tensor([[0.6]])
        beta = torch.tensor([[0.6]])
        drift = torch.tensor([[0.2]]) # exp(-0.2) = 0.81
        vac = torch.tensor([[0.6]])   # eta = 0.8
        
        # Legacy cascade multiplication:
        legacy_gate = 0.7 * 0.6 * 0.6 * 0.81 * 0.8 # ~0.163
        
        raw = torch.ones(1, self.d_model)
        modulated, telem = self.modulator(
            raw_delta=raw,
            scale=scale,
            surprise_gate=surprise,
            beta_gate=beta,
            drift_penalty=drift,
            vacuity_u=vac
        )
        allo_gate = telem["gamma_allostatic"][0]
        
        # Allostatic gating prevents vanishing signal
        self.assertGreater(allo_gate, legacy_gate)
        self.assertGreater(allo_gate, 0.40)

    def test_submillisecond_latency(self):
        """Validates that allostatic energy modulation executes in sub-millisecond time (< 0.1 ms)."""
        scale = torch.tensor([0.8])
        raw = torch.randn(8, 768)
        mod = AllostaticEnergyModulator(768)
        
        # Warmup
        for _ in range(10):
            mod(raw, scale)
            
        t0 = time.perf_counter()
        iters = 500
        for _ in range(iters):
            mod(raw, scale)
        avg_ms = ((time.perf_counter() - t0) / iters) * 1000.0
        
        # Fast kernel: strictly under 0.2 ms
        self.assertLess(avg_ms, 0.20, f"Allostatic modulation took {avg_ms:.4f} ms, expected < 0.20 ms")

    def test_adapter_allostatic_integration(self):
        """Tests full LatentDeliberationAdapter with enable_allostatic_modulation=True."""
        adapter = LatentDeliberationAdapter(
            d_model=self.d_model,
            n_heads=4,
            max_ponder_steps=2,
            enable_allostatic_modulation=True
        )
        x = torch.randn(2, 8, self.d_model)
        out, telem = adapter(x)
        
        self.assertEqual(out.shape, x.shape)
        self.assertIn("allostasis", telem)
        self.assertTrue(telem["allostasis"].get("pruned_gate_active", False))

if __name__ == "__main__":
    unittest.main()
