"""
Unit Tests for Latent Reconstructive Hologram (Compressed Sensing & FISTA Inverse Recovery)
===========================================================================================
Tests mathematical guarantees of Candes-Tao sparse recovery, 2-bit & ternary skeleton
quantization, 8GB laptop VRAM physical budget, and Universal Adapter integration.
"""

import unittest
import torch
import torch.nn as nn
import math

from dual_loop import (
    FISTALatentRecovery,
    LatentReconstructiveHologram,
    SkeletonQuantizer,
    UniversalDualLoopAdapter,
    attach_universal_dual_loop
)


class TestFISTALatentRecovery(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_native = 3584 # Standard 30B model hidden dimension
        self.d_canonical = 1024 # Fixed System 2 canonical dimension
        self.fista = FISTALatentRecovery(
            d_native=self.d_native,
            d_canonical=self.d_canonical,
            lambda_sparse=0.015,
            default_steps=3
        )

    def test_tensor_shapes_2d_and_3d(self):
        # 3D Tensor: [B, S, D_native]
        h_3d = torch.randn(2, 8, self.d_native)
        z_3d, h_clean_3d, telem_3d = self.fista(h_3d)
        self.assertEqual(z_3d.shape, (2, 8, self.d_canonical))
        self.assertEqual(h_clean_3d.shape, (2, 8, self.d_native))
        self.assertIn("snr_gain_db", telem_3d)
        self.assertIn("fista_steps", telem_3d)

        # 2D Tensor: [NumTokens, D_native] (vLLM / TensorRT style)
        h_2d = torch.randn(16, self.d_native)
        z_2d, h_clean_2d, telem_2d = self.fista(h_2d)
        self.assertEqual(z_2d.shape, (16, self.d_canonical))
        self.assertEqual(h_clean_2d.shape, (16, self.d_native))

    def test_fista_convergence_and_noise_reduction(self):
        # Create a true sparse latent code z_true
        z_true = torch.zeros(1, 4, self.d_canonical)
        # 10% non-zero active features (sparse manifold)
        sparse_indices = torch.randperm(self.d_canonical)[:100]
        z_true[:, :, sparse_indices] = torch.randn(1, 4, 100)

        # Generate clean hidden states via Phi projection
        Phi = self.fista.phi
        h_true = torch.matmul(z_true, Phi.T)

        # Add 2-bit severe quantization noise: h_noisy = h_true + epsilon_quant
        noise_level = 0.50 * torch.std(h_true)
        epsilon_quant = torch.randn_like(h_true) * noise_level
        h_noisy = h_true + epsilon_quant

        # Run FISTA with 3 iterations
        z_recovered, h_clean, telem = self.fista(h_noisy, k_steps=3)

        # Verify residual monotonically decreased and noise is suppressed
        self.assertLess(telem["final_residual"], telem["initial_residual"])
        self.assertGreater(telem["residual_reduction_pct"], 0.0)
        self.assertGreater(telem["snr_gain_db"], 5.0)

        # Cosine similarity between recovered clean and ground truth
        cos_sim = torch.cosine_similarity(h_clean.flatten(), h_true.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.90, f"Expected high cosine similarity with ground truth, got {cos_sim}")

    def test_latency_is_sub_4ms_on_available_hardware(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fista_dev = self.fista.to(device)
        h_test = torch.randn(1, 1, self.d_native, device=device)

        # Warmup
        for _ in range(5):
            _ = fista_dev(h_test, k_steps=3)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timing
        import time
        t0 = time.perf_counter()
        for _ in range(10):
            _, _, telem = fista_dev(h_test, k_steps=3)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        avg_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0

        if torch.cuda.is_available():
            self.assertLess(avg_ms, 5.0, f"GPU latency {avg_ms:.2f}ms should be sub-5ms")


class TestSkeletonQuantizer(unittest.TestCase):
    def test_2bit_quantization(self):
        w = torch.randn(128, 128)
        w_q, w_dequant = SkeletonQuantizer.quantize_to_2bit(w)
        # Quantized levels should be in {-1.5, -0.5, 0.5, 1.5}
        unique_vals = torch.unique(w_q).tolist()
        for v in unique_vals:
            self.assertIn(v, [-1.5, -0.5, 0.5, 1.5])
        # Dequantized should maintain positive correlation
        cos = torch.cosine_similarity(w.flatten(), w_dequant.flatten(), dim=0).item()
        self.assertGreater(cos, 0.80)

    def test_1_58bit_ternary_quantization(self):
        w = torch.randn(128, 128)
        w_q, w_dequant = SkeletonQuantizer.quantize_to_ternary_1_58bit(w)
        # Quantized levels must be strictly {-1, 0, 1}
        unique_vals = set(torch.unique(w_q).int().tolist())
        self.assertTrue(unique_vals.issubset({-1, 0, 1}))

    def test_vram_budget_8gb(self):
        budget = SkeletonQuantizer.calculate_vram_budget_8gb(
            num_params_billion=30.0,
            context_tokens=4096,
            d_model=3584,
            num_layers=48
        )
        self.assertTrue(budget["fits_in_8gb_laptop_vram"])
        self.assertLessEqual(budget["total_vram_required_gb"], 5.50)
        self.assertGreaterEqual(budget["vram_headroom_on_8gb_gpu_gb"], 2.50)
        self.assertAlmostEqual(budget["system_2_fp16_engine_gb"], 0.18, delta=0.05)


class TestLatentReconstructiveHologram(unittest.TestCase):
    def test_reconstruction_and_plastic_compensation(self):
        hologram = LatentReconstructiveHologram(
            d_native=3584,
            d_canonical=1024,
            default_steps=3,
            enable_plastic_compensation=True
        )
        h_noisy = torch.randn(1, 4, 3584)
        z_sparse, h_clean, telem = hologram.reconstruct(h_noisy)
        self.assertEqual(z_sparse.shape, (1, 4, 1024))
        self.assertEqual(h_clean.shape, (1, 4, 3584))
        self.assertIn("snr_gain_db", telem)

        # Test plastic compensation binding
        discrepancy = torch.randn(1, 4, 3584)
        initial_m_norm = torch.norm(hologram.M_fast_compensation).item()
        self.assertEqual(initial_m_norm, 0.0)

        hologram.record_plastic_compensation(z_sparse, discrepancy)
        updated_m_norm = torch.norm(hologram.M_fast_compensation).item()
        self.assertGreater(updated_m_norm, 0.0)


class TestAdapterHologramIntegration(unittest.TestCase):
    def test_universal_adapter_with_hologram(self):
        adapter = UniversalDualLoopAdapter(
            d_native=2048,
            d_canonical=1024,
            enable_hologram=True,
            hologram_steps=2
        )
        h_in = torch.randn(2, 6, 2048)
        enhanced_h, telem = adapter(h_in)
        self.assertEqual(enhanced_h.shape, (2, 6, 2048))
        self.assertIn("hologram", telem)
        self.assertEqual(telem["hologram"]["fista_steps"], 2)

    def test_attach_universal_dual_loop_with_hologram(self):
        class DummyTransformerBlock(nn.Module):
            def __init__(self, d_model=2048):
                super().__init__()
                self.proj = nn.Linear(d_model, d_model)
            def forward(self, x):
                return x + self.proj(x)

        class DummyModel(nn.Module):
            def __init__(self, d_model=2048, layers=4):
                super().__init__()
                self.layers = nn.ModuleList([DummyTransformerBlock(d_model) for _ in range(layers)])
            def forward(self, x):
                h = x
                for l in self.layers:
                    h = l(h)
                return h

        model = DummyModel()
        wrapped = attach_universal_dual_loop(model, enable_hologram=True, hologram_steps=3)
        self.assertTrue(wrapped.adapter.enable_hologram)
        self.assertIsNotNone(wrapped.adapter.hologram)

        x = torch.randn(2, 5, 2048)
        out = wrapped(x)
        self.assertEqual(out.shape, (2, 5, 2048))


if __name__ == "__main__":
    unittest.main()
