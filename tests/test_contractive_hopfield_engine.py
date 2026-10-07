import unittest
import torch
import torch.nn as nn
from dual_loop.contractive_hopfield_engine import (
    HouseholderInvolutiveProjector,
    ModernHopfieldDiscreteMemory,
    ContractiveFixedPointDeliberator,
    AdvancedCognitiveEngine,
    AdvancedContractiveModelWrapper
)


class TestContractiveHopfieldEngine(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.batch_size = 2
        self.seq_len = 4

    def test_householder_involutive_property(self):
        """Verifies H^2 == I (Householder reflection matrix involution)."""
        projector = HouseholderInvolutiveProjector(d_model=self.d_model, num_generators=4)
        for gen_idx in range(4):
            H = projector.get_householder_matrix(gen_idx)
            H_squared = torch.matmul(H, H)
            identity = torch.eye(self.d_model, dtype=H.dtype)
            error = torch.norm(H_squared - identity).item()
            self.assertLess(error, 1e-4, f"Householder matrix H does not satisfy H^2 = I for generator {gen_idx}")

    def test_modern_hopfield_discrete_attractor(self):
        """Verifies Hopfield pattern retrieval and finite free energy."""
        hopfield = ModernHopfieldDiscreteMemory(d_model=self.d_model, num_prototypes=16, beta=8.0)
        h = torch.randn(self.batch_size, self.seq_len, self.d_model)
        h_snapped, energy = hopfield(h)

        self.assertEqual(h_snapped.shape, h.shape)
        self.assertEqual(energy.shape, (self.batch_size, self.seq_len))
        self.assertFalse(torch.isnan(h_snapped).any())
        self.assertFalse(torch.isnan(energy).any())
        self.assertFalse(torch.isinf(energy).any())

    def test_contractive_fixed_point_convergence(self):
        """Verifies Banach contractive iteration converges and reduces Cauchy residual."""
        deliberator = ContractiveFixedPointDeliberator(
            d_model=self.d_model,
            gamma_lipschitz=0.80,
            max_iterations=5,
            tolerance=1e-2
        )
        h_init = torch.randn(self.batch_size, self.seq_len, self.d_model)
        h_star, iters, residual = deliberator(h_init)

        self.assertEqual(h_star.shape, h_init.shape)
        self.assertGreater(iters, 0)
        self.assertLess(residual, 0.5, "Contractive iteration failed to diminish residual")

    def test_advanced_engine_isometry_preservation(self):
        """Verifies length-preserving isometry (||h'|| == ||h|| with < 1e-4 drift)."""
        engine = AdvancedCognitiveEngine(d_model=self.d_model, num_generators=4, num_prototypes=16)
        h = torch.randn(self.batch_size, self.seq_len, self.d_model)
        orig_norm = torch.norm(h, dim=-1)

        h_out, telemetry = engine(h)
        out_norm = torch.norm(h_out, dim=-1)

        diff = (out_norm - orig_norm).abs().mean().item()
        self.assertLess(diff, 1e-4, f"Isometry violated: norm drift {diff} exceeds 1e-4")
        self.assertIn("isometry_error", telemetry)
        self.assertIn("fixed_point_iters", telemetry)
        self.assertIn("hopfield_mean_energy", telemetry)

    def test_dummy_model_wrapper(self):
        """Verifies layer hook integration on a dummy transformer block."""
        class DummyLayer(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.linear = nn.Linear(d, d)
            def forward(self, x):
                return self.linear(x)

        class DummyModel(nn.Module):
            def __init__(self, d):
                super().__init__()
                class SubModel(nn.Module):
                    def __init__(self, d):
                        super().__init__()
                        self.layers = nn.ModuleList([DummyLayer(d) for _ in range(3)])
                self.model = SubModel(d)
            def forward(self, x):
                for layer in self.model.layers:
                    x = layer(x)
                return x

        dummy_base = DummyModel(self.d_model)
        wrapper = AdvancedContractiveModelWrapper(dummy_base, target_layer_idx=1)
        x = torch.randn(2, 3, self.d_model)
        out = wrapper(x)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNotNone(wrapper.hook_handle)


if __name__ == "__main__":
    unittest.main()
