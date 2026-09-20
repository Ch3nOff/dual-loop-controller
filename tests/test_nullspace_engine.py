import unittest
import torch
from dual_loop.nullspace_engine import OrthogonalNullspaceProjector

class TestNullspaceEngine(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.num_slots = 8
        self.projector = OrthogonalNullspaceProjector(d_model=self.d_model)

    def test_orthogonality_guarantee(self):
        """Nullspace projection must be strictly orthogonal to all basis memory vectors."""
        B = 2
        basis_memory = torch.randn(B, self.num_slots, self.d_model)
        candidate = torch.randn(B, self.d_model)
        
        h_novel, telem = self.projector(candidate, basis_memory)
        
        # Verify telemetry
        self.assertTrue(telem["is_strictly_orthogonal"])
        self.assertLess(telem["max_basis_leakage"], 1e-4)
        
        # Verify explicit dot products: V Q^T x_perp == 0
        V = basis_memory.transpose(1, 2) # [B, D, M]
        dots = torch.bmm(V.transpose(1, 2), h_novel.unsqueeze(-1)).squeeze(-1) # [B, M]
        max_leakage = float(torch.max(torch.abs(dots)).item())
        self.assertLess(max_leakage, 1e-3)

    def test_shape_preservation(self):
        """Preserves [B, D] or [B, 1, D] shapes."""
        B = 3
        basis = torch.randn(B, 4, self.d_model)
        
        c2 = torch.randn(B, self.d_model)
        out2, _ = self.projector(c2, basis)
        self.assertEqual(out2.shape, (B, self.d_model))
        
        c3 = torch.randn(B, 1, self.d_model)
        out3, _ = self.projector(c3, basis)
        self.assertEqual(out3.shape, (B, 1, self.d_model))

    def test_norm_preservation(self):
        """Purified vector retains non-zero semantic magnitude."""
        basis = torch.randn(1, 4, self.d_model)
        candidate = torch.randn(1, self.d_model) * 5.0
        
        h_novel, _ = self.projector(candidate, basis)
        self.assertGreater(torch.norm(h_novel).item(), 0.1)

if __name__ == "__main__":
    unittest.main()
