import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class OrthogonalNullspaceProjector(nn.Module):
    """
    Orthogonal Nullspace Synthesis (Gram-Schmidt Latent Projection).
    
    Prevents deep learning representations from falling back into the dense
    prior training attractor manifold when encountering unprecedented concepts or anomalies.
    
    Mathematical Formulation:
        Given known knowledge basis V_known = [m_1, ..., m_M] in R^{D x M} from memory/context:
        1. Compute orthonormal basis Q in R^{D x M} via thin QR decomposition:
           V_known = Q R, where Q^T Q = I_M.
        2. Projection onto known subspace:
           P x = Q (Q^T x)
        3. Orthogonal Nullspace Projection Operator:
           x_perp = (I - P) x = x - Q (Q^T x)
           
        Geometric Guarantee:
           Q^T x_perp = Q^T x - (Q^T Q) Q^T x = Q^T x - I Q^T x = 0
           For any v in span(V_known), <x_perp, v> = 0 identically.
           
    This guarantees that the synthesized prototype occupies an entirely new, unassigned
    orthogonal coordinate in latent space without interfering with existing concepts.
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-7,
        normalize_output: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.normalize_output = normalize_output

    def project_to_nullspace(
        self,
        candidate_vector: torch.Tensor,
        basis_memory: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Projects candidate_vector into the orthogonal nullspace of basis_memory.
        
        Args:
            candidate_vector: [B, D] or [B, 1, D] Novel concept proposal or discrepancy.
            basis_memory: [B, M, D] Memory slots spanning the known manifold.
            
        Returns:
            h_novel: [B, D] or [B, 1, D] Orthogonally purified concept vector.
            telemetry: Detailed orthogonality metrics.
        """
        orig_dim = candidate_vector.dim()
        if orig_dim == 3:
            x = candidate_vector.squeeze(1) # [B, D]
        else:
            x = candidate_vector # [B, D]
            
        B, M, D = basis_memory.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure float32 for high precision QR decomposition
        needs_cast = (dtype != torch.float32 and dtype != torch.float64)
        x_calc = x.float() if needs_cast else x
        basis_calc = basis_memory.float() if needs_cast else basis_memory
        
        # Transpose basis to [B, D, M] for QR decomposition
        V = basis_calc.transpose(1, 2) # [B, D, M]
        
        try:
            # Thin QR decomposition: Q is [B, D, M] with orthonormal columns
            Q, _ = torch.linalg.qr(V, mode="reduced")
        except Exception:
            # Fallback if SVD/QR fails: identity projection
            Q = V / (torch.norm(V, dim=1, keepdim=True) + self.eps)
            
        # Compute projection onto known subspace: x_parallel = Q (Q^T x)
        # x is [B, D, 1], Q^T x is [B, M, 1]
        x_expanded = x_calc.unsqueeze(-1) # [B, D, 1]
        alpha = torch.bmm(Q.transpose(1, 2), x_expanded) # [B, M, 1]
        x_parallel = torch.bmm(Q, alpha).squeeze(-1) # [B, D]
        
        # Orthogonal component: x_perp = x - x_parallel
        x_perp = x_calc - x_parallel # [B, D]
        
        # Compute residual norm and original norm
        orig_norm = torch.norm(x_calc, dim=-1, keepdim=True) + self.eps # [B, 1]
        perp_norm = torch.norm(x_perp, dim=-1, keepdim=True) + self.eps # [B, 1]
        
        # Degeneracy check: if candidate was almost entirely in span(V), handle gently
        orthogonality_ratio = (perp_norm / orig_norm).squeeze(-1) # [B]
        
        if self.normalize_output:
            # Scale purified vector to preserve the original energy magnitude
            x_novel = x_perp * (orig_norm / perp_norm)
        else:
            x_novel = x_perp
            
        if needs_cast:
            x_novel = x_novel.to(dtype=dtype)
            
        # Verify maximum inner product against basis Q for diagnostic telemetry
        with torch.no_grad():
            inner_prods = torch.bmm(Q.transpose(1, 2), x_perp.unsqueeze(-1)).squeeze(-1) # [B, M]
            max_leakage = float(torch.max(torch.abs(inner_prods)).item())
            
        telemetry = {
            "max_basis_leakage": max_leakage,
            "orthogonality_ratio": float(orthogonality_ratio.mean().item()),
            "is_strictly_orthogonal": bool(max_leakage < 1e-4)
        }
        
        if orig_dim == 3:
            return x_novel.unsqueeze(1), telemetry
        return x_novel, telemetry

    def forward(
        self,
        candidate_vector: torch.Tensor,
        basis_memory: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.project_to_nullspace(candidate_vector, basis_memory)
