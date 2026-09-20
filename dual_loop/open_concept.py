import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Any

class OpenConceptSynthesizer(nn.Module):
    """
    Open-Concept Continuous Manifold Synthesizer.
    
    When confronting completely unprecedented / out-of-vocabulary phenomena
    (evidential epistemic vacuity u(x) >= tau_unseen), this unit synthesizes
    a new continuous semantic prototype vector c* in R^D that occupies an unassigned
    coordinate in latent manifold space.
    
    This concept vector is dynamically injected into Cognitive Working Memory (CWM),
    enabling the network to reference and reason about novel concepts without requiring
    an existing discrete token in the pre-trained vocabulary.
    
    Mathematical Formulation:
        Given query anchor h_anchor in R^{B, D} and deliberation discrepancy e_K in R^{B, D}:
        Delta_c = GELU(W_proto e_K)
        c* = LayerNorm(h_anchor + Delta_c) in R^{B, 1, D}
        
        The synthesized prototype is appended into the memory buffer:
        Memory_ext = [Memory_CWM ; c*]
    """
    def __init__(
        self,
        d_model: int,
        tau_unseen: float = 0.65,
        enable_nullspace_projection: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_unseen = float(tau_unseen)
        self.enable_nullspace_projection = enable_nullspace_projection
        
        # Prototype synthesis projection
        self.proto_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.norm_proto = nn.LayerNorm(d_model)
        
        if self.enable_nullspace_projection:
            from .nullspace_engine import OrthogonalNullspaceProjector
            self.nullspace_projector = OrthogonalNullspaceProjector(d_model=d_model)
        else:
            self.nullspace_projector = None
        
        # Initialize final layer with small scale
        nn.init.normal_(self.proto_net[-1].weight, std=0.02)
        nn.init.zeros_(self.proto_net[-1].bias)

    def forward(
        self,
        h_anchor: torch.Tensor,
        discrepancy: torch.Tensor,
        vacuity_u: torch.Tensor,
        basis_memory: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Synthesizes a continuous semantic prototype if vacuity exceeds threshold.
        
        Args:
            h_anchor: [B, D] or [B, 1, D] Base query representation.
            discrepancy: [B, D] or [B, 1, D] Metacognitive critique discrepancy vector.
            vacuity_u: [B] Epistemic vacuity score in [0, 1].
            basis_memory: Optional [B, M, D] existing memory slots for orthogonal nullspace projection.
            
        Returns:
            prototype: Optional [B, 1, D] Synthesized prototype vector, or None if u < tau_unseen.
            telemetry: Diagnostic dict.
        """
        B = h_anchor.size(0)
        device = h_anchor.device
        dtype = h_anchor.dtype
        
        if h_anchor.dim() == 2:
            h_anchor = h_anchor.unsqueeze(1) # [B, 1, D]
        if discrepancy.dim() == 2:
            discrepancy = discrepancy.unsqueeze(1) # [B, 1, D]
            
        # Check per-sample whether unseen novelty is triggered
        mask_unseen = (vacuity_u >= self.tau_unseen) # [B]
        
        if not mask_unseen.any():
            return None, {
                "synthesized_count": 0,
                "prototype_norm": 0.0,
                "is_active": False,
                "nullspace_telemetry": {}
            }
            
        # Synthesize prototype vector: c* = LayerNorm(h_anchor + W(e))
        delta_c = self.proto_net(discrepancy) # [B, 1, D]
        
        nullspace_telem = {}
        if self.nullspace_projector is not None and basis_memory is not None:
            # Purify delta_c to lie strictly in the orthogonal nullspace of basis_memory
            delta_c, nullspace_telem = self.nullspace_projector(delta_c, basis_memory)
            
        c_star = self.norm_proto(h_anchor + delta_c) # [B, 1, D]
        
        # Apply mask: only samples that exceed tau_unseen retain the synthesized concept
        mask_expanded = mask_unseen.reshape(B, 1, 1).expand(-1, 1, self.d_model)
        c_star_gated = torch.where(mask_expanded, c_star, torch.zeros_like(c_star))
        
        telemetry = {
            "synthesized_count": int(mask_unseen.sum().item()),
            "prototype_norm": float(c_star_gated.norm().item()),
            "is_active": True,
            "nullspace_telemetry": nullspace_telem
        }
        
        return c_star_gated, telemetry
