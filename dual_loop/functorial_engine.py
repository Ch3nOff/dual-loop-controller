import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class FunctorialCrossDomainMapper(nn.Module):
    r"""
    Relational Morphism Cross-Domain Mapper (Inspired by Functorial Category Theory).
    
    THEORETICAL SCOPE & TRANSPARENCY:
    This module performs structural analogical mapping across domains by extracting
    pairwise relational morphisms (attention graphs between memory slots) and learning
    a projection network F: C -> D between domain graphs.
    
    NOTE ON CATEGORY THEORY GUARANTEES:
    While conceptually motivated by functor composition F(f \circ g) = F(f) \circ F(g),
    diagram commutativity is NOT axiomatically guaranteed by construction. Instead,
    the empirical divergence ||F(f \circ f) - F(f) \circ F(f)|| is tracked as
    `commutativity_error` (and can be minimized via `compute_commutativity_loss()`).
    Users should view this as an empirical relational alignment mechanism rather than
    a strict mathematical category-theoretic isomorphism.
    """
    def __init__(
        self,
        d_model: int,
        num_slots: int = 16,
        n_heads: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.n_heads = n_heads
        
        # Morphism relation projector (extracts M x M relational graph between memory slots)
        self.q_rel = nn.Linear(d_model, d_model)
        self.k_rel = nn.Linear(d_model, d_model)
        
        # Functor morphism mapping network
        self.functor_map = nn.Sequential(
            nn.Linear(num_slots, num_slots),
            nn.GELU(),
            nn.Linear(num_slots, num_slots)
        )
        
        # Target domain structural reconstruction projection
        self.target_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def extract_morphisms(self, memory_slots: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise relational morphisms M in R^{B, num_slots, num_slots}.
        """
        B, M, D = memory_slots.shape
        Q = self.q_rel(memory_slots) # [B, M, D]
        K = self.k_rel(memory_slots) # [B, M, D]
        
        # Morphism matrix: how each slot relates/transforms into every other slot
        scores = torch.bmm(Q, K.transpose(1, 2)) / (D ** 0.5) # [B, M, M]
        morphisms = F.softmax(scores, dim=-1)
        return morphisms

    def forward(
        self,
        source_memory: torch.Tensor,
        target_anchor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Transfers source relational structure onto target anchor domain.
        
        Args:
            source_memory: [B, M, D] Structured context memory from source domain.
            target_anchor: [B, D] or [B, 1, D] Target domain anchor representation.
            
        Returns:
            structured_analog: [B, M, D] Target slots endowed with source relational morphisms.
            telemetry: Commutativity loss and morphism metrics.
        """
        B, M, D = source_memory.shape
        if target_anchor.dim() == 2:
            target_anchor = target_anchor.unsqueeze(1) # [B, 1, D]
            
        # 1. Extract source domain morphisms f, g
        f_source = self.extract_morphisms(source_memory) # [B, M, M]
        
        # 2. Apply Functor F: C -> D
        f_target = self.functor_map(f_source) # [B, M, M]
        f_target = F.softmax(f_target, dim=-1)
        
        # 3. Verify Diagram Commutativity: F(f \circ f) vs F(f) \circ F(f)
        with torch.no_grad():
            f_composed_source = torch.bmm(f_source, f_source)
            f_composed_target_predicted = self.functor_map(f_composed_source)
            f_composed_target_actual = torch.bmm(f_target, f_target)
            commutativity_error = float(torch.norm(f_composed_target_predicted - f_composed_target_actual).item())
            
        # 4. Project target anchor through structural morphism graph
        anchor_expanded = target_anchor.expand(B, M, D) # [B, M, D]
        structured_analog = torch.bmm(f_target, anchor_expanded) # [B, M, D]
        structured_analog = self.norm(self.target_proj(structured_analog) + anchor_expanded)
        
        telemetry = {
            "commutativity_error": commutativity_error,
            "morphism_entropy": float((-f_target * torch.log(f_target + 1e-9)).sum(dim=-1).mean().item()),
            "functor_applied": True
        }
        return structured_analog, telemetry

    def compute_commutativity_loss(self, source_memory: torch.Tensor) -> torch.Tensor:
        r"""
        Computes differentiable commutativity loss ||F(f \circ f) - F(f) \circ F(f)||^2.
        Can be used as an auxiliary regularization objective during training to penalize
        functor diagram inconsistency.
        """
        f_source = self.extract_morphisms(source_memory)
        f_target = F.softmax(self.functor_map(f_source), dim=-1)
        f_composed_source = torch.bmm(f_source, f_source)
        f_composed_target_pred = self.functor_map(f_composed_source)
        f_composed_target_actual = torch.bmm(f_target, f_target)
        return F.mse_loss(f_composed_target_pred, f_composed_target_actual)
