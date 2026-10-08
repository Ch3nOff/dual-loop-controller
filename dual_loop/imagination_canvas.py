"""
Emergent Latent Imagination Canvas (ELIC / H-DREAM)
===================================================
HADL v4.5+ Mental Simulation & Unconstrained Concept Assembly Engine.

Core Principles:
1. Zero Hardcoding:
   Does NOT hardcode static physics, chemistry, or symbolic rules.
   Instead, operates on continuous, free-form latent concept fragments.
2. Incoherent-to-Coherent Self-Assembly:
   Fragments that appear unconstrained or disjoint to humans are dynamically
   connected, modulated, and understood by the model through a learned associative
   flux tensor field.
3. Energy-Bounded Mental Simulation ("Dream Cycles"):
   Simulates multi-step emergent interactions (K_dream steps) in latent space
   before collapsing back into text tokens, grounding the model's outputs in
   coherent structural reality rather than surface ngram statistics.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class EmergentLatentCanvas(nn.Module):
    """
    Emergent Latent Imagination Canvas (ELIC).
    
    Deconstructs hidden state h into M free-form concept fragments, simulates their
    continuous non-linear interaction across K dream steps via an asymmetric flux field,
    evaluates latent coherence, and collapses the resolved mental state back to R^D.
    """
    def __init__(
        self,
        d_model: int = 2048,
        num_fragments: int = 8,
        dream_steps: int = 2,
        contraction_factor: float = 0.20,
        enable_stochastic_flux: bool = False,
        confusion_tolerance: float = 0.50,
        enable_confusion_gating: bool = True
    ):
        super().__init__()
        # Adaptive fragment sizing if d_model is not evenly divisible
        if d_model % num_fragments != 0:
            for candidate in [8, 4, 2, 1]:
                if d_model % candidate == 0:
                    num_fragments = candidate
                    break

        self.d_model = d_model
        self.num_fragments = num_fragments
        self.d_frag = d_model // num_fragments
        self.dream_steps = dream_steps
        self.contraction_factor = float(contraction_factor)
        self.enable_stochastic_flux = enable_stochastic_flux
        self.confusion_tolerance = float(confusion_tolerance)
        self.enable_confusion_gating = enable_confusion_gating
        
        assert d_model % num_fragments == 0, (
            f"d_model ({d_model}) must be divisible by num_fragments ({num_fragments})"
        )
        
        # 1. Cognitive Confusion & Context Ambiguity Sensor:
        # Measures whether the meaning/context of a token is ambiguous ("bingung")
        self.w_confusion = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.w_confusion.weight, std=0.01)
        nn.init.constant_(self.w_confusion.bias, 0.0)
        
        # 2. Fragment Deconstructor (D -> M x d_frag)
        self.w_unroll = nn.Linear(d_model, d_model, bias=False)
        self.norm_fragments = nn.LayerNorm(self.d_frag)
        
        # 3. Continuous Associative Flux Tensor (Asymmetric Causal Directionality)
        self.w_flux = nn.Linear(self.d_frag, self.d_frag, bias=False)
        self.w_evolve = nn.Linear(self.d_frag, self.d_frag, bias=False)
        self.norm_evolve = nn.LayerNorm(self.d_frag)
        
        # 4. Holographic Canvas Collapse (M x d_frag -> D)
        self.w_collapse = nn.Linear(d_model, d_model, bias=False)
        self.norm_collapse = nn.LayerNorm(d_model)
        
        # ReZero Gating parameter (starts small to guarantee initial identity preservation)
        self.alpha_canvas = nn.Parameter(torch.tensor(0.05))
        
        self.scale = 1.0 / math.sqrt(self.d_frag)
        
        # Initialize projections orthogonally / small variance
        nn.init.orthogonal_(self.w_unroll.weight)
        nn.init.normal_(self.w_flux.weight, std=0.02)
        nn.init.orthogonal_(self.w_evolve.weight)
        nn.init.zeros_(self.w_collapse.weight)

    def forward(
        self,
        h: torch.Tensor,
        return_energy: bool = False,
        return_confusion: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Executes emergent mental simulation on hidden representation h.
        
        Args:
            h: Hidden state tensor of shape (B, L, D) or (B, D)
            return_energy: Whether to return the latent coherence energy
            
        Returns:
            delta_dream: Grounded mental simulation delta (B, L, D) or (B, D)
            energy: Optional latent dissonance energy scalar/tensor
        """
        orig_shape = h.shape
        if h.ndim == 2:
            h = h.unsqueeze(1)  # (B, 1, D)
            
        B, L, D = h.shape
        N_tokens = B * L
        
        # 0. Cognitive Confusion Assessment (Bingung / Ambiguitas Konteks):
        confusion_gate = None
        confusion_scores = None
        if self.enable_confusion_gating:
            confusion_raw = torch.sigmoid(self.w_confusion(h))  # (B, L, 1)
            confusion_scores = confusion_raw
            is_confused = confusion_raw > self.confusion_tolerance  # (B, L, 1)
            
            # If no token in the sequence exceeds the confusion tolerance threshold,
            # bypass dream cycles completely (model understands clearly, zero imagination overhead)!
            if not is_confused.any():
                delta_dream = torch.zeros(B, L, D, device=h.device, dtype=h.dtype)
                if len(orig_shape) == 2:
                    delta_dream = delta_dream.squeeze(1)
                if return_confusion:
                    return delta_dream, None, confusion_scores
                return delta_dream, None
                
            # Gate scales smoothly with excess confusion above tolerance:
            confusion_gate = torch.where(
                is_confused,
                (confusion_raw - self.confusion_tolerance) / max(1e-4, 1.0 - self.confusion_tolerance),
                torch.zeros_like(confusion_raw)
            )

        # 1. Deconstruct into M free-form concept fragments: (N_tokens, M, d_frag)
        h_flat = h.reshape(N_tokens, D)
        unrolled = F.gelu(self.w_unroll(h_flat))
        fragments = self.norm_fragments(unrolled.reshape(N_tokens, self.num_fragments, self.d_frag))
        
        # 2. Emergent Mental Simulation Loop (Dream Cycles)
        F_curr = fragments
        for step in range(self.dream_steps):
            # Symmetric associative affinity (concept similarity)
            dot_aff = torch.bmm(F_curr, F_curr.transpose(1, 2)) * self.scale
            
            # Asymmetric directional flux (cause-and-effect / influence flow)
            flux_target = self.w_flux(F_curr)
            flux_aff = torch.bmm(F_curr, flux_target.transpose(1, 2)) * self.scale
            
            # Continuous Associative Field (Softmax over canvas fragments)
            field = F.softmax(dot_aff + flux_aff, dim=-1)  # (N_tokens, M, M)
            
            # Evolve fragments along the associative field
            evolved = self.norm_evolve(self.w_evolve(F_curr))
            dF = torch.tanh(torch.bmm(field, evolved))  # (N_tokens, M, d_frag)
            
            # Optional stochastic flux (dream thermal exploration)
            if self.enable_stochastic_flux and self.training:
                xi = torch.randn_like(dF) * 0.01
                dF = dF + xi
                
            # Contractive update: bounded energy dissipation
            F_curr = F_curr + self.contraction_factor * dF
            
        # 3. Energy Coherence Evaluation: Measure fragment dispersion
        energy = None
        if return_energy:
            # Low variance among assembled fragments indicates stable mental consensus
            centroid = F_curr.mean(dim=1, keepdim=True)
            energy = torch.mean(torch.norm(F_curr - centroid, p=2, dim=-1))
            
        # 4. Holographic Canvas Collapse: condense M fragments back to D
        collapsed_flat = F_curr.reshape(N_tokens, D)
        delta_dream = torch.tanh(self.alpha_canvas) * self.w_collapse(self.norm_collapse(collapsed_flat))
        delta_dream = delta_dream.reshape(B, L, D)
        
        # Modulate dream contribution by confusion gate:
        if confusion_gate is not None:
            delta_dream = delta_dream * confusion_gate
        
        if len(orig_shape) == 2:
            delta_dream = delta_dream.squeeze(1)
            
        if return_confusion:
            return delta_dream, energy, confusion_scores
        return delta_dream, energy
