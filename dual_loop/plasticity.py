import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Any

class PlasticFastWeightUnit(nn.Module):
    """
    In-Situ Plastic Fast-Weight Unit (Low-Rank Associative Virtual Parameter Generator).
    
    Equips the cognitive controller with dynamic synaptic plasticity during inference.
    When confronting non-protocol / out-of-distribution scenarios, this unit synthesizes
    and adapts local virtual parameters W_virtual in-situ without requiring backpropagation
    into the frozen 2B base model.
    
    Mathematical Formulation:
        Rank R << D (e.g. R=32, D=2048):
        u_k = GELU(W_u(thoughts)) in R^{B, L, R}
        v_k = LayerNorm(W_v(discrepancy)) in R^{B, L, R}
        
        Associative outer-product trace:
        Delta_M = (1/L) * sum_{l=1}^L (v_{k, l} @ u_{k, l}^T) in R^{B, R, R}
        
        Hebbian Plastic Synaptic Update (modulated by epistemic vacuity u(x)):
        M_fast^{(k)} = (1 - lambda_decay) * M_fast^{(k-1)} + eta_plastic * u(x) * Delta_M
        
        Plastic Virtual Transformation:
        z_fast = u_k @ M_fast^{(k)} in R^{B, L, R}
        delta_plastic = W_out(z_fast) in R^{B, L, D}
        
    Thread & Request Safety:
        M_fast is an in-memory trace initialized to zeros at the beginning of each
        forward pass and explicitly cleared by `reset_state()` to prevent cross-request leakage.
    """
    def __init__(
        self,
        d_model: int,
        rank: int = 32,
        plastic_lr: float = 0.15,
        decay_rate: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.plastic_lr = float(plastic_lr)
        self.decay_rate = float(decay_rate)
        
        # Fast weight factor projections
        self.proj_u = nn.Linear(d_model, rank, bias=False)
        self.proj_v = nn.Linear(d_model, rank, bias=False)
        self.norm_v = nn.LayerNorm(rank)
        
        self.proj_out = nn.Linear(rank, d_model)
        self.norm_out = nn.LayerNorm(d_model)
        
        # Initialize output projection to start small and gentle
        nn.init.normal_(self.proj_out.weight, std=0.01)
        nn.init.zeros_(self.proj_out.bias)
        
        # Instance state for in-situ memory trace (B, R, R)
        self.last_m_fast: Optional[torch.Tensor] = None
        self.last_update_norm: float = 0.0
        self.continual_mode: bool = False
        self.continual_decay: float = 0.90

    def set_continual_mode(self, enabled: bool = True, decay: float = 0.90):
        """Toggles continual learning mode across requests/trials."""
        self.continual_mode = enabled
        self.continual_decay = float(decay)

    def reset_state(self, force: bool = False):
        """Resets virtual parameters, or applies soft decay if in continual mode."""
        if self.continual_mode and not force and self.last_m_fast is not None:
            self.last_m_fast = self.last_m_fast * self.continual_decay
            self.last_update_norm = 0.0
        else:
            self.last_m_fast = None
            self.last_update_norm = 0.0

    def apply_critique_falsification(
        self,
        thoughts: torch.Tensor,
        critique_vector: torch.Tensor,
        u_epistemic: Optional[torch.Tensor] = None,
        anti_lr: float = 0.20
    ) -> Dict[str, Any]:
        """
        Anti-Hebbian / Orthogonalizing perturbation to escape ambiguous local minima
        discovered during Pass 1 uncertainty audits.
        """
        B, L, D = thoughts.shape
        device = thoughts.device
        dtype = thoughts.dtype
        if critique_vector.dim() == 2:
            critique_vector = critique_vector.unsqueeze(1).expand(-1, L, -1)
            
        u_k = F.gelu(self.proj_u(thoughts))
        v_critique = self.norm_v(self.proj_v(critique_vector))
        delta_m_critique = torch.bmm(v_critique.transpose(1, 2), u_k) / float(L)
        
        g_u = u_epistemic.reshape(B, 1, 1).to(device=device, dtype=dtype) if u_epistemic is not None else 1.0
        anti_update = -anti_lr * g_u * delta_m_critique
        
        if self.last_m_fast is None:
            self.last_m_fast = torch.zeros(B, self.rank, self.rank, device=device, dtype=dtype)
        self.last_m_fast = self.last_m_fast + anti_update
        m_norm = self.last_m_fast.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        self.last_m_fast = self.last_m_fast * torch.clamp(10.0 / m_norm, max=1.0)
        return {"anti_update_norm": float(anti_update.norm().item())}

    def forward(
        self,
        thoughts: torch.Tensor,
        discrepancy: torch.Tensor,
        u_epistemic: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes an in-situ Hebbian plastic step and transforms thoughts via virtual weights.
        
        Args:
            thoughts: [B, L, D] Current thought tokens.
            discrepancy: [B, L, D] or [B, D] Discrepancy / critique signal against context.
            u_epistemic: Optional [B] or [B, 1] Epistemic uncertainty from EvidentialEpistemicGate.
            
        Returns:
            delta_plastic: [B, L, D] Corrective perturbation from synthesized virtual parameters.
            telemetry: Diagnostic dict containing plastic trace norms.
        """
        B, L, D = thoughts.shape
        device = thoughts.device
        dtype = thoughts.dtype
        
        # Expand discrepancy if 2D [B, D] -> [B, L, D]
        if discrepancy.dim() == 2:
            discrepancy = discrepancy.unsqueeze(1).expand(-1, L, -1)
            
        # Epistemic vacuity gating factor [B, 1, 1]
        if u_epistemic is not None:
            g_u = u_epistemic.reshape(B, 1, 1).to(device=device, dtype=dtype)
        else:
            g_u = torch.ones(B, 1, 1, device=device, dtype=dtype)
            
        # 1. Compute low-rank associative representations
        u_k = F.gelu(self.proj_u(thoughts))               # [B, L, R]
        v_k = self.norm_v(self.proj_v(discrepancy))       # [B, L, R]
        
        # 2. Batch associative outer product: [B, R, L] @ [B, L, R] -> [B, R, R]
        # Delta_M = sum_l (v_l @ u_l^T) / L
        v_trans = v_k.transpose(1, 2)                     # [B, R, L]
        delta_m = torch.bmm(v_trans, u_k) / float(L)      # [B, R, R]
        
        # 3. Retrieve or initialize fast weight trace M_fast in [B, R, R]
        if self.last_m_fast is None or self.last_m_fast.size(0) != B or self.last_m_fast.device != device:
            m_fast = torch.zeros(B, self.rank, self.rank, device=device, dtype=dtype)
        else:
            m_fast = self.last_m_fast
            
        # 4. In-Situ Hebbian Plastic Update:
        # M_fast = (1 - decay) * M_fast + lr * u(x) * Delta_M
        decay_factor = 1.0 - self.decay_rate
        plastic_update = self.plastic_lr * g_u * delta_m  # [B, R, R]
        m_fast = decay_factor * m_fast + plastic_update   # [B, R, R]
        
        # Bounded spectral normalization on M_fast to prevent runaway excitation
        m_norm = m_fast.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        max_allowed_norm = 10.0
        m_scale = torch.clamp(max_allowed_norm / m_norm, max=1.0)
        m_fast = m_fast * m_scale
        
        self.last_m_fast = m_fast.detach()
        self.last_update_norm = float(plastic_update.norm().item())
        
        # 5. Transform thoughts through synthesized virtual parameters:
        # z_fast = u_k @ M_fast -> [B, L, R] @ [B, R, R] -> [B, L, R]
        z_fast = torch.bmm(u_k, m_fast)                   # [B, L, R]
        delta_plastic = self.norm_out(self.proj_out(z_fast)) # [B, L, D]
        
        telemetry = {
            "m_fast_norm": float(m_fast.norm().item()),
            "plastic_update_norm": self.last_update_norm,
            "plastic_delta_norm": float(delta_plastic.norm().item())
        }
        
        return delta_plastic, telemetry
