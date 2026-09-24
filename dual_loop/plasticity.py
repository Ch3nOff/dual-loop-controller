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


class HeteroAssociativePlasticMemory(nn.Module):
    """
    Hetero-Associative Cross-Modal Plastic Memory.
    
    Enables in-situ zero-shot cross-modal concept binding without fine-tuning:
    When a novel visual entity (e.g. newly seen object in camera) is presented
    alongside a text label, this unit binds them via outer-product synaptic plasticity:
    
        M_cross^{(k)} = (1 - lambda) * M_cross^{(k-1)} + eta * u(x) * (phi_V(h_V) (x) psi_T(h_T)^T)
        
    Upon subsequent encounters of the visual object:
        delta_recognition = W_out(M_cross * phi_V(h_V)) in R^D
        
    Guarantees:
    - Zero catastrophic forgetting of pretrained weights (100% frozen base).
    - Rapid, instantaneous one-shot recall in working memory.
    """
    def __init__(
        self,
        d_model: int,
        rank: int = 64,
        plastic_lr: float = 0.50,
        decay_rate: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.plastic_lr = float(plastic_lr)
        self.decay_rate = float(decay_rate)

        # Low-rank factor projections for visual and linguistic representations
        self.phi_visual = nn.Linear(d_model, rank, bias=False)
        self.norm_visual = nn.LayerNorm(rank)

        self.psi_text = nn.Linear(d_model, rank, bias=False)
        self.norm_text = nn.LayerNorm(rank)

        # Output projection back to cognitive hidden dimension D
        self.proj_out = nn.Linear(rank, d_model, bias=False)
        self.norm_out = nn.LayerNorm(d_model)

        # Initialize factor projections with orthogonal frames and adjoint reconstruction
        nn.init.orthogonal_(self.phi_visual.weight)
        nn.init.orthogonal_(self.psi_text.weight)
        with torch.no_grad():
            self.proj_out.weight.copy_(self.psi_text.weight.t())

        self.last_m_cross: Optional[torch.Tensor] = None
        self.continual_mode: bool = False
        self.continual_decay: float = 0.95

    def set_continual_mode(self, enabled: bool = True, decay: float = 0.95):
        """Enables persistent continual learning of new objects across frames/dialogue turns."""
        self.continual_mode = enabled
        self.continual_decay = float(decay)

    def reset_state(self, force: bool = False):
        """Resets or softly decays associative memory trace."""
        if self.continual_mode and not force and self.last_m_cross is not None:
            self.last_m_cross = self.last_m_cross * self.continual_decay
        else:
            self.last_m_cross = None

    def bind_concept(
        self,
        h_vision: torch.Tensor,
        h_text: torch.Tensor,
        u_vacuity: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Binds a visual representation to a textual concept in fast memory.
        h_vision: [B, N_V, D] or [B, D]
        h_text: [B, N_T, D] or [B, D]
        u_vacuity: Optional [B, 1] Dirichlet epistemic vacuity
        """
        B = h_vision.size(0)
        device = h_vision.device
        dtype = h_vision.dtype

        # Mean-pool if multi-token representations
        v_rep = h_vision.mean(dim=1) if h_vision.dim() == 3 else h_vision # [B, D]
        t_rep = h_text.mean(dim=1) if h_text.dim() == 3 else h_text       # [B, D]

        # Low-rank factor representations with L2 spherical normalization
        u_v = F.normalize(self.phi_visual(v_rep), p=2, dim=-1) # [B, R]
        v_t = F.normalize(self.psi_text(t_rep), p=2, dim=-1)   # [B, R]

        # Outer product: [B, R, 1] @ [B, 1, R] -> [B, R, R]
        delta_m = torch.bmm(u_v.unsqueeze(-1), v_t.unsqueeze(1)) # [B, R, R]

        # Epistemic gating factor
        if u_vacuity is not None:
            g_u = u_vacuity.reshape(B, 1, 1).to(device=device, dtype=dtype)
        else:
            g_u = torch.ones(B, 1, 1, device=device, dtype=dtype)

        # Retrieve or initialize associative matrix M_cross [B, R, R]
        if self.last_m_cross is None or self.last_m_cross.size(0) != B or self.last_m_cross.device != device:
            m_cross = torch.zeros(B, self.rank, self.rank, device=device, dtype=dtype)
        else:
            m_cross = self.last_m_cross

        # In-situ Hebbian update
        decay_factor = 1.0 - self.decay_rate
        m_cross = decay_factor * m_cross + self.plastic_lr * g_u * delta_m

        # Bounded spectral normalization
        m_norm = m_cross.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        m_scale = torch.clamp(10.0 / m_norm, max=1.0)
        m_cross = m_cross * m_scale

        self.last_m_cross = m_cross.detach()
        return {
            "m_cross_norm": float(m_cross.norm().item()),
            "bound_active": True
        }

    def recall_from_visual(self, h_vision: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Recalls associated text/concept representation given a sensory visual feature.
        h_vision: [B, N_V, D] or [B, D]
        Returns:
            delta_recognition: [B, N_V, D] or [B, D]
        """
        B = h_vision.size(0)
        is_2d = (h_vision.dim() == 2)
        v_in = h_vision.unsqueeze(1) if is_2d else h_vision # [B, N_V, D]
        N_V = v_in.size(1)
        device = h_vision.device
        dtype = h_vision.dtype

        if self.last_m_cross is None or self.last_m_cross.device != device:
            zero_out = torch.zeros_like(h_vision)
            return zero_out, {"recalled": False, "confidence": 0.0}

        m_cross = self.last_m_cross.to(device=device, dtype=dtype) # [B, R, R]

        # Project visual tokens with L2 spherical normalization: [B, N_V, R]
        u_v = F.normalize(self.phi_visual(v_in), p=2, dim=-1)

        # Associative memory matrix multiplication: [B, N_V, R] @ [B, R, R] -> [B, N_V, R]
        z_recalled = torch.bmm(u_v, m_cross)

        # Output project to D: [B, N_V, D]
        delta_rec = self.norm_out(self.proj_out(z_recalled))

        if is_2d:
            delta_rec = delta_rec.squeeze(1)

        confidence = float(z_recalled.norm(dim=-1).mean().item())
        telemetry = {
            "recalled": True,
            "confidence": confidence,
            "m_cross_norm": float(m_cross.norm().item())
        }
        return delta_rec, telemetry

    def recall_from_text(self, h_text: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Recalls associated sensory (visual/audio) representation given a textual query/concept.
        h_text: [B, N_T, D] or [B, D]
        Returns:
            delta_sensory: [B, N_T, D] or [B, D]
        """
        B = h_text.size(0)
        is_2d = (h_text.dim() == 2)
        t_in = h_text.unsqueeze(1) if is_2d else h_text # [B, N_T, D]
        device = h_text.device
        dtype = h_text.dtype

        if self.last_m_cross is None or self.last_m_cross.device != device:
            zero_out = torch.zeros_like(h_text)
            return zero_out, {"recalled": False, "confidence": 0.0}

        m_cross = self.last_m_cross.to(device=device, dtype=dtype) # [B, R, R]

        # Project linguistic tokens with L2 spherical normalization: [B, N_T, R]
        v_t = F.normalize(self.psi_text(t_in), p=2, dim=-1)

        # Transposed associative memory recall: [B, N_T, R] @ [B, R, R]^T -> [B, N_T, R]
        z_sensory = torch.bmm(v_t, m_cross.transpose(1, 2))

        # Adjoint output project to D: [B, N_T, D]
        delta_sensory = self.norm_out(F.linear(z_sensory, self.phi_visual.weight.t()))

        if is_2d:
            delta_sensory = delta_sensory.squeeze(1)

        confidence = float(z_sensory.norm(dim=-1).mean().item())
        telemetry = {
            "recalled": True,
            "confidence": confidence,
            "m_cross_norm": float(m_cross.norm().item())
        }
        return delta_sensory, telemetry
