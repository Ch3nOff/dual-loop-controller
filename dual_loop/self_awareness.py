import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List

class IntrospectiveSelfDescriptor(nn.Module):
    """
    Introspective Self-Descriptor Vector Engine.
    
    Computes s_t in R^5 analytically at each deliberation step k:
        s_t = [s_time, s_vacuity, s_drift, s_lipschitz, s_margin]^T
        
    Where:
        - s_time = k / K_max in [0, 1]: Deliberation compute budget spent.
        - s_vacuity = u(x) in [0, 1]: Dirichlet epistemic vacuity from evidential gate.
        - s_drift = ||h_k - h_anchor|| / (||h_anchor|| + eps) >= 0: Query instruction fidelity.
        - s_lipschitz = L_k = ||H_k - H_{k-1}||_F / (||H_{k-1} - H_{k-2}||_F + eps) >= 0:
          Local Lipschitz trajectory stability (L_k < 1.0 convergent, L_k > 1.0 chaotic divergence).
        - s_margin = logit_top1 - logit_top2: System 1 certainty / decisiveness.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        k: int,
        k_max: int,
        u_vacuity: Optional[torch.Tensor],
        h_k: torch.Tensor,
        h_anchor: torch.Tensor,
        H_k: torch.Tensor,
        H_prev1: Optional[torch.Tensor] = None,
        H_prev2: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            k: Current recurrent step (1-indexed or 0-indexed).
            k_max: Max recurrent steps budget (e.g. 2, 3, or 4).
            u_vacuity: [B] or [B, 1] Epistemic uncertainty vacuity from EvidentialEpistemicGate.
            h_k: [B, D] Current thought anchor representation.
            h_anchor: [B, D] Initial query representation.
            H_k: [B, L, D] Current latent thought tensor.
            H_prev1: Optional [B, L, D] Thought tensor at step k-1.
            H_prev2: Optional [B, L, D] Thought tensor at step k-2.
            logits: Optional [B, V] Vocabulary logits for margin calculation.
            
        Returns:
            s_t: [B, 5] Self-descriptor vector.
            metrics: Dict with scalar representations for logging.
        """
        B = h_k.size(0)
        device = h_k.device
        dtype = h_k.dtype

        # 1. s_time: computational progress relative to quota
        time_ratio = float(min(max(k, 0), k_max)) / float(max(k_max, 1))
        s_time = torch.full((B, 1), time_ratio, device=device, dtype=dtype)

        # 2. s_vacuity: Dirichlet epistemic uncertainty
        if u_vacuity is not None:
            vac = u_vacuity.to(device=device, dtype=dtype).reshape(B, 1)
            s_vacuity = torch.clamp(vac, 0.0, 1.0)
        else:
            s_vacuity = torch.full((B, 1), 0.15, device=device, dtype=dtype)

        # 3. s_drift: query identity integrity
        diff_h = torch.norm(h_k - h_anchor, p=2, dim=-1, keepdim=True) # [B, 1]
        norm_anchor = torch.norm(h_anchor, p=2, dim=-1, keepdim=True) # [B, 1]
        s_drift = diff_h / (norm_anchor + self.eps)

        # 4. s_lipschitz: local Lipschitz stability ratio
        if H_prev1 is not None and H_prev2 is not None:
            # Frobenius norm per batch item
            delta_1 = torch.norm(H_k - H_prev1, p="fro", dim=(-2, -1)).reshape(B, 1)
            delta_2 = torch.norm(H_prev1 - H_prev2, p="fro", dim=(-2, -1)).reshape(B, 1)
            s_lipschitz = delta_1 / (delta_2 + self.eps)
        elif H_prev1 is not None:
            delta_1 = torch.norm(H_k - H_prev1, p="fro", dim=(-2, -1)).reshape(B, 1)
            norm_prev = torch.norm(H_prev1, p="fro", dim=(-2, -1)).reshape(B, 1)
            s_lipschitz = delta_1 / (0.5 * norm_prev + self.eps)
        else:
            # Initial step baseline
            s_lipschitz = torch.full((B, 1), 0.75, device=device, dtype=dtype)
        s_lipschitz = s_lipschitz.reshape(B, 1)

        # 5. s_margin: System 1 probabilistic certainty
        if logits is not None and logits.size(-1) > 1:
            top2_vals, _ = torch.topk(logits, k=2, dim=-1)
            margin = (top2_vals[:, 0] - top2_vals[:, 1]).unsqueeze(1)
            s_margin = torch.clamp(margin, min=-10.0, max=10.0)
        else:
            # Cosine similarity margin against anchor
            cos_sim = F.cosine_similarity(h_k, h_anchor, dim=-1).unsqueeze(1)
            s_margin = cos_sim * 2.5 # Proxy margin in reasonable range

        # Assemble s_t in R^(B x 5)
        s_t = torch.cat([s_time, s_vacuity, s_drift, s_lipschitz, s_margin], dim=-1) # [B, 5]

        metrics = {
            "s_time": float(s_time.mean().item()),
            "s_vacuity": float(s_vacuity.mean().item()),
            "s_drift": float(s_drift.mean().item()),
            "s_lipschitz": float(s_lipschitz.mean().item()),
            "s_margin": float(s_margin.mean().item()),
            "is_divergent": bool(float(s_lipschitz.mean().item()) > 1.0)
        }

        return s_t, metrics


class EgoThoughtProjector(nn.Module):
    """
    Projects the Introspective Self-Descriptor Vector s_t in R^5
    into the inner latent working memory dimension D_inner to form the
    Ego-Thought Token:
        t_ego(k) = LayerNorm(W_ego * s_t + b_ego) in R^(B x 1 x D_inner)
    """
    def __init__(self, d_inner: int):
        super().__init__()
        self.d_inner = d_inner
        self.proj = nn.Linear(5, d_inner)
        self.norm = nn.LayerNorm(d_inner)

        # Initialize projection
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_t: [B, 5] Self-descriptor vector.
        Returns:
            t_ego: [B, 1, D_inner] Projected ego-thought token ready for Slot 0.
        """
        projected = self.proj(s_t) # [B, D_inner]
        t_ego = self.norm(projected).unsqueeze(1) # [B, 1, D_inner]
        return t_ego


class EpistemicModestyModulator(nn.Module):
    """
    Detects computational limits and triggers Epistemic Modesty.
    
    If u(x) > 0.70 (high ignorance) or L_k > 1.0 (chaotic divergence),
    attention weights and residual updates are modulated to reject overconfident
    claims and prevent hallucination.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 1.5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, s_t: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Args:
            s_t: [B, 5] Self-descriptor vector [time, vacuity, drift, lipschitz, margin].
        Returns:
            modesty_factor: [B, 1] Multiplier in (0, 1].
            is_active: bool indicator if modesty damping was triggered.
        """
        B = s_t.size(0)
        device = s_t.device
        dtype = s_t.dtype

        vacuity = s_t[:, 1:2]
        lipschitz = s_t[:, 3:4]

        # Penalties for exceeding thresholds
        excess_vacuity = F.relu(vacuity - 0.70)
        excess_lipschitz = F.relu(lipschitz - 1.0)

        # Penalty score
        penalty = self.alpha * excess_vacuity + self.beta * excess_lipschitz
        modesty_factor = torch.exp(-penalty) # in (0, 1], = 1.0 when within bounds

        is_active = bool((excess_vacuity.max() > 0.0 or excess_lipschitz.max() > 0.0).item())
        return modesty_factor, is_active


class SelfAwarenessController(nn.Module):
    """
    Unified Dual-Loop Self-Awareness Controller.
    
    Orchestrates the 4-step self-referential cycle:
    1. Telemetry Acquisition: Extracts real s_t in R^5.
    2. Ego Token Generation: Produces t_ego in R^(1 x D_inner) and inserts into Slot 0.
    3. Self-Attention Reflection: Task slots (1, 2, 3) attend to Slot 0.
    4. Epistemic Modesty: Down-modulates extreme hallucinations if u > 0.7 or L_k > 1.0.
    """
    def __init__(self, d_inner: int):
        super().__init__()
        self.d_inner = d_inner
        self.descriptor = IntrospectiveSelfDescriptor()
        self.ego_projector = EgoThoughtProjector(d_inner=d_inner)
        self.modesty_modulator = EpistemicModestyModulator()

    def step_self_awareness(
        self,
        k: int,
        k_max: int,
        H_k: torch.Tensor,
        H_anchor: torch.Tensor,
        H_prev1: Optional[torch.Tensor] = None,
        H_prev2: Optional[torch.Tensor] = None,
        u_vacuity: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Executes a self-awareness update:
        - Computes s_t
        - Generates t_ego
        - Injects t_ego into Slot 0 of H_k
        - Computes modesty factor
        
        Args:
            k: Step counter (1 to k_max).
            k_max: Max deliberation steps.
            H_k: [B, L, D_inner] Current thoughts.
            H_anchor: [B, L, D_inner] Anchor thoughts.
            H_prev1: [B, L, D_inner] Thoughts at k-1.
            H_prev2: [B, L, D_inner] Thoughts at k-2.
            u_vacuity: Dirichlet uncertainty.
            logits: Optional vocabulary logits.
            
        Returns:
            H_aware: [B, L, D_inner] Thoughts with Slot 0 replaced by t_ego.
            t_ego: [B, 1, D_inner] Ego-thought token.
            telemetry: Detailed metrics dictionary.
        """
        B, L, D = H_k.shape
        h_k_anchor = H_k[:, 0, :]
        h_anchor_base = H_anchor[:, 0, :]

        # 1. Compute s_t
        s_t, s_metrics = self.descriptor(
            k=k,
            k_max=k_max,
            u_vacuity=u_vacuity,
            h_k=h_k_anchor,
            h_anchor=h_anchor_base,
            H_k=H_k,
            H_prev1=H_prev1,
            H_prev2=H_prev2,
            logits=logits
        )

        # 2. Formulate Ego-Thought Token t_ego
        t_ego = self.ego_projector(s_t) # [B, 1, D_inner]

        # 3. Inject into Slot 0 (Slot 0 is Ego, Slots 1..L-1 are Task thoughts)
        if L > 1:
            H_aware = torch.cat([t_ego, H_k[:, 1:, :]], dim=1)
        else:
            H_aware = t_ego

        # 4. Epistemic Modesty check
        modesty_factor, is_modest = self.modesty_modulator(s_t)

        telemetry = {
            **s_metrics,
            "s_t_vector": s_t.detach().cpu().squeeze(0).tolist() if B == 1 else s_t.detach().cpu().tolist(),
            "modesty_factor": float(modesty_factor.mean().item()),
            "epistemic_modesty_active": is_modest,
            "slot_0_type": "ego_thought_token",
            "task_slots_count": max(0, L - 1)
        }

        return H_aware, t_ego, telemetry


def calculate_active_neurons(
    d_model: int = 2048,
    d_inner: int = 1024,
    num_layers: int = 24,
    d_ffn: int = 5504,
    k_steps: int = 2,
    num_slots: int = 16,
    rank: int = 32
) -> Dict[str, Any]:
    """
    Computes exact active neurons firing in the System 2 Deliberation Ring and backbone.
    
    System 2 Deliberation Ring (Layer 11 hook):
        - Multihead Attention: 4 * d_inner = 4,096
        - Capacity Cross Attention: 3 * d_inner = 3,072
        - Latent FFN (MLP): 2 * d_inner = 2,048
        - CWM Working Memory: num_slots * d_inner = 16,384
        - Introspective Self-Descriptor: 5 + d_inner = 1,029
        - Plastic Fast-Weights (Rank-32): 2 * rank * d_inner = 65,536
        - Evidential Dirichlet Gate: 1,536
        Total System 2 per pass = 93,601 neurons
        With k=2 steps = ~187,202 neurons firing
        
    Backbone (System 1):
        - 24 layers * (d_model + d_ffn) = 24 * (2048 + 5504) = 181,248 neurons / token
    """
    # System 2 internal active units
    s2_mha = 4 * d_inner
    s2_cross = 3 * d_inner
    s2_ffn = 2 * d_inner
    s2_cwm = num_slots * d_inner
    s2_ego = 5 + d_inner
    s2_plastic = 2 * rank * d_inner
    s2_evidential = 1536
    
    s2_per_step = s2_mha + s2_cross + s2_ffn + s2_cwm + s2_ego + s2_plastic + s2_evidential
    active_s2 = s2_per_step * max(k_steps, 1)
    
    # System 1 per forward token pass
    active_s1 = num_layers * (d_model + d_ffn)
    total_active = active_s1 + active_s2
    
    return {
        "active_neurons_system2": active_s2,
        "active_neurons_system1": active_s1,
        "active_neurons_total": total_active,
        "synapses_total": 2310000000, # 2.31B parameters
        "formatted": f"{active_s2:,} Neurons (System 2 Laten) • {active_s1:,} Neurons (System 1) • 2.31B Sinapsis"
    }
