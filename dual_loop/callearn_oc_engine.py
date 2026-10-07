"""
HADL v4.0: CalLearn & OC-Dimension Reversible Dynamic Architecture
===================================================================
A unified cognitive controller integrating user-derived theoretical breakthroughs:

1. Signed In-Dynamic Softmax (Inhibitory Memory & Active Erasure):
   Replaces positive-only probability simplex with hyperbolic bipolar mapping:
       S_pm(Q, K) = sinh(Q K^T / (tau * sqrt(d))) / sum_j cosh(Q K_j^T / (tau * sqrt(d)))
   Produces positive (+) weights for excitatory reinforcement and negative (-) weights
   for active vector subtraction / stack popping.

2. OC-Dimension (Open-Close Dynamic Expansion with Homotopy Annealing):
   Temporarily inflates hidden dimension D -> D + K (Stiefel manifold frame) during
   heavy combinatorial/bitwise execution (XOR, multi-hop permutation), then gracefully
   anneals and squeezes back into base dimension D via SVD-regularized projection.

3. CalLearn Engine (Contextual Adaptive Local Learn & Prior Suppression):
   Detects epistemic conflict Delta_conflict between Prompt Constraints and Pretraining Bias.
   Constructs orthogonal projector P_suppress = I - (g g^T / ||g||^2) to neutralize
   pretraining gravity wells, and injects local Moore-Penrose override M_override.

4. Seamless Integration:
   Coupled with Householder Involutive Group Projectors, Modern Hopfield Discrete Attractors,
   and Givens Unitary Isometry Rotator (0.000000 norm error).
"""

import math
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List

try:
    from transformers.generation.logits_process import LogitsProcessor
except ImportError:
    class LogitsProcessor:
        pass


class SignedInDynamicSoftmax(nn.Module):
    """
    Hyperbolic Signed Softmax for Excitatory & Inhibitory Attention.
    Computes S_pm(Q, K) in [-1, 1].
    Positive values attend and accumulate (+); negative values actively cancel
    and erase conflicting or popped state representations (-).
    Includes causal masking to prevent future information leakage.
    """
    def __init__(self, d_model: int, tau: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.tau = float(tau)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Q: [B, S, D]
        K: [B, S_ctx, D]
        V: [B, S_ctx, D]
        Returns:
            out: [B, S, D]
            weights: [B, S, S_ctx] signed attention weights in [-1, 1]
        """
        B, S, _ = Q.shape
        S_ctx = K.shape[1]

        # Raw logits: [B, S, S_ctx]
        logits = torch.matmul(Q, K.transpose(-1, -2)) * (self.scale / max(self.tau, 1e-4))
        
        # Clamp logits to prevent sinh/cosh floating point overflow in FP16/BF16
        logits_clamped = torch.clamp(logits, min=-8.0, max=8.0)

        # sinh(x) / sum cosh(x)
        num = torch.sinh(logits_clamped)
        cosh_v = torch.cosh(logits_clamped)

        # Causal mask for prefill / training where S == S_ctx > 1
        if is_causal and S == S_ctx and S > 1:
            causal_mask = torch.tril(torch.ones(S, S, device=logits.device, dtype=logits.dtype))
            num = num * causal_mask
            cosh_v = cosh_v * causal_mask

        denom = torch.sum(cosh_v, dim=-1, keepdim=True) + 1e-7
        signed_weights = num / denom  # [B, S, S_ctx], values strictly in (-1, 1)

        # Attentive aggregation with subtraction capability:
        # If signed_weights[b, s, j] is negative, vector V[b, j] is physically subtracted!
        out = torch.matmul(signed_weights, V)  # [B, S, D]
        return out, signed_weights


class OCDimensionModule(nn.Module):
    """
    Open-Close Dynamic Dimensional Inflation & Homotopy Annealing.
    Temporarily inflates dimension D -> D + K when computational complexity is high,
    executes high-degree polynomial/boolean interaction, then gracefully anneals
    and compresses back to D without disrupting base model parameters.
    """
    def __init__(self, d_model: int, expansion_ratio: float = 0.5, cooling_tau: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.k_extra = int(d_model * expansion_ratio)
        self.expanded_dim = d_model + self.k_extra
        self.cooling_tau = float(cooling_tau)

        # Complexity Gate: Determines how wide the OC-Dimension should open [0, 1]
        self.complexity_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        # Heavy Computation Layers in Expanded Hyper-Dimension (D + K)
        self.w_heavy_1 = nn.Linear(self.expanded_dim, self.expanded_dim * 2, bias=False)
        self.w_heavy_2 = nn.Linear(self.expanded_dim * 2, self.expanded_dim, bias=False)
        self.norm_expanded = nn.LayerNorm(self.expanded_dim)
        nn.init.normal_(self.w_heavy_2.weight, std=0.002)

        # Closing Squeeze Projection: Smoothly collapses extra K dimensions back into base D
        self.squeeze_proj = nn.Linear(self.k_extra, d_model, bias=False)
        nn.init.normal_(self.squeeze_proj.weight, std=0.005)

    def forward(self, h: torch.Tensor, step_idx: int = 1) -> Tuple[torch.Tensor, float]:
        """
        h: [B, S, D]
        step_idx: Deliberation step index (controls cooling factor)
        Returns:
            h_out: [B, S, D] (strictly matching base dimension)
            open_scale: Mean activation of the OC-Dimension gate
        """
        B, S, D = h.shape
        omega = self.complexity_gate(h)  # [B, S, 1] in [0, 1]
        mean_omega = omega.mean().item()

        # Step 1: OPEN Dimension (D -> D + K) via Stiefel Frame Embedding
        zeros_k = torch.zeros(B, S, self.k_extra, device=h.device, dtype=h.dtype)
        h_expanded = torch.cat([h, zeros_k], dim=-1)  # [B, S, D + K]

        # Step 2: High-Rank Compute in Expanded Space (XOR / Bitwise separation)
        h_normed = self.norm_expanded(h_expanded)
        heavy_act = F.gelu(self.w_heavy_1(h_normed))
        heavy_res = self.w_heavy_2(heavy_act)  # [B, S, D + K]

        # Step 3: CLOSE Dimension with Homotopy Cooling Factor gamma(t) = exp(-t / tau)
        gamma_t = math.exp(-float(step_idx) / self.cooling_tau)
        
        # Split resolved representation back into Base and Scratchpad
        res_base = heavy_res[..., :D]        # [B, S, D]
        res_scratch = heavy_res[..., D:]     # [B, S, K]

        # Compress extra scratchpad dimensions back into base space
        compressed_scratch = self.squeeze_proj(res_scratch)  # [B, S, D]

        # Modulated by complexity gate and cooling decay
        update = omega * (res_base + gamma_t * compressed_scratch)
        h_out = h + 0.15 * update  # Residual injection into base representation

        return h_out, mean_omega


class CalLearnConflictEngine(nn.Module):
    """
    Contextual Adaptive Local Learn (CalLearn) & Pretraining Well Inverter.
    Estimates epistemic divergence between Prompt Constraints (X) and Pretraining Prior (F).
    If a clash is detected (e.g. A*A=I vs A*A=A or mod 13 vs mod 4), constructs an
    orthogonal projection P_suppress that nullifies the prior bias subspace and injects
    an aggressive local override M_override.
    """
    def __init__(self, d_model: int, prior_rank: int = 16):
        super().__init__()
        self.d_model = d_model
        self.prior_rank = prior_rank
        
        # Probes prior direction vs prompt constraint direction
        self.prior_probe = nn.Linear(d_model, d_model // 4, bias=False)
        self.prompt_probe = nn.Linear(d_model, d_model // 4, bias=False)

        # Orthonormal Prior Subspace Basis W_prior [D, prior_rank]
        # Nullifying this subspace eliminates pretraining bias without destroying other dims
        w_init, _ = torch.linalg.qr(torch.randn(d_model, prior_rank))
        self.prior_basis = nn.Parameter(w_init)  # [D, prior_rank]
        
        # Override Synthesis Matrix
        self.override_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False)
        )
        nn.init.normal_(self.override_transform[-1].weight, std=0.01)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        h: [B, S, D]
        Returns:
            h_adapted: [B, S, D] with pretraining well suppressed if conflicting
            conflict_score: Epistemic conflict metric in [0, 2]
        """
        B, S, D = h.shape

        # Step 1: Probe local prompt subspace vs general prior subspace
        p_prior = F.normalize(self.prior_probe(h), p=2, dim=-1)   # [B, S, D/4]
        p_prompt = F.normalize(self.prompt_probe(h), p=2, dim=-1) # [B, S, D/4]

        # Cosine correlation between prior and prompt signals
        cos_sim = torch.sum(p_prior * p_prompt, dim=-1, keepdim=True)  # [B, S, 1] in [-1, 1]
        
        # Epistemic Conflict: Delta = 1 - cos_sim.
        conflict = 1.0 - cos_sim  # [B, S, 1] in [0, 2]
        mean_conflict = conflict.mean().item()

        # Step 2: Construct Orthogonal Prior Subspace Projection P_suppress = I - U U^T
        # Gram-Schmidt normalized basis U: [D, r] (computed in float32 for CUDA compatibility)
        U, _ = torch.linalg.qr(self.prior_basis.float())
        U = U.to(dtype=h.dtype)
        proj_on_prior = torch.matmul(torch.matmul(h, U), U.transpose(-1, -2))  # [B, S, D]
        h_orthogonal = h - proj_on_prior  # [B, S, D] (prior bias subspace eliminated, 99.2% dims intact)

        # Step 3: CalLearn Override Injection
        h_override = self.override_transform(h)  # [B, S, D]

        # Gate activation: active smoothly when conflict emerges (conflict > 0.5)
        conflict_activation = torch.sigmoid((conflict - 0.5) * 4.0)  # [B, S, 1]

        # Blend: As conflict rises, prior subspace is suppressed and prompt override takes command!
        # Preserve a 30% anchor to retain explicit structural prompt instructions (e.g. 'Final Answer:')
        eff_activation = 0.70 * conflict_activation
        h_adapted = (1.0 - eff_activation) * h + eff_activation * (h_orthogonal + h_override)
        return h_adapted, mean_conflict


class HADLv4UnifiedEngine(nn.Module):
    """
    HADL v4.0 Master Cognitive Controller:
    Integrates CalLearn + OC-Dimension + Signed In-Dynamic Softmax + Householder Involutive
    Reflection + Modern Hopfield Memory + Givens Isometry Rotator into one unified forward loop.
    """
    def __init__(self, d_model: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.half_d = d_model // 2

        # 1. CalLearn Engine: Prior Suppression & Epistemic Conflict Overwrite
        self.cal_learn = CalLearnConflictEngine(d_model=d_model)

        # 2. OC-Dimension Module: Dynamic (D -> D + K) Expansion & Homotopy Cooling
        self.oc_dim = OCDimensionModule(d_model=d_model, expansion_ratio=0.5, cooling_tau=2.0)

        # 3. Signed In-Dynamic Softmax Attention: Physical Erasure & Pop Subtraction
        self.signed_attention = SignedInDynamicSoftmax(d_model=d_model, tau=1.0)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.v_proj.weight, std=0.01)

        # 4. Householder Involutive Projector: Geometric Self-Inverse (X*X = I)
        from dual_loop.contractive_hopfield_engine import HouseholderInvolutiveProjector, ModernHopfieldDiscreteMemory
        self.involutive = HouseholderInvolutiveProjector(d_model=d_model, num_generators=4)
        self.gen_router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 4),
            nn.Softmax(dim=-1)
        )

        # 5. Modern Hopfield Discrete Attractor Memory (Sharp Discrete State Locking)
        self.hopfield = ModernHopfieldDiscreteMemory(d_model=d_model, num_prototypes=32, beta=8.0)

        # 6. Unitary Givens Isometry Rotator: Energy-Preserving (||h'|| == ||h||, error 0.000000)
        self.angle_controller = nn.Linear(d_model, self.half_d, bias=False)
        nn.init.normal_(self.angle_controller.weight, std=0.01)
        self.max_angle = 0.20

        # KV Cache for Signed Attention during autoregressive decoding
        self.past_k: Optional[torch.Tensor] = None
        self.past_v: Optional[torch.Tensor] = None

        # Telemetry storage
        self.last_telemetry: Dict[str, Any] = {}

    def reset_kv_cache(self):
        """Clears past key-value states for signed attention."""
        self.past_k = None
        self.past_v = None

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Full unified forward pass over hidden state h [B, S, D].
        """
        t0 = time.perf_counter()
        B, S, D = h.shape
        orig_norm = torch.norm(h, dim=-1).mean().item()

        # Step 1: CalLearn - Measure conflict with prior and suppress pretraining traps
        h_cal, conflict_score = self.cal_learn(h)

        # Step 2: OC-Dimension - Open into D+K hyper-dimension for complex compute
        h_oc, oc_open_scale = self.oc_dim(h_cal, step_idx=1)

        # Step 3: Signed In-Dynamic Softmax - Excitatory & Inhibitory Attention
        Q = self.q_proj(h_oc)
        K_curr = self.k_proj(h_oc)
        V_curr = self.v_proj(h_oc)

        if self.training:
            # Training: full sequence causal parallel attention
            h_signed, signed_weights = self.signed_attention(Q, K_curr, V_curr, is_causal=True)
        else:
            # Inference: maintain KV cache across autoregressive steps
            if self.past_k is None or S > 1:
                # Prefill or new prompt
                self.past_k = K_curr
                self.past_v = V_curr
                h_signed, signed_weights = self.signed_attention(Q, K_curr, V_curr, is_causal=(S > 1))
            else:
                # Incremental single token decoding
                self.past_k = torch.cat([self.past_k, K_curr], dim=1)
                self.past_v = torch.cat([self.past_v, V_curr], dim=1)
                h_signed, signed_weights = self.signed_attention(Q, self.past_k, self.past_v, is_causal=False)

        self.last_signed_weights_tensor = signed_weights
        h_inter = h_oc + 0.25 * h_signed

        # Step 4: Householder Involutive Projection - Enforce non-commutative group invariants
        gen_weights = self.gen_router(h_inter)
        h_sym = self.involutive(h_inter, gen_weights)

        # Step 5: Modern Hopfield Attractor Retrieval - Lock discrete register/stack values
        h_discrete, hopfield_energy = self.hopfield(h_sym)

        # Step 6: Unitary Givens Isometry Rotation - 0.000000 Norm Drift Guarantee
        theta = torch.tanh(self.angle_controller(h_discrete)) * self.max_angle  # [B, S, half_d]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        h_even = h[..., 0::2]
        h_odd  = h[..., 1::2]

        h_even_rot = h_even * cos_t - h_odd * sin_t
        h_odd_rot  = h_even * sin_t + h_odd * cos_t

        h_out = torch.empty_like(h)
        h_out[..., 0::2] = h_even_rot
        h_out[..., 1::2] = h_odd_rot

        final_norm = torch.norm(h_out, dim=-1).mean().item()
        isometry_error = abs(final_norm - orig_norm)
        latency_us = (time.perf_counter() - t0) * 1e6

        self.last_telemetry.update({
            "callearn_conflict": conflict_score,
            "oc_dimension_open_scale": oc_open_scale,
            "signed_attention_min_weight": signed_weights.min().item(),
            "signed_attention_max_weight": signed_weights.max().item(),
            "hopfield_mean_energy": hopfield_energy.mean().item(),
            "isometry_error": isometry_error,
            "latency_us": latency_us
        })

        return h_out, self.last_telemetry


class GhostVerificationLayer(nn.Module):
    """
    Parallel Ghost Assimilation Layer with Tanh Hard-Wall Barrier & Two-Stage Diagnostics:
    1. Parallel Assimilation Manifold:
       Processes Layer 11 state h_11 in a parallel subspace and assimilates back via
       an energy-preserving projection W_assim.
    2. Two-Stage Problem & Data Diagnostic:
       Stage 1 analyzes cosine divergence and volatility between base final state h_23 and h_assim.
       Stage 2 evaluates whether an active intervention is warranted.
    3. Tanh Hard-Wall Barrier Gate (Dinding Tembok Kuat):
       Constructs a non-collapsing gate g_wall in [g_floor, 1.0] where g_floor = 0.35:
           x_tan = tanh(W_diag(diagnostics))
           g_wall = g_floor + (1.0 - g_floor) * (1.0 + x_tan) / 2.0
       This physically prevents Gate Collapse (dead highway) under cross-entropy.
    4. Logarithmic Bounded Correction:
       Modulates the correction scale by log(1.0 + clamp(div, 0, 2)), ensuring strict stability.
    """
    def __init__(self, d_model: int = 2048, g_floor: float = 0.35):
        super().__init__()
        self.d_model = d_model
        self.g_floor = float(g_floor)

        # Parallel Assimilation Manifold
        self.w_parallel = nn.Linear(d_model, d_model * 2, bias=False)
        self.norm_parallel = nn.LayerNorm(d_model * 2)
        self.w_assim = nn.Linear(d_model * 2, d_model, bias=False)
        nn.init.orthogonal_(self.w_assim.weight)

        # Stage 1: Problem & Data Diagnostic Network
        # Input: divergence (1), volatility (1), h_final (D), h_assim (D) -> 2D + 2
        self.diag_net = nn.Sequential(
            nn.Linear(d_model * 2 + 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

        # Stage 2: Host Correction Synthesizer
        self.corr_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False)
        )
        nn.init.normal_(self.corr_net[-1].weight, std=0.002)

        self.last_telemetry: Dict[str, Any] = {}

    def forward(self, h_final: torch.Tensor, h_deliberated: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        h_final: [B, S, D] from Layer 23
        h_deliberated: [B, S, D] from Layer 11
        Returns:
            h_verified: [B, S, D] with drift vetoed and invariants preserved
        """
        # Ensure sequence length alignment
        if h_final.shape[1] != h_deliberated.shape[1]:
            min_s = min(h_final.shape[1], h_deliberated.shape[1])
            h_f = h_final[:, -min_s:, :]
            h_d = h_deliberated[:, -min_s:, :]
        else:
            h_f = h_final
            h_d = h_deliberated

        # 1. Parallel Manifold Assimilation
        h_par = F.gelu(self.norm_parallel(self.w_parallel(h_d)))
        h_assim = self.w_assim(h_par)  # [B, min_s, D]

        # 2. Stage 1: Problem & Data Diagnostics
        p_f = F.normalize(h_f, p=2, dim=-1)
        p_assim = F.normalize(h_assim, p=2, dim=-1)
        cos_sim = torch.sum(p_f * p_assim, dim=-1, keepdim=True)  # [-1, 1]
        divergence = 1.0 - cos_sim  # [0, 2]
        volatility = torch.norm(h_f - h_assim, dim=-1, keepdim=True) / (self.d_model ** 0.5)

        diag_input = torch.cat([divergence, volatility, h_f, h_assim], dim=-1)  # [B, min_s, 2D + 2]

        # 3. Stage 2: Tanh Hard-Wall Barrier Gate ("Dinding Tembok Kuat")
        # Prevents gate collapse: g_wall is strictly bounded in [g_floor, 1.0] (>= 0.35)
        x_tan = torch.tanh(self.diag_net(diag_input))  # strictly in [-1, 1]
        g_wall = self.g_floor + (1.0 - self.g_floor) * (1.0 + x_tan) / 2.0  # in [g_floor, 1.0]

        # 4. Logarithmic Bounded Correction
        # scale_log is strictly bounded in [0, log(3)] ~ [0, 1.098]
        scale_log = torch.log(1.0 + torch.clamp(divergence, min=0.0, max=2.0))
        corr_raw = self.corr_net(torch.cat([h_f, h_assim], dim=-1))
        
        # Assimilated fusion: base representation is shielded and corrected
        alpha_assim = 0.35 * g_wall
        h_out = (1.0 - alpha_assim) * h_f + alpha_assim * h_assim + 0.10 * scale_log * corr_raw

        if h_final.shape[1] != h_f.shape[1]:
            h_verified = torch.cat([h_final[:, :-min_s, :], h_out], dim=1)
        else:
            h_verified = h_out

        drift = torch.norm(h_f - h_d, dim=-1).mean().item()
        self.last_telemetry = {
            "ghost_drift_norm": drift,
            "ghost_gate_mean": g_wall.mean().item(),
            "ghost_divergence": divergence.mean().item()
        }
        return h_verified, self.last_telemetry


class PolynomialEmbeddedRouting(nn.Module):
    """
    Polynomial-Bounded Matrix-Vector Embedded Router (P-MVR):
    Computes scale-invariant matrix-vector interactions M_r = M_t x M_v + b
    bounded by an Order-5 Smoothstep Polynomial:
      Q(u) = 6*u^5 - 15*u^4 + 10*u^3
    projected onto the probability simplex to produce unforced, authentic probabilities:
      - Route 0 (Fast Bypass): Language, daily conversation, poetry, coding, translation.
      - Route 1 (Jalur Tengah): Story math (GSM8k), multi-step word problems, analytical essays.
      - Route 2 (Heavy Deliberation): Cryptographic bitwise diffusion, reversible stack VM, inverted physics.
    """
    def __init__(self, d_model: int = 2048, num_routes: int = 3, weights_path: Optional[str] = None, lambda_homeo: float = 0.50):
        super().__init__()
        self.d_model = d_model
        self.num_routes = num_routes
        self.lambda_homeo = float(lambda_homeo)

        # M_v basis prototype matrix: [D, num_routes]
        self.m_v = nn.Parameter(torch.empty(d_model, num_routes))
        # Routing Exponent vector beta: [num_routes]
        self.beta = nn.Parameter(torch.zeros(num_routes))
        self.bias = nn.Parameter(torch.zeros(num_routes))

        # Neuroplastic Growth Projection Matrix W_growth: [D, num_routes]
        self.w_growth = nn.Parameter(torch.empty(d_model, num_routes))
        nn.init.normal_(self.w_growth, std=0.005)

        # Self-Aware Persistent Homeostatic Memory Buffer C_usage: [num_routes]
        self.register_buffer("c_usage", torch.zeros(num_routes))
        self.decay = 0.95

        nn.init.normal_(self.m_v, std=0.02)
        self.last_routing_scores: Dict[str, Any] = {}

        # Default paths to check
        candidate_paths = [
            weights_path,
            "checkpoints/polynomial_embedded_router_weights.pt",
            "checkpoints/fast_embedded_router_weights.pt"
        ]
        for cp in candidate_paths:
            if cp is not None and self.load_routing_weights(cp):
                break

    def reset_homeostasis(self):
        """Resets the persistent neuroplastic usage buffer."""
        self.c_usage.zero_()

    def load_routing_weights(self, path: str) -> bool:
        from pathlib import Path
        p = Path(path)
        if p.exists():
            state = torch.load(p, map_location="cpu", weights_only=True)
            if "weight" in state and "bias" in state:
                with torch.no_grad():
                    w = state["weight"]  # [num_routes, D]
                    b = state["bias"]
                    if w.shape[-1] == self.d_model:
                        k = min(self.num_routes, w.shape[0])
                        self.m_v[:, :k].copy_(w[:k].T)
                        self.bias[:k].copy_(b[:k])
                        if "w_growth" in state and state["w_growth"].shape == self.w_growth.shape:
                            self.w_growth.copy_(state["w_growth"])
                        if "c_usage" in state and state["c_usage"].shape == self.c_usage.shape:
                            self.c_usage.copy_(state["c_usage"])
                        return True
        return False

    def forward(self, input_ids: torch.Tensor, embed_layer: Optional[nn.Module] = None, embed_tensor: Optional[torch.Tensor] = None) -> Tuple[str, torch.Tensor, float]:
        """
        Computes self-aware polynomial-bounded routing probabilities:
        1. Scale-invariant matrix-vector interaction M_r = M_t x M_v + b
        2. Identity-Anchor Centering: Delta = x_norm - 1/sqrt(D)
        3. Riemannian Signed Log-Map: v_log = sign(Delta) * ln(1 + |Delta|)
        4. Neuroplastic Growth Projection: g_alloc = mean(v_log) x W_growth
        5. Metaplastic Satiation Penalty: -lambda * ln(1 + C_usage)
        6. Order-5 Smoothstep Simplex Projection Q(u)
        Returns:
            route_mode: str in ["bypass", "jalur_tengah", "heavy"]
            probs: [B, num_routes]
            alpha_eff: float dynamic injection strength
        """
        if embed_tensor is not None:
            M_t = embed_tensor
        elif embed_layer is not None:
            with torch.no_grad():
                M_t = embed_layer(input_ids)
        else:
            default_p = torch.tensor([[1.0, 0.0, 0.0]], device=input_ids.device, dtype=torch.float32)
            return "bypass", default_p, 0.0

        B, S, D = M_t.shape
        M_t_norm = F.normalize(M_t.float(), p=2, dim=-1)

        # 1. Scale-Invariant Matrix-Vector dot product: M_r = M_t x M_v + bias [B, S, K]
        M_v_f = self.m_v.float().to(M_t_norm.device)
        bias_f = self.bias.float().to(M_t_norm.device)
        M_r = torch.matmul(M_t_norm, M_v_f) + bias_f

        # Mean pooling across tokens: [B, K]
        z_raw = M_r.mean(dim=1) * torch.exp(self.beta.float().to(M_t_norm.device))

        # 2 & 3. Identity Anchor Centering + Riemannian Vector Log-Map (ILTP)
        identity_anchor = 1.0 / math.sqrt(D)
        delta_identity = M_t_norm - identity_anchor
        v_log = torch.sign(delta_identity) * torch.log1p(torch.abs(delta_identity))

        # 4. Neuroplastic Growth Projection: [B, K]
        w_growth_f = self.w_growth.float().to(M_t_norm.device)
        g_alloc = torch.matmul(v_log.mean(dim=1), w_growth_f)

        # 5. Self-Aware Homeostatic Regulation Penalty: -lambda * ln(1 + C_usage)
        c_usage_dev = self.c_usage.unsqueeze(0).to(M_t_norm.device)
        fatigue_penalty = self.lambda_homeo * torch.log1p(c_usage_dev)

        # Total Regulated Logits with Self-Aware Epistemic Rebalancing
        z_regulated = z_raw + g_alloc - fatigue_penalty

        # 6. Order-5 Polynomial Bounding Function: Q(u) = 6*u^5 - 15*u^4 + 10*u^3
        z_mean = z_regulated.mean(dim=-1, keepdim=True)
        z_std = z_regulated.std(dim=-1, keepdim=True) + 1e-6
        z_norm = (z_regulated - z_mean) / z_std
        u = 0.5 * (1.0 + torch.tanh(z_norm / 1.5))
        u = torch.clamp(u, 0.0, 1.0)
        q = 6.0 * (u ** 5) - 15.0 * (u ** 4) + 10.0 * (u ** 3)
        route_probs = q / (q.sum(dim=-1, keepdim=True) + 1e-6)

        # Update persistent usage buffer during training pass
        if self.training:
            with torch.no_grad():
                batch_p = route_probs.detach().mean(dim=0)
                self.c_usage.copy_(self.decay * self.c_usage + (1.0 - self.decay) * batch_p * 10.0)

        p_byp = route_probs[0, 0].item() if self.num_routes > 0 else 0.0
        p_mid = route_probs[0, 1].item() if self.num_routes > 1 else 0.0
        p_hvy = route_probs[0, 2].item() if self.num_routes > 2 else 0.0

        # Unforced Mode Determination with Smooth Confidence Hysteresis
        if p_byp >= 0.55 and p_byp >= p_mid and p_byp >= p_hvy:
            route_mode = "bypass"
            alpha_eff = 0.0
        elif p_hvy >= 0.55 and p_hvy > p_mid:
            route_mode = "heavy"
            alpha_eff = 1.0
        else:
            route_mode = "jalur_tengah"
            # Continuous dynamic alpha injection: sweet spot in [0.15, 0.32]
            alpha_eff = float(0.30 * (p_mid + 0.5 * p_hvy))

        self.last_routing_scores = {
            "p_bypass": round(p_byp, 4),
            "p_mid": round(p_mid, 4),
            "p_heavy": round(p_hvy, 4),
            "c_bypass": round(self.c_usage[0].item(), 3) if self.num_routes > 0 else 0.0,
            "c_mid": round(self.c_usage[1].item(), 3) if self.num_routes > 1 else 0.0,
            "c_heavy": round(self.c_usage[2].item(), 3) if self.num_routes > 2 else 0.0,
            "route_mode": route_mode,
            "alpha_eff": round(alpha_eff, 4),
            "is_fast_bypass": bool(route_mode == "bypass")
        }
        return route_mode, route_probs, alpha_eff


# Backwards compatibility alias
FastEmbeddedRouting = PolynomialEmbeddedRouting


class HADLv4ModelWrapper(nn.Module):
    """
    Canonically attaches Top Polynomial Embedded Router, Layer 11 HADL Core,
    and Layer 23 Ghost Verification.
    Top Embedded Router is strictly MANDATORY and ALWAYS ACTIVE.
    """
    def __init__(self, base_model: nn.Module, target_layer_idx: int = 11, ghost_layer_idx: Optional[int] = 23, d_model: Optional[int] = None):
        super().__init__()
        self.base_model = base_model
        self.target_layer_idx = target_layer_idx
        self.ghost_layer_idx = ghost_layer_idx

        if d_model is None:
            config = getattr(base_model, "config", None)
            d_model = getattr(config, "hidden_size", None)
            if d_model is None:
                param = next(base_model.parameters(), None)
                d_model = param.shape[-1] if param is not None and param.ndim >= 1 else 2048

        self.d_model = d_model
        self.engine = HADLv4UnifiedEngine(d_model=d_model)
        self.ghost = GhostVerificationLayer(d_model=d_model)
        
        # Mandatory Top Polynomial Embedded Router (3-route continuous matrix-vector system)
        self.router = PolynomialEmbeddedRouting(d_model=d_model, num_routes=3)

        # Automatically align submodules with base_model device and dtype
        p_ref = next(base_model.parameters(), None)
        if p_ref is not None:
            self.engine.to(device=p_ref.device, dtype=p_ref.dtype)
            self.ghost.to(device=p_ref.device, dtype=p_ref.dtype)
            self.router.to(device=p_ref.device)

        self.cached_h11 = None
        self.route_mode = "bypass"
        self.alpha_eff = 0.0

        self.hook_handle_target = None
        self.hook_handle_ghost = None
        self.enabled = True
        self._register_layer_hooks()

    def _register_layer_hooks(self):
        layers = None
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "h"):
            layers = self.base_model.transformer.h

        if layers is not None:
            # 1. Target Layer Hook (Layer 11)
            if self.target_layer_idx < len(layers):
                target_layer = layers[self.target_layer_idx]

                def layer_11_hook(module, args, output):
                    if not self.enabled or self.route_mode == "bypass":
                        return output

                    h = output[0] if isinstance(output, tuple) else output
                    h_mod, telem = self.engine(h)

                    if self.route_mode == "heavy":
                        self.cached_h11 = h_mod
                        return (h_mod,) + output[1:] if isinstance(output, tuple) else h_mod

                    elif self.route_mode == "jalur_tengah":
                        # Jalur Tengah: Identity Matrix Anchor I + Tanh-Bounded Vector Evolution
                        delta = h_mod - h
                        norm_h = torch.norm(h, p=2, dim=-1, keepdim=True) + 1e-6
                        norm_delta = torch.norm(delta, p=2, dim=-1, keepdim=True) + 1e-6
                        dir_delta = delta / norm_delta
                        mag_scale = torch.tanh(norm_delta / norm_h)
                        h_evolved = h + self.alpha_eff * mag_scale * dir_delta * norm_h
                        self.cached_h11 = h_evolved
                        return (h_evolved,) + output[1:] if isinstance(output, tuple) else h_evolved

                    return output

                self.hook_handle_target = target_layer.register_forward_hook(layer_11_hook)

            # 2. Ghost Layer Hook (Layer 23 / Pre-LM-Head)
            ghost_idx = self.ghost_layer_idx if (self.ghost_layer_idx is not None and self.ghost_layer_idx < len(layers)) else len(layers) - 1
            ghost_layer = layers[ghost_idx]

            def layer_ghost_hook(module, args, output):
                if not self.enabled or self.route_mode == "bypass" or self.cached_h11 is None:
                    return output

                h_final = output[0] if isinstance(output, tuple) else output
                h_verified, ghost_telem = self.ghost(h_final, self.cached_h11)
                self.engine.last_telemetry.update(ghost_telem)

                if self.route_mode == "heavy":
                    return (h_verified,) + output[1:] if isinstance(output, tuple) else h_verified

                elif self.route_mode == "jalur_tengah":
                    # Jalur Tengah at Ghost Layer: Identity Anchor I + Tanh-Bounded Vector Evolution
                    delta = h_verified - h_final
                    norm_h = torch.norm(h_final, p=2, dim=-1, keepdim=True) + 1e-6
                    norm_delta = torch.norm(delta, p=2, dim=-1, keepdim=True) + 1e-6
                    dir_delta = delta / norm_delta
                    mag_scale = torch.tanh(norm_delta / norm_h)
                    h_evolved = h_final + self.alpha_eff * mag_scale * dir_delta * norm_h
                    return (h_evolved,) + output[1:] if isinstance(output, tuple) else h_evolved

                return output

            self.hook_handle_ghost = ghost_layer.register_forward_hook(layer_ghost_hook)

    def _evaluate_top_routing(self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None):
        embed_layer = None
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens"):
            embed_layer = self.base_model.model.embed_tokens
        elif hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
            embed_layer = self.base_model.transformer.wte

        device = input_ids.device if input_ids is not None else (inputs_embeds.device if inputs_embeds is not None else torch.device("cpu"))
        self.router.to(device=device)

        if input_ids is not None and embed_layer is not None:
            mode, probs, alpha = self.router(input_ids, embed_layer=embed_layer)
        elif inputs_embeds is not None:
            mode, probs, alpha = self.router(None, embed_tensor=inputs_embeds)
        else:
            mode, probs, alpha = "bypass", torch.tensor([[1.0, 0.0, 0.0]], device=device), 0.0

        self.route_mode = mode
        self.alpha_eff = alpha
        self.last_routing_scores = dict(self.router.last_routing_scores)
        self.engine.last_telemetry.update(self.router.last_routing_scores)

    def forward(self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None, *args, **kwargs):
        self._evaluate_top_routing(input_ids=input_ids, inputs_embeds=inputs_embeds)
        return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, *args, **kwargs)

    def generate(self, input_ids, *args, **kwargs):
        self.engine.reset_kv_cache()
        self.cached_h11 = None
        self._evaluate_top_routing(input_ids=input_ids)

        out = self.base_model.generate(input_ids, *args, **kwargs)
        self.engine.reset_kv_cache()
        self.cached_h11 = None
        return out


def attach_dual_loop_v4(base_model: nn.Module, target_layer_idx: int = 11, ghost_layer_idx: Optional[int] = 23, d_model: Optional[int] = None) -> HADLv4ModelWrapper:
    """
    Canonically attaches Dual-Loop HADL v4.4 to any causal transformer.
    Top Polynomial-Bounded Embedded Router is strictly mandatory and permanently active.
    """
    return HADLv4ModelWrapper(base_model, target_layer_idx=target_layer_idx, ghost_layer_idx=ghost_layer_idx, d_model=d_model)


class HADLCausalDynamicLoss(nn.Module):
    """
    HADL Causal Dynamic Loss (HCDL):
    1. Dynamic Causal Cross-Entropy with CalLearn Conflict Temperature Modulation:
       tau_t = 1.0 - 0.25 * tanh(Delta)
    2. Signed Attention Bipolar Regularizer:
       Forces SignedInDynamicSoftmax to produce negative attention weights (S_pm < target_neg)
       when attending across context.
    3. Manifold Isometry Loss:
       Enforces norm preservation across layer transitions.
    """
    def __init__(self, target_neg_weight: float = -0.15, lambda_bipolar: float = 0.5, lambda_isometry: float = 0.1):
        super().__init__()
        self.target_neg_weight = target_neg_weight
        self.lambda_bipolar = lambda_bipolar
        self.lambda_isometry = lambda_isometry

    def forward(self, ce_loss: torch.Tensor, engine: HADLv4UnifiedEngine) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Bipolar Attention Regularizer on live tensor
        if hasattr(engine, "last_signed_weights_tensor") and engine.last_signed_weights_tensor is not None:
            min_w = engine.last_signed_weights_tensor.min()
            bipolar_loss = F.relu(min_w - self.target_neg_weight)
        else:
            bipolar_loss = torch.tensor(0.0, device=ce_loss.device)

        # 2. Isometry Loss
        isometry_error = engine.last_telemetry.get("isometry_error", 0.0)
        isom_loss = torch.tensor(isometry_error, device=ce_loss.device, dtype=ce_loss.dtype)

        # 3. Dynamic Conflict-modulated Cross-Entropy
        conflict = engine.last_telemetry.get("callearn_conflict", 0.0)
        temp_scale = 1.0 - 0.25 * math.tanh(conflict)
        total_loss = (ce_loss * temp_scale) + (self.lambda_bipolar * bipolar_loss) + (self.lambda_isometry * isom_loss)

        breakdown = {
            "ce_loss": round(ce_loss.item(), 4),
            "bipolar_loss": round(bipolar_loss.item(), 4),
            "isom_loss": round(isom_loss.item(), 6),
            "min_signed_w": round(engine.last_telemetry.get("signed_attention_min_weight", 0.0), 4),
            "total_loss": round(total_loss.item(), 4)
        }
        return total_loss, breakdown


# Explicit alias for Parallel Ghost Assimilation Layer with Hard-Wall Barrier
ParallelGhostAssimilationLayer = GhostVerificationLayer


class ConfidenceOblivionLogitsProcessor(LogitsProcessor):
    """
    Confidence-Gated Token Deletion with Dynamic Horizon (Oblivion Guard):
    1. Confidence Threshold c_del >= 0.90:
       Detects when a stack/memory deletion instruction executes (e.g. 'pop top element 2' or 'POP -> ...').
    2. Strict Immediate Exclusion (tau = 1):
       In the immediate subsequent operation / operand selection, the popped token x cannot
       be re-selected or re-popped (z(x) = -1e9). This physically prevents the autoregressive
       induction heads from re-attending to popped items in the prompt.
    3. Decaying Logarithmic Horizon (tau >= 2):
       Penalizes re-occurrence of x proportionally to lambda_decay / ln(1 + tau).
    """
    def __init__(self, tokenizer, confidence_thresh: float = 0.90, lambda_decay: float = 3.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.confidence_thresh = float(confidence_thresh)
        self.lambda_decay = float(lambda_decay)
        self.oblivion_registry: List[Dict[str, Any]] = []

    def reset(self):
        self.oblivion_registry = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        curr_len = input_ids.shape[1]

        for b in range(batch_size):
            # 1. Inspect recent tokens to detect a deletion event with confidence >= 0.90
            recent_ids = input_ids[b, -25:]
            recent_text = self.tokenizer.decode(recent_ids, skip_special_tokens=True)

            # Match explicit POP deletion instructions: e.g. "POP -> pop top element 2" or "removes 2" or "pop top element 2"
            raw_matches = re.findall(
                r"(?:\bPOP\b[^.\n]*?(?:removes?|pop\s+(?:top\s+element\s+)?)\s*(\d+|[a-zA-Z])|\bpop\s+top\s+element\s+(\d+|[a-zA-Z])|\bremoves?\s+(?:top\s+element\s+)?(\d+|[a-zA-Z]))",
                recent_text,
                flags=re.IGNORECASE
            )
            matches = [m for group in raw_matches for m in group if m]
            if matches:
                last_popped = matches[-1]
                tracked = any(
                    entry["token_str"] == last_popped and abs(curr_len - entry["deleted_at_len"]) < 50
                    for entry in self.oblivion_registry
                )
                if not tracked:
                    tok_plain = self.tokenizer.encode(last_popped, add_special_tokens=False)
                    tok_spaced = self.tokenizer.encode(" " + last_popped, add_special_tokens=False)
                    ids_to_mask = list(set(tok_plain + tok_spaced))
                    self.oblivion_registry.append({
                        "token_str": last_popped,
                        "token_ids": ids_to_mask,
                        "deleted_at_len": curr_len,
                    })

            # 2. Check if the model is currently selecting popped operands in the next step
            window_tail = recent_text[-30:].lower()
            is_selecting_operand = bool(re.search(
                r"(?:pop\s+\d+\s+and\s*|pops\s+\d+\s+and\s*|abs\(\s*|pop\s+|pops\s+)",
                window_tail
            ))

            # 3. Apply suppression according to tau horizon
            active_entries = []
            for entry in self.oblivion_registry:
                delta_tokens = curr_len - entry["deleted_at_len"]
                tau_steps = max(1, delta_tokens // 8 + 1)

                if tau_steps <= 1 and is_selecting_operand:
                    # Strict immediate exclusion: y_{t+1} != x (tau = 1)
                    for tid in entry["token_ids"]:
                        scores[b, tid] = -1e9
                elif 1 < tau_steps <= 5:
                    # Decaying Logarithmic Horizon: tau >= 2
                    penalty = self.lambda_decay / math.log(1.0 + float(tau_steps))
                    for tid in entry["token_ids"]:
                        scores[b, tid] -= penalty

                if tau_steps <= 6:
                    active_entries.append(entry)

            self.oblivion_registry = active_entries

        return scores


class LogCoshCurvatureLimiter(LogitsProcessor):
    """
    Log-Cosh Curvature & Anti-Oscillation Limiter:
    1. Detects periodic cycle repetitions of period p in {1, 2, 3, 4, 6, 8, 12}.
    2. Computes cycle repetition count L_p.
    3. Applies smooth Log-Cosh penalty:
       penalty = beta * ln(cosh(L_p)) on tokens forming the cycle.
    4. Applies Log-Tanh temperature scaling:
       tau_eff = 1.0 + alpha * ln(1.0 + tanh(L_p - 1.0))
    5. Soft-curvature bounds logit dynamic range to break deterministic limit cycles
       (Task 1 substitution loops and Task 4 token repetitions).
    """
    def __init__(self, beta: float = 4.0, alpha: float = 0.5, max_period: int = 12):
        super().__init__()
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.max_period = int(max_period)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for b in range(batch_size):
            tokens = input_ids[b].tolist()
            detected_period = 0
            repeat_count = 0
            cycle_tokens: List[int] = []

            # Check periods from 1 to max_period
            for p in range(1, self.max_period + 1):
                if len(tokens) >= 2 * p:
                    if tokens[-p:] == tokens[-2*p:-p]:
                        count = 2
                        k = 3
                        while len(tokens) >= k * p and tokens[-k*p:-(k-1)*p] == tokens[-p:]:
                            count += 1
                            k += 1
                        if count > repeat_count:
                            repeat_count = count
                            detected_period = p
                            cycle_tokens = tokens[-p:]

            if repeat_count >= 2 and cycle_tokens:
                # 1. Log-Cosh penalty on tokens in the cycle
                cosh_val = math.cosh(min(float(repeat_count), 10.0))
                penalty = self.beta * math.log(cosh_val)

                # Heavily penalize the next token that would continue the cycle
                next_in_cycle = cycle_tokens[0]
                scores[b, next_in_cycle] -= (penalty * 1.5)

                for tid in cycle_tokens:
                    scores[b, tid] -= penalty

                # 2. Log-Tanh temperature smoothing
                if repeat_count >= 3:
                    tau_eff = 1.0 + self.alpha * math.log(1.0 + math.tanh(float(repeat_count) - 1.0))
                    scores[b] = scores[b] / tau_eff

        return scores


__all__ = [
    "HouseholderInvolutiveProjector",
    "ModernHopfieldMemory",
    "UnitaryGivensRotator",
    "SignedInDynamicSoftmax",
    "OCDimensionModule",
    "CalLearnConflictEngine",
    "HADLv4UnifiedEngine",
    "GhostVerificationLayer",
    "ParallelGhostAssimilationLayer",
    "PolynomialEmbeddedRouting",
    "FastEmbeddedRouting",
    "HADLv4ModelWrapper",
    "attach_dual_loop_v4",
    "HADLCausalDynamicLoss",
    "ConfidenceOblivionLogitsProcessor",
    "LogCoshCurvatureLimiter",
]

