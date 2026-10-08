"""
Living Cognitive Organ System (LCOS)
=====================================
Higher-Order Biological Cognitive Organs for HADL v4.5+ & Dual-Loop Architectures.

This module evolves the mechanical hydraulic foundation into an Organic Living Cognitive System:
1. DynamicAdaptiveConfusionSensor (DACS):
   Self-calibrating, multi-head drift-aware confusion organ. Replaces static scalar thresholds
   with an online running distribution (mean + k*std) tailored dynamically to any LLM brain backbone.
2. SemanticAffinityEnergyMatrix (SAEM):
   Ockham's Razor concept-slot binder with vertical energy hierarchy. High energy (top) = ill-fitting,
   low energy (bottom) = maximum parsimony and functional elegance.
3. HierarchicalPlanCache (HLPC) & LostPlanRegister:
   Dynamic Prefrontal planning organ with Context Engine gating (1-0), compact latent waypoints (<2KB VRAM),
   decaying LostPlan episodic staging, and goal cross-verification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List


# ==============================================================================
# 0. EXECUTIVE LOGIC ROUTER (TOP-LEVEL META-CONTROLLER)
# ==============================================================================

class ExecutiveLogicRouter(nn.Module):
    """
    Top-Level Cognitive Logic Router (Executive Meta-Controller).
    
    Acts as the primary decision-making brain center at the highest hierarchy level.
    Instead of hardcoding static parsimony or static expansion, the Router dynamically
    evaluates context and decides the operating regime:
    - Mode 0: Fast & Solid (compact, direct, parsimonious execution)
    - Mode 1: Balanced Reasoning (standard deliberate synthesis)
    - Mode 2: Long & Detailed (multi-module, expansive architecture, state-machine synthesis)
    
    Generates continuous modulation parameters:
    - parsimony_scale: in [0.10, 1.50] (modulates SAEM vertical penalty)
    - elaboration_scale: in [0.35, 1.80] (modulates ELIC imagination canvas capacity)
    - planning_urgency: in [0.10, 0.95] (modulates HLPC plan formulation hurdle)
    - delta_anchor: Structural namespace grounding tensor to preserve package directory hierarchy
    """
    def __init__(self, d_model: int = 2048):
        super().__init__()
        self.d_model = d_model
        
        # Dual-Axis Evaluator:
        # Axis 1: Structural Scope (0=Micro/Single-turn, 1=Macro/Multi-file)
        # Axis 2: Density Pressure (0=Expansive, 1=Dense/Brevity-constrained)
        self.w_scope = nn.Linear(d_model, 1, bias=True)
        self.w_density = nn.Linear(d_model, 1, bias=True)
        
        # Meta-Routing Network: evaluates coupled interaction (S x D)
        self.meta_net = nn.Sequential(
            nn.Linear(d_model, max(64, d_model // 4), bias=True),
            nn.LayerNorm(max(64, d_model // 4)),
            nn.GELU(),
            nn.Linear(max(64, d_model // 4), 4, bias=True)
            # [Fast & Direct, Balanced, Dense Structural Plan, Expansive Detailed]
        )
        
        # Structural Namespace Anchor Projector (helps ground package hierarchy)
        self.w_namespace_anchor = nn.Linear(d_model, d_model, bias=False)
        self.alpha_anchor = nn.Parameter(torch.tensor(0.04))
        
        # Initialize cleanly
        nn.init.orthogonal_(self.meta_net[0].weight, gain=0.1)
        nn.init.zeros_(self.meta_net[0].bias)
        nn.init.normal_(self.meta_net[3].weight, std=0.01)
        nn.init.constant_(self.meta_net[3].bias, 0.0)
        nn.init.normal_(self.w_scope.weight, std=0.01)
        nn.init.constant_(self.w_scope.bias, 0.0)
        nn.init.normal_(self.w_density.weight, std=0.01)
        nn.init.constant_(self.w_density.bias, 0.2)
        nn.init.normal_(self.w_namespace_anchor.weight, std=0.01)

    def forward(
        self,
        h: torch.Tensor,
        current_step_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Determines the dynamic cognitive regime and namespace anchors from input representations.
        """
        orig_ndim = h.ndim
        if orig_ndim == 3:
            h_mean = h.mean(dim=1)  # (B, D)
        else:
            h_mean = h
            
        B = h_mean.shape[0]
        scope_score = torch.sigmoid(self.w_scope(h_mean))      # 1 = Multi-file repo
        density_score = torch.sigmoid(self.w_density(h_mean))  # 1 = Concise/dense target
        
        logits = self.meta_net(h_mean)  # (B, 4)
        probs = F.softmax(logits, dim=-1)  # (B, 4)
        
        p_fast = probs[:, 0:1]        # Fast & Direct
        p_bal = probs[:, 1:2]         # Balanced
        p_dense_plan = probs[:, 2:3]  # Dense Structural Plan (Besar namun Singkat-Padat)
        p_expansive = probs[:, 3:4]   # Expansive Detailed
        
        # Coupled Cross-Talk Modulation (A <-> B):
        # Step-dependent density modulation: setup.py & __init__.py kept tight, core given space
        step_factor = 1.0
        if current_step_idx is not None:
            if current_step_idx == 0:
                step_factor = 1.25  # setup.py: tight metadata
            elif current_step_idx == 1:
                step_factor = 1.15  # __init__.py: exports
            else:
                step_factor = 0.85  # core module: allow implementation
                
        parsimony_scale = (p_fast * 1.50 + p_bal * 0.85 + p_dense_plan * 0.85 + p_expansive * 0.15).squeeze(-1) * step_factor
        elaboration_scale = (p_fast * 0.35 + p_bal * 1.00 + p_dense_plan * 1.10 + p_expansive * 1.80).squeeze(-1)
        planning_urgency = (p_fast * 0.10 + p_bal * 0.45 + p_dense_plan * 0.95 + p_expansive * 0.85).squeeze(-1)
        
        # Structural Namespace Anchor Delta (injects directory/package tree grounding)
        delta_anchor = torch.tanh(self.alpha_anchor) * self.w_namespace_anchor(h) * scope_score.unsqueeze(1)
        if orig_ndim == 2:
            delta_anchor = delta_anchor.squeeze(1)
            
        dominant_idx = torch.argmax(probs, dim=-1)[0].item()
        regime_names = ["Fast & Direct", "Balanced", "Dense Structural Plan", "Expansive Detailed"]
        dominant_mode = regime_names[dominant_idx]
        
        info = {
            "dominant_mode": dominant_mode,
            "scope_score": float(scope_score.mean().item()),
            "density_score": float(density_score.mean().item()),
            "parsimony_scale": float(parsimony_scale.mean().item()),
            "elaboration_scale": float(elaboration_scale.mean().item()),
            "planning_urgency": float(planning_urgency.mean().item()),
            "delta_anchor": delta_anchor,
            "probabilities": probs.detach().cpu()
        }
        return probs, info


# ==============================================================================
# 1. DYNAMIC ADAPTIVE CONFUSION SENSOR (DACS)
# ==============================================================================

class DynamicAdaptiveConfusionSensor(nn.Module):
    """
    Dynamic Adaptive Confusion Sensor (DACS).
    
    Autonomously calibrates cognitive ambiguity and confusion thresholds for any LLM
    backbone by tracking latent representation drift across attention heads and maintaining
    an online exponential moving distribution (mu, sigma) without hardcoded static thresholds.
    """
    def __init__(
        self,
        d_model: int = 2048,
        num_heads: int = 8,
        k_sigma: float = 0.50,
        ema_momentum: float = 0.05,
        min_tau: float = 0.30,
        max_tau: float = 0.85
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_sigma = float(k_sigma)
        self.ema_momentum = float(ema_momentum)
        self.min_tau = float(min_tau)
        self.max_tau = float(max_tau)
        
        # Multi-head drift projection
        self.head_dim = max(16, d_model // num_heads)
        self.w_head_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.w_confusion_head = nn.Linear(self.head_dim, 1, bias=True)
        
        # Semantic context ambiguity evaluator
        self.w_context_ambiguity = nn.Linear(d_model, 1, bias=True)
        
        # Running statistical state (persistent across generation tokens, not backpropped)
        self.register_buffer("running_mean", torch.tensor(0.50))
        self.register_buffer("running_var", torch.tensor(0.01))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_h_head", torch.zeros(1, num_heads, self.head_dim))
        
        # Initialize projections gently
        nn.init.orthogonal_(self.w_head_proj.weight)
        nn.init.normal_(self.w_confusion_head.weight, std=0.01)
        nn.init.constant_(self.w_confusion_head.bias, 0.0)
        nn.init.normal_(self.w_context_ambiguity.weight, std=0.01)
        nn.init.constant_(self.w_context_ambiguity.bias, 0.0)

    def reset_history(self):
        """Resets running state for a fresh model inference session."""
        self.running_mean.fill_(0.50)
        self.running_var.fill_(0.01)
        self.step_count.zero_()
        self.last_h_head.zero_()

    def get_dynamic_threshold(self) -> float:
        """Calculates current dynamic confusion threshold based on running distribution."""
        std = math.sqrt(max(1e-6, self.running_var.item()))
        tau = self.running_mean.item() + self.k_sigma * std
        return float(min(max(tau, self.min_tau), self.max_tau))

    def forward(
        self,
        h: torch.Tensor,
        update_stats: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Evaluates dynamic confusion and returns gating weight.
        
        Args:
            h: Hidden representation (B, L, D) or (B, D)
            update_stats: Whether to update online EMA statistics
            
        Returns:
            gating_weight: Confusion gating tensor in [0, 1] matching h's spatial dims
            raw_confusion: Raw confusion score in [0, 1]
            dynamic_tau: Currently calibrated adaptive threshold scalar
        """
        orig_ndim = h.ndim
        if orig_ndim == 2:
            h = h.unsqueeze(1)
            
        B, L, D = h.shape
        
        # 1. Project into multi-head subspace
        heads = self.w_head_proj(h).reshape(B, L, self.num_heads, self.head_dim)
        
        # 2. Evaluate head-wise confusion
        head_conf = torch.sigmoid(self.w_confusion_head(heads)).squeeze(-1)  # (B, L, num_heads)
        mean_head_conf = head_conf.mean(dim=-1, keepdim=True)                # (B, L, 1)
        
        # 3. Evaluate context ambiguity
        ctx_ambiguity = torch.sigmoid(self.w_context_ambiguity(h))           # (B, L, 1)
        
        # Combined raw confusion score
        raw_confusion = 0.5 * (mean_head_conf + ctx_ambiguity)               # (B, L, 1)
        
        # 4. Update online EMA statistics across tokens
        current_batch_mean = raw_confusion.detach().mean().float()
        if update_stats and self.training:
            # Batch mode update
            diff = current_batch_mean - self.running_mean
            self.running_mean.copy_(self.running_mean + self.ema_momentum * diff)
            self.running_var.copy_(self.running_var + self.ema_momentum * (diff ** 2 - self.running_var))
            self.step_count += 1
        elif update_stats and not self.training:
            # Inference mode autoregressive smoothing
            diff = current_batch_mean - self.running_mean
            alpha = max(0.01, self.ema_momentum / (1.0 + 0.05 * float(self.step_count.item())))
            self.running_mean.copy_(self.running_mean + alpha * diff)
            self.running_var.copy_(self.running_var + alpha * (diff ** 2 - self.running_var))
            self.step_count += 1
            
        # 5. Compute dynamic adaptive threshold
        dynamic_tau = self.get_dynamic_threshold()
        
        # 6. Gating activation (0 if <= dynamic_tau, smoothly scaling above tau)
        is_confused = raw_confusion > dynamic_tau
        gate = torch.where(
            is_confused,
            (raw_confusion - dynamic_tau) / max(1e-4, 1.0 - dynamic_tau),
            torch.zeros_like(raw_confusion)
        )
        
        if orig_ndim == 2:
            gate = gate.squeeze(1)
            raw_confusion = raw_confusion.squeeze(1)
            
        return gate, raw_confusion, dynamic_tau


# ==============================================================================
# 2. SEMANTIC AFFINITY ENERGY MATRIX (SAEM) - OCKHAM'S RAZOR SLOT MANIFOLD
# ==============================================================================

class SemanticAffinityEnergyMatrix(nn.Module):
    """
    Semantic Affinity Energy Matrix (SAEM).
    
    Organizes concepts into a vertical energy hierarchy of functional slots:
    - High energy (top): Incoherent, bloated, ill-fitting conceptual connections.
    - Low energy (bottom): Deep functional parsimony, coherent minimal solutions (Ockham's Razor).
    
    Enables the model to bind token keys and values into contextual slots that maximize
    functional outcome while minimizing computational/token excess.
    """
    def __init__(
        self,
        d_model: int = 2048,
        num_slots: int = 16,
        d_slot: int = 256,
        temperature: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.d_slot = d_slot
        self.temperature = float(temperature)
        
        # Concept to Slot Key/Value projections
        self.w_key_proj = nn.Linear(d_model, d_slot, bias=False)
        self.w_val_proj = nn.Linear(d_model, d_slot, bias=False)
        
        # Learnable Functional Conceptual Slots (Kotak-kotak Konteks Nyata)
        self.slot_keys = nn.Parameter(torch.randn(num_slots, d_slot) * 0.02)
        self.slot_values = nn.Parameter(torch.randn(num_slots, d_slot) * 0.02)
        
        # Vertical Parsimony Prior: Higher index / top slots incur higher energy penalty
        # Ground state (bottom slots) represent parsimonious, minimal-footprint primitives
        vertical_ladder = torch.linspace(0.0, 1.0, num_slots).unsqueeze(0)
        self.register_buffer("vertical_penalty", vertical_ladder)
        
        # Parsimonious Collapse back to d_model
        self.w_out = nn.Linear(d_slot, d_model, bias=False)
        self.norm_out = nn.LayerNorm(d_model)
        self.alpha_parsimony = nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        h: torch.Tensor,
        return_energy: bool = False,
        parsimony_scale: Optional[float] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Binds input representations into minimal-energy parsimonious slots.
        
        Args:
            h: Hidden representation (B, L, D) or (B, D)
            return_energy: Whether to return vertical energy distribution
            parsimony_scale: Optional continuous scale from Executive Router (never hardcoded)
            
        Returns:
            delta_parsimonious: Parsimonious modulation vector (B, L, D) or (B, D)
            energy_matrix: Energy tensor across slots (B, L, num_slots)
        """
        orig_ndim = h.ndim
        if orig_ndim == 2:
            h = h.unsqueeze(1)
            
        B, L, D = h.shape
        q_keys = self.w_key_proj(h)   # (B, L, d_slot)
        q_vals = self.w_val_proj(h)   # (B, L, d_slot)
        
        # Compute Euclidean Hamiltonian Distance to each functional slot
        # Using float32 for cdist to support bfloat16 on CUDA seamlessly
        q_keys_f = q_keys.float()
        slots_f = self.slot_keys.float().unsqueeze(0).expand(B, -1, -1)
        dist_sq = (torch.cdist(q_keys_f, slots_f, p=2) ** 2).to(dtype=q_keys.dtype)
        
        # Energy formulation: Base distance + Vertical Hierarchy Penalty modulated by parsimony_scale
        scale = 1.0 if parsimony_scale is None else float(parsimony_scale)
        penalty = (self.vertical_penalty.to(dtype=q_keys.dtype) * 2.0) * scale
        energy = (dist_sq / math.sqrt(self.d_slot)) + penalty
        
        # Boltzmann Probability Routing: P(slot) ~ exp(-Energy / T)
        # Lowest energy slots receive highest activation probability (Ockham's Razor)
        routing_weights = F.softmax(-energy / self.temperature, dim=-1)  # (B, L, num_slots)
        
        # Extract parsimonious slot values
        slot_v_expanded = self.slot_values.unsqueeze(0).expand(B, -1, -1).to(dtype=q_keys.dtype)  # (B, num_slots, d_slot)
        routed_val = torch.bmm(routing_weights.reshape(B * L, 1, self.num_slots),
                               slot_v_expanded.repeat_interleave(L, dim=0)).reshape(B, L, self.d_slot)
        
        # Synthesize parsimonious correction
        delta = torch.tanh(self.alpha_parsimony) * self.norm_out(self.w_out(routed_val + q_vals * 0.1))
        
        if orig_ndim == 2:
            delta = delta.squeeze(1)
            energy = energy.squeeze(1)
            
        if return_energy:
            return delta, energy
        return delta, None


# ==============================================================================
# 3. COGNITIVE FLOODGATE & COMPILER TEST-DRIVE ORGAN (TEMBOK ANTI-BANJIR)
# ==============================================================================

class CognitiveFloodgateEngine(nn.Module):
    """
    Cognitive Floodgate & Compiler Test-Drive Feedback Organ.
    
    1. Compiler & Execution Test-Drive Feedback (Poin 1):
       Errors are real verified outcomes from compiler/sandbox execution ('test drive').
       Computes dynamic error penalties using log(tan(...)) scaling, with incoherent (heterogeneous)
       multi-channel projection across model weights so each sub-dimension perceives its unique error footprint.
    2. Semantic Chunk Accumulation & Tan x Sin Trigonometric Divergence (Poin 2):
       Replaces myopic per-token divergence with structured semantic chunk pooling.
       Replaces flat cosine metric with non-linear trigonometric formulation:
           D_trig = tan(theta / 2) * sin(theta / 2)
       where theta is the angle between the pooled semantic chunk and the waypoint expectation.
       Near zero divergence, D_trig -> 0 smoothly (quadratic relaxation); large divergence blows up sharply.
    3. Dynamic Key-LogTan Noise Calibration (Poin 3):
       Noise threshold evolves dynamically per token using percentage of log(tan(...))
       multiplied element-wise by the Key projection vector:
           Noise_t = log(1 + tan(pi/4 * alpha_t)) * (W_k h_t)
       Calibrates dynamic noise dispersion continuously without hardcoded static constants.
    4. Floodgate Capacity Reservoir (Tembok Anti-Banjir, Poin 5):
       Analogous to a dam reservoir holding water. As valid, verified code/data is synthesized
       and passes compiler/AST verification, verified water accumulates:
           V_water <- V_water + delta_v
       When verified water reaches capacity (air melebihi kapasitas tanggul), the floodgate trips:
       task is cleanly declared completed (is_completed = True), preventing runaway over-expansion.
    """
    def __init__(
        self,
        d_model: int = 2048,
        chunk_size: int = 16,
        capacity_limit: float = 100.0,
        ema_momentum: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.capacity_limit = float(capacity_limit)
        self.ema_momentum = float(ema_momentum)
        
        # 0. Neutral Identity Matrix Anchor (Poin 1)
        self.register_buffer("eye_d", torch.eye(d_model))
        
        # 1. Compiler Test-Drive Penalty Projections (Incoherent multi-channel)
        self.w_err_proj = nn.Linear(d_model, d_model, bias=False)
        self.incoherent_weight_mask = nn.Parameter(torch.randn(d_model) * 0.05 + 1.0)
        self.alpha_compiler_penalty = nn.Parameter(torch.tensor(0.08))
        
        # 2. Key Vector Projection for Dynamic Noise Calibration
        self.w_key = nn.Linear(d_model, d_model, bias=False)
        self.w_noise_gate = nn.Linear(d_model, 1, bias=True)
        
        # 3. Semantic Chunk Projections
        self.w_chunk_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm_chunk = nn.LayerNorm(d_model)
        
        # Staging Buffers
        self.register_buffer("accumulated_water", torch.tensor(0.0))
        self.register_buffer("is_task_completed", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("running_noise_mean", torch.tensor(0.50))
        self.register_buffer("running_noise_var", torch.tensor(0.05))
        self.register_buffer("chunk_token_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("chunk_accumulator", torch.zeros(1, d_model))
        self.register_buffer("last_trig_divergence", torch.tensor(0.0))
        self.register_buffer("last_dynamic_threshold", torch.tensor(0.50))
        
        # Initialize cleanly
        nn.init.orthogonal_(self.w_err_proj.weight, gain=0.1)
        nn.init.orthogonal_(self.w_key.weight, gain=0.1)
        nn.init.normal_(self.w_noise_gate.weight, std=0.01)
        nn.init.constant_(self.w_noise_gate.bias, 0.0)
        nn.init.orthogonal_(self.w_chunk_proj.weight, gain=0.1)

    def reset_state(self):
        """Drains floodgate water and resets accumulators for a fresh query/task."""
        self.accumulated_water.zero_()
        self.is_task_completed.fill_(False)
        self.chunk_token_count.zero_()
        self.chunk_accumulator.zero_()
        self.last_trig_divergence.zero_()

    def compute_dynamic_key_noise(self, h: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Calibrates noise dynamically per token: % of log-tan multiplied by key vector.
        Formula: Noise_t = log(1 + tan(pi/4 * alpha_t)) * (W_k h_t)
        Anchored by Identity Matrix (Poin 1).
        """
        orig_ndim = h.ndim
        h_flat = h.reshape(-1, self.d_model) if orig_ndim == 3 else h
        
        # Multiply with identity matrix anchor first (Poin 1)
        if self.eye_d.shape[0] == h_flat.shape[-1]:
            h_neutral = torch.matmul(h_flat, self.eye_d)
        else:
            h_neutral = h_flat
            
        alpha = torch.sigmoid(self.w_noise_gate(h_neutral))  # in [0, 1]
        k_vec = self.w_key(h_neutral)                        # (N, d_model)
        
        # % of log(tan) scaling: log(1 + tan(pi/4 * alpha))
        tan_term = torch.tan(alpha * (math.pi / 4.0))
        p_logtan = torch.log1p(tan_term)                 # in [0, log(2) ~ 0.693]
        
        noise_vector = p_logtan * k_vec                  # (N, d_model)
        noise_norm = noise_vector.norm(dim=-1).mean().item()
        
        # Update running distribution
        diff = noise_norm - self.running_noise_mean.item()
        self.running_noise_mean.copy_(self.running_noise_mean + self.ema_momentum * diff)
        self.running_noise_var.copy_(self.running_noise_var + self.ema_momentum * (diff ** 2 - self.running_noise_var))
        
        # Dynamic threshold = mean + 0.5 * std (evolves naturally, zero hardcoding)
        std = math.sqrt(max(1e-6, self.running_noise_var.item()))
        dynamic_threshold = float(self.running_noise_mean.item() + 0.5 * std)
        self.last_dynamic_threshold.fill_(dynamic_threshold)
        
        if orig_ndim == 3:
            noise_vector = noise_vector.reshape(h.shape[0], h.shape[1], self.d_model)
        return noise_vector, dynamic_threshold

    def compute_trigonometric_divergence(
        self,
        h_chunk: torch.Tensor,
        target_waypoint: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Computes non-linear trigonometric divergence: tan(theta / 2) * sin(theta / 2)
        between semantic chunk representation and expected waypoint.
        """
        h_c = F.normalize(h_chunk, p=2, dim=-1)
        w_t = F.normalize(target_waypoint, p=2, dim=-1)
        
        cos_sim = torch.sum(h_c * w_t, dim=-1).clamp(-0.999, 0.999)
        theta = torch.acos(cos_sim)                       # in (0, pi)
        theta_half = theta * 0.5                          # in (0, pi/2)
        
        # tan(theta / 2) * sin(theta / 2) = sin^2(theta / 2) / cos(theta / 2)
        sin_half = torch.sin(theta_half)
        cos_half = torch.cos(theta_half)
        trig_div = (sin_half ** 2) / (cos_half + 1e-6)    # strictly positive, non-flat
        
        mean_div = float(trig_div.mean().item())
        self.last_trig_divergence.fill_(mean_div)
        return trig_div, mean_div

    def apply_compiler_test_drive_penalty(
        self,
        h: torch.Tensor,
        compiler_error_signal: float
    ) -> torch.Tensor:
        """
        Applies dynamic log-tan penalty from compiler test-drive feedback with incoherent weighting.
        Penalty = log(1 + tan(pi/4 * error)) * W_err(h) * incoherent_mask
        """
        if compiler_error_signal <= 0.01:
            return torch.zeros_like(h)
            
        err_clamped = min(max(float(compiler_error_signal), 0.0), 0.95)
        p_base = math.log1p(math.tan(err_clamped * (math.pi / 4.0)))
        
        # Incoherent heterogeneous projection across model weights anchored by Identity (Poin 1)
        h_flat = h.reshape(-1, self.d_model)
        if self.eye_d.shape[0] == h_flat.shape[-1]:
            h_neutral = torch.matmul(h_flat, self.eye_d)
        else:
            h_neutral = h_flat
        proj = self.w_err_proj(h_neutral) * self.incoherent_weight_mask.unsqueeze(0)
        penalty = torch.tanh(self.alpha_compiler_penalty) * p_base * proj
        
        return penalty.reshape(h.shape)

    def accumulate_semantic_chunk(
        self,
        h: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """
        Accumulates per-token representations into a coherent semantic chunk.
        Returns (h_chunk, True) when a full chunk is formed, otherwise (None, False).
        """
        h_pool = h.mean(dim=1) if h.ndim == 3 else h  # (B, D)
        B = h_pool.shape[0]
        
        # Accumulate into running buffer
        if self.chunk_accumulator.shape[0] != B:
            self.chunk_accumulator = torch.zeros(B, self.d_model, device=h.device, dtype=h.dtype)
            
        self.chunk_accumulator = self.chunk_accumulator + h_pool.detach()
        self.chunk_token_count += 1
        
        if self.chunk_token_count.item() >= self.chunk_size:
            raw_chunk = self.chunk_accumulator / max(1, self.chunk_size)
            h_chunk = self.norm_chunk(self.w_chunk_proj(raw_chunk))
            
            self.chunk_accumulator.zero_()
            self.chunk_token_count.zero_()
            return h_chunk, True
            
        return None, False

    def update_floodgate_capacity(
        self,
        progress_increment: float,
        is_verified_valid: bool = True
    ) -> Tuple[float, bool]:
        """
        Updates verified floodgate water capacity (Tembok Anti-Banjir).
        When accumulated verified progress reaches capacity, trips is_task_completed.
        """
        if is_verified_valid:
            self.accumulated_water += float(progress_increment)
            
        current_water = float(self.accumulated_water.item())
        if current_water >= self.capacity_limit:
            self.is_task_completed.fill_(True)
            
        return current_water, bool(self.is_task_completed.item())

    def forward(
        self,
        h: torch.Tensor,
        target_waypoint: Optional[torch.Tensor] = None,
        compiler_error_signal: float = 0.0,
        is_chunk_valid: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass executing interconnected floodgate dynamics.
        """
        noise_vec, dyn_thresh = self.compute_dynamic_key_noise(h)
        delta_compiler = self.apply_compiler_test_drive_penalty(h, compiler_error_signal)
        h_chunk, is_chunk_ready = self.accumulate_semantic_chunk(h)
        
        trig_div_val = float(self.last_trig_divergence.item())
        is_divergent = False
        
        if is_chunk_ready and target_waypoint is not None:
            if target_waypoint.shape[-1] == self.d_model:
                _, trig_div_val = self.compute_trigonometric_divergence(h_chunk, target_waypoint)
                is_divergent = trig_div_val > dyn_thresh
                
            if is_chunk_valid and not is_divergent:
                self.update_floodgate_capacity(25.0, is_verified_valid=True)
                
        info = {
            "is_task_completed": bool(self.is_task_completed.item()),
            "accumulated_water": float(self.accumulated_water.item()),
            "capacity_limit": self.capacity_limit,
            "is_chunk_ready": is_chunk_ready,
            "trig_divergence": trig_div_val,
            "dynamic_threshold": dyn_thresh,
            "is_divergent": is_divergent,
            "compiler_penalty_applied": compiler_error_signal > 0.01
        }
        return delta_compiler, info


# ==============================================================================
# 4. HIERARCHICAL LATENT PLAN CACHE (HLPC) & LOSTPLAN STAGING REGISTER
# ==============================================================================

class HierarchicalPlanCache(nn.Module):
    """
    Hierarchical Latent Plan Cache (HLPC), Dynamic Alternative Brancher & LostPlan Staging Register.
    Integrated with Cognitive Floodgate Engine, Compiler Test-Drive Feedback,
    Identity Matrix Neutral Anchoring (Poin 1), and Candidate Overlay Projection with
    Selective Blending ("Proyeksi Timpa & Dipilihlah yang Bagus", Poin 2).
    """
    def __init__(
        self,
        d_model: int = 2048,
        num_waypoints: int = 4,
        d_plan: int = 128,
        plan_threshold: float = 0.50,
        decay_factor: float = 0.80,
        chunk_size: int = 16,
        capacity_limit: float = 100.0,
        temperature: float = 0.15
    ):
        super().__init__()
        self.d_model = d_model
        self.num_waypoints = num_waypoints
        self.d_plan = d_plan
        self.plan_threshold = float(plan_threshold)
        self.decay_factor = float(decay_factor)
        self.temperature = float(temperature)
        
        # 0. Neutral Identity Matrix Anchor (Poin 1)
        self.register_buffer("eye_d", torch.eye(d_model))
        self.register_buffer("eye_plan", torch.eye(d_plan))
        
        # 1. Context Engine Plan Necessity Gate (PNG: 1=Plan, 0=Direct)
        self.w_plan_gate = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.w_plan_gate.weight, std=0.01)
        nn.init.constant_(self.w_plan_gate.bias, -0.2)
        
        # 2. Shared Metric Projector (Identity-Anchored Shared Metric Space)
        self.w_metric_proj = nn.Linear(d_model, d_plan, bias=False)
        nn.init.orthogonal_(self.w_metric_proj.weight, gain=0.1)
        
        # 3. Plan Formulator (Waypoints Expansion)
        self.w_plan_expansion = nn.Linear(d_plan, num_waypoints * d_plan, bias=False)
        nn.init.zeros_(self.w_plan_expansion.weight)
        with torch.no_grad():
            self.w_plan_expansion.weight[:d_plan, :d_plan] = torch.eye(d_plan)
        self.norm_plan = nn.LayerNorm(d_plan)
        
        # 4. Dynamic Alternative Route Branching Projector (Plan B Synthesizer)
        # Inputs: target_wp (d_plan) + h_obs (d_plan) + z_incoherent (d_plan) + err_feat (1) = 3 * d_plan + 1
        self.w_alt_branch = nn.Sequential(
            nn.Linear(d_plan * 3 + 1, d_plan, bias=True),
            nn.LayerNorm(d_plan),
            nn.GELU(),
            nn.Linear(d_plan, d_plan, bias=True)
        )
        nn.init.orthogonal_(self.w_alt_branch[0].weight, gain=0.1)
        nn.init.zeros_(self.w_alt_branch[0].bias)
        nn.init.normal_(self.w_alt_branch[3].weight, std=0.01)
        nn.init.constant_(self.w_alt_branch[3].bias, 0.0)
        
        # 5. Plan Guidance Projector back to d_model
        self.w_plan_guidance = nn.Linear(d_plan, d_model, bias=False)
        nn.init.orthogonal_(self.w_plan_guidance.weight, gain=0.1)
        self.alpha_plan = nn.Parameter(torch.tensor(0.08))
        self.alpha_alt = nn.Parameter(torch.tensor(0.08))
        
        # 6. Connected Cognitive Floodgate Engine (Tembok Anti-Banjir)
        self.floodgate = CognitiveFloodgateEngine(
            d_model=d_model,
            chunk_size=chunk_size,
            capacity_limit=capacity_limit
        )
        
        # Staging Buffers & State
        self.register_buffer("active_plan", torch.zeros(1, num_waypoints, d_plan))
        self.register_buffer("lost_plan", torch.zeros(1, num_waypoints, d_plan))
        self.register_buffer("completed_mask", torch.zeros(1, num_waypoints, dtype=torch.bool))
        self.register_buffer("current_step_idx", torch.tensor(0, dtype=torch.long))
        self.register_buffer("is_plan_active", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("alternative_branches_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_discrepancy", torch.tensor(0.0))
        self.register_buffer("is_branching", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("overlay_weights", torch.tensor([1.0, 0.0, 0.0]))

    def reset_plan(self):
        """Clears active plan, LostPlan staging registers, and resets floodgate."""
        self.active_plan.zero_()
        self.lost_plan.zero_()
        self.completed_mask.zero_()
        self.current_step_idx.zero_()
        self.is_plan_active.fill_(False)
        self.alternative_branches_count.zero_()
        self.last_discrepancy.zero_()
        self.is_branching.fill_(False)
        self.overlay_weights.copy_(torch.tensor([1.0, 0.0, 0.0]))
        self.floodgate.reset_state()

    def should_formulate_plan(self, h: torch.Tensor, planning_urgency: Optional[float] = None) -> Tuple[bool, float]:
        """Evaluates Context Engine to determine if task requires planning."""
        h_pool = h.mean(dim=1) if h.ndim == 3 else h
        gate_score = torch.sigmoid(self.w_plan_gate(h_pool)).mean().item()
        if planning_urgency is not None:
            effective_threshold = self.plan_threshold * (1.5 - float(planning_urgency))
        else:
            effective_threshold = self.plan_threshold
        needs_plan = gate_score >= effective_threshold
        return needs_plan, gate_score

    def formulate_plan(self, h: torch.Tensor) -> torch.Tensor:
        """
        Formulates a compact latent plan of K waypoints (<2 KB VRAM)
        anchored by Identity Matrix and Shared Metric Projection.
        """
        orig_ndim = h.ndim
        h_pool = h.mean(dim=1) if orig_ndim == 3 else h
        if self.eye_d.shape[0] == h_pool.shape[-1]:
            h_neutral = torch.matmul(h_pool, self.eye_d)
        else:
            h_neutral = h_pool
            
        B = h_neutral.shape[0]
        h_goal = self.norm_plan(self.w_metric_proj(h_neutral))
        waypoints = self.w_plan_expansion(h_goal).reshape(B, self.num_waypoints, self.d_plan)
        waypoints = self.norm_plan(waypoints)
        
        self.active_plan = waypoints.detach()
        self.lost_plan.zero_()
        self.completed_mask.zero_()
        self.current_step_idx.zero_()
        self.is_plan_active.fill_(True)
        self.alternative_branches_count.zero_()
        self.last_discrepancy.zero_()
        self.is_branching.fill_(False)
        self.floodgate.reset_state()
        return waypoints

    def advance_substep(self, step_idx: Optional[int] = None):
        """
        Transitions completed waypoint into LostPlan staging with decaying memory.
        """
        if not self.is_plan_active:
            return
            
        curr = int(self.current_step_idx.item()) if step_idx is None else step_idx
        if curr < self.num_waypoints:
            self.completed_mask[0, curr] = True
            completed_waypoint = self.active_plan[:, curr:curr+1, :]
            self.lost_plan = self.lost_plan * self.decay_factor
            self.lost_plan[:, curr:curr+1, :] = completed_waypoint * 0.90
            self.current_step_idx.fill_(curr + 1)
            # Advance floodgate verified water on verified substep
            self.floodgate.update_floodgate_capacity(25.0, is_verified_valid=True)
            
        if self.current_step_idx.item() >= self.num_waypoints:
            self.is_plan_active.fill_(False)

    def check_and_branch_alternative(
        self,
        h: torch.Tensor,
        curr_step: int,
        compiler_error_signal: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
        """
        Evaluates whether observed semantic chunk matches the planned waypoint using
        trigonometric tan(theta/2) * sin(theta/2) formulation and dynamic Key-LogTan noise threshold.
        Returns: (target_wp, alt_wp, mean_div, is_branching)
        """
        B = h.shape[0] if h.ndim == 3 else (h.shape[0] if h.ndim == 2 else 1)
        plan_b = self.active_plan.shape[0]
        if plan_b == B:
            target_wp = self.active_plan[:, curr_step, :]
        elif plan_b == 1:
            target_wp = self.active_plan[0:1, curr_step, :].expand(B, -1)
        else:
            target_wp = self.active_plan[:min(B, plan_b), curr_step, :]
            if target_wp.shape[0] < B:
                target_wp = target_wp.repeat(B // target_wp.shape[0] + 1, 1)[:B]

        # 1. Accumulate into semantic chunk (Poin 2)
        h_chunk, is_chunk_ready = self.floodgate.accumulate_semantic_chunk(h)
        
        # If still accumulating inside chunk and no compiler error, maintain current plan smoothly
        if not is_chunk_ready and compiler_error_signal <= 0.05:
            self.is_branching.fill_(False)
            return target_wp, target_wp, float(self.last_discrepancy.item()), False
            
        # 2. Semantic chunk is ready or compiler error reported: evaluate divergence
        h_eval = h_chunk if (is_chunk_ready and h_chunk is not None) else (h.mean(dim=1) if h.ndim == 3 else h)
        if self.eye_d.shape[0] == h_eval.shape[-1]:
            h_eval_neutral = torch.matmul(h_eval, self.eye_d)
        else:
            h_eval_neutral = h_eval
        h_obs = self.norm_plan(self.w_metric_proj(h_eval_neutral))  # (B, d_plan)
        
        # 3. Trigonometric non-linear divergence: tan(theta/2) * sin(theta/2) (Poin 2)
        _, mean_div = self.floodgate.compute_trigonometric_divergence(h_obs, target_wp)
        self.last_discrepancy.fill_(mean_div)
        
        # 4. Calibrate against dynamic Key-LogTan noise threshold (Poin 3)
        _, dyn_thresh = self.floodgate.compute_dynamic_key_noise(h_eval_neutral)
        
        # Trigger alternative branching if divergence > dynamic threshold OR real compiler failure occurred
        is_mismatch = (mean_div > dyn_thresh) or (compiler_error_signal > 0.15)
        
        # Compute real residual deviation vector (Poin 2: Menghilangkan incoherent yang berbelok)
        r_dev = target_wp - h_obs  # (B, d_plan)
        # Deterministic multi-channel heterogeneity without random white noise
        channel_freq = torch.cos(torch.linspace(0, math.pi, self.d_plan, device=h.device, dtype=target_wp.dtype))
        channel_mask = 1.0 + 0.10 * channel_freq.unsqueeze(0)
        z_incoherent = r_dev * channel_mask
        
        err_feat = torch.tensor([[compiler_error_signal]], device=h.device, dtype=target_wp.dtype).expand(B, 1)
        branch_input = torch.cat([target_wp, h_obs, z_incoherent, err_feat], dim=-1)
        alt_waypoint = self.norm_plan(self.w_alt_branch(branch_input))
        
        if is_mismatch:
            self.lost_plan[:, curr_step:curr_step+1, :] = target_wp[:self.lost_plan.shape[0]].unsqueeze(1) * 0.50
            if self.active_plan.shape[0] == B:
                self.active_plan[:, curr_step, :] = alt_waypoint.detach()
            else:
                self.active_plan[0, curr_step, :] = alt_waypoint[0:1].detach()
                
            self.alternative_branches_count += 1
            self.is_branching.fill_(True)
            return target_wp, alt_waypoint, mean_div, True
        else:
            self.floodgate.update_floodgate_capacity(12.5, is_verified_valid=True)
            self.is_branching.fill_(False)
            return target_wp, alt_waypoint, mean_div, False

    def verify_goal_satisfaction(self, h_goal: torch.Tensor) -> Tuple[bool, float]:
        """
        Cross-checks accumulated LostPlan against original goal representation.
        """
        h_pool = h_goal.mean(dim=1) if h_goal.ndim == 3 else h_goal
        accumulated_trace = self.lost_plan.sum(dim=1)
        reconstructed_goal = self.w_plan_guidance(accumulated_trace)
        sim = F.cosine_similarity(h_pool, reconstructed_goal).mean().item()
        is_satisfied = sim >= 0.40
        return is_satisfied, sim

    def forward(
        self,
        h: torch.Tensor,
        planning_urgency: Optional[float] = None,
        compiler_error_signal: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Applies dynamic latent plan guidance with Identity Matrix Anchoring,
        Candidate Overlay Projection & Selective Blending ("Proyeksi Timpa & Dipilihlah yang Bagus").
        """
        orig_ndim = h.ndim
        if orig_ndim == 2:
            h = h.unsqueeze(1)
            
        B, L, D = h.shape
        needs_plan, gate_score = self.should_formulate_plan(h, planning_urgency=planning_urgency)
        
        if not self.is_plan_active and needs_plan:
            self.formulate_plan(h)
            
        # 1. Identity Matrix Neutral Baseline (Poin 1: Netral Anchor)
        if self.eye_d.shape[0] == D:
            y_0 = torch.matmul(h, self.eye_d)
        else:
            y_0 = h.clone()
            
        # Compiler test-drive dynamic penalty injection (Poin 1)
        delta_compiler = self.floodgate.apply_compiler_test_drive_penalty(h, compiler_error_signal)
        
        discrepancy_score = 0.0
        is_branching = False
        lateral_norm = 0.0
        weights_selected = [1.0, 0.0, 0.0]
        
        if self.is_plan_active:
            curr_step = min(int(self.current_step_idx.item()), self.num_waypoints - 1)
            
            # Check expectation-reality match at semantic chunk boundaries
            target_wp, alt_wp, discrepancy_score, is_branching = self.check_and_branch_alternative(
                h, curr_step, compiler_error_signal=compiler_error_signal
            )
            
            # 2. Candidate Overlay Projections (Poin 2: Proyeksi Timpa)
            # Candidate 1: Macro Plan Guidance Overlay
            guidance_plan = self.w_plan_guidance(target_wp.to(dtype=h.dtype)).unsqueeze(1).expand(B, L, D)
            y_1 = y_0 + torch.tanh(self.alpha_plan) * guidance_plan
            
            # Candidate 2: Alternative Recovery Overlay
            guidance_alt = self.w_plan_guidance(alt_wp.to(dtype=h.dtype)).unsqueeze(1).expand(B, L, D)
            y_2 = y_0 + torch.tanh(self.alpha_alt) * guidance_alt - delta_compiler
            
            # 3. Quality Evaluation & Boundary Check ("dipilihlah yang bagus")
            candidates = [y_0, y_1, y_2]
            q_scores = []
            
            for k, y_cand in enumerate(candidates):
                # A. Syntax Fidelity Boundary (cosine similarity with neutral y_0)
                cos_syntax = F.cosine_similarity(y_cand, y_0, dim=-1).mean(dim=-1)  # (B,)
                # Hard boundary barrier: below 0.90 is heavily penalized to protect Python grammar
                syntax_pen = torch.where(cos_syntax >= 0.90, cos_syntax, cos_syntax - 6.0 * (0.90 - cos_syntax))
                
                # B. Goal Specification Affinity
                goal_target = y_0 + guidance_plan
                cos_goal = F.cosine_similarity(y_cand, goal_target, dim=-1).mean(dim=-1)  # (B,)
                
                # C. Compiler Error Penalty
                if compiler_error_signal > 0.01:
                    err_target = y_0 + delta_compiler
                    err_penalty = compiler_error_signal * torch.relu(F.cosine_similarity(y_cand, err_target, dim=-1).mean(dim=-1))
                else:
                    err_penalty = torch.zeros(B, device=h.device, dtype=h.dtype)
                    
                # Candidate Fitness Score
                q_k = 2.0 * syntax_pen + 1.0 * cos_goal - 2.0 * err_penalty
                q_scores.append(q_k)
                
            q_stack = torch.stack(q_scores, dim=-1)  # (B, 3)
            weights = F.softmax(q_stack / self.temperature, dim=-1)  # (B, 3)
            weights_selected = [float(weights[0, 0].item()), float(weights[0, 1].item()), float(weights[0, 2].item())]
            self.overlay_weights.copy_(weights[0].detach())
            
            # Selective Overlay Combination ("dipilihlah yang bagus")
            y_star = (
                weights[:, 0:1, None] * y_0 +
                weights[:, 1:2, None] * y_1 +
                weights[:, 2:3, None] * y_2
            )
            delta_plan = y_star - h
            lateral_norm = float((y_2 - y_1).norm().item()) / max(1, B * L)
        else:
            delta_plan = delta_compiler
            
        if orig_ndim == 2:
            delta_plan = delta_plan.squeeze(1)
            
        info = {
            "is_plan_active": bool(self.is_plan_active.item()),
            "gate_score": float(gate_score),
            "current_step_idx": int(self.current_step_idx.item()),
            "completed_count": int(self.completed_mask.sum().item()),
            "discrepancy_score": float(discrepancy_score),
            "is_branching": bool(is_branching),
            "alternative_branches_count": int(self.alternative_branches_count.item()),
            "lateral_norm": float(lateral_norm),
            "is_task_completed": bool(self.floodgate.is_task_completed.item()),
            "accumulated_water": float(self.floodgate.accumulated_water.item()),
            "capacity_limit": self.floodgate.capacity_limit,
            "overlay_weights": weights_selected
        }
        return delta_plan, info
