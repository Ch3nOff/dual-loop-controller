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
        current_batch_mean = raw_confusion.detach().mean()
        if update_stats and self.training:
            # Batch mode update
            diff = current_batch_mean - self.running_mean
            self.running_mean += self.ema_momentum * diff
            self.running_var += self.ema_momentum * (diff ** 2 - self.running_var)
            self.step_count += 1
        elif update_stats and not self.training:
            # Inference mode autoregressive smoothing
            diff = current_batch_mean - self.running_mean
            alpha = max(0.01, self.ema_momentum / (1.0 + 0.05 * float(self.step_count.item())))
            self.running_mean += alpha * diff
            self.running_var += alpha * (diff ** 2 - self.running_var)
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
        return_energy: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Binds input representations into minimal-energy parsimonious slots.
        
        Args:
            h: Hidden representation (B, L, D) or (B, D)
            return_energy: Whether to return vertical energy distribution
            
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
        # Distance squared: ||q - slot_k||^2
        dist_sq = torch.cdist(q_keys, self.slot_keys.unsqueeze(0).expand(B, -1, -1), p=2) ** 2
        
        # Energy formulation: Base distance + Vertical Hierarchy Penalty
        # Top = high energy (ill-fitting/mubazir), Bottom = ground state (parsimonious)
        energy = (dist_sq / math.sqrt(self.d_slot)) + (self.vertical_penalty * 2.0)
        
        # Boltzmann Probability Routing: P(slot) ~ exp(-Energy / T)
        # Lowest energy slots receive highest activation probability (Ockham's Razor)
        routing_weights = F.softmax(-energy / self.temperature, dim=-1)  # (B, L, num_slots)
        
        # Extract parsimonious slot values
        slot_v_expanded = self.slot_values.unsqueeze(0).expand(B, -1, -1)  # (B, num_slots, d_slot)
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
# 3. HIERARCHICAL LATENT PLAN CACHE (HLPC) & LOSTPLAN STAGING REGISTER
# ==============================================================================

class HierarchicalPlanCache(nn.Module):
    """
    Hierarchical Latent Plan Cache (HLPC) & LostPlan Staging Register.
    
    Executive function organ operating at human cognitive scale:
    1. Context Engine Plan Necessity Gate (PNG):
       Continuously measures complexity in [0, 1]. If 0 (routine/simple), planning is bypassed.
       If 1 ("wah ini bakal ribet"), triggers compact latent plan formulation.
    2. Ultra-Lightweight Latent Waypoints:
       Stores K plan waypoints as compact latent vectors (<2 KB VRAM footprint).
    3. LostPlan Staged Decay Register:
       Completed sub-plans do NOT vanish instantly; they transition into a decaying
       LostPlan buffer to retain historical trace without cluttering working memory.
    4. Goal Cross-Verification:
       Cross-checks completed LostPlan trace against the original goal representation.
    """
    def __init__(
        self,
        d_model: int = 2048,
        num_waypoints: int = 4,
        d_plan: int = 128,
        plan_threshold: float = 0.50,
        decay_factor: float = 0.80
    ):
        super().__init__()
        self.d_model = d_model
        self.num_waypoints = num_waypoints
        self.d_plan = d_plan
        self.plan_threshold = float(plan_threshold)
        self.decay_factor = float(decay_factor)
        
        # 1. Context Engine Plan Necessity Gate (PNG: 1=Plan, 0=Direct)
        self.w_plan_gate = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.w_plan_gate.weight, std=0.01)
        nn.init.constant_(self.w_plan_gate.bias, -0.2)  # Slight bias towards direct execution
        
        # 2. Plan Formulator (Compact Latent Waypoints)
        self.w_plan_encoder = nn.Linear(d_model, num_waypoints * d_plan, bias=False)
        self.norm_plan = nn.LayerNorm(d_plan)
        
        # 3. Plan Guidance Projector back to d_model
        self.w_plan_guidance = nn.Linear(d_plan, d_model, bias=False)
        self.alpha_plan = nn.Parameter(torch.tensor(0.08))
        
        # 4. Goal Verifier
        self.w_goal_verifier = nn.CosineSimilarity(dim=-1)
        
        # Staging Buffers
        self.register_buffer("active_plan", torch.zeros(1, num_waypoints, d_plan))
        self.register_buffer("lost_plan", torch.zeros(1, num_waypoints, d_plan))
        self.register_buffer("completed_mask", torch.zeros(1, num_waypoints, dtype=torch.bool))
        self.register_buffer("current_step_idx", torch.tensor(0, dtype=torch.long))
        self.register_buffer("is_plan_active", torch.tensor(False, dtype=torch.bool))

    def reset_plan(self):
        """Clears active plan and LostPlan staging registers."""
        self.active_plan.zero_()
        self.lost_plan.zero_()
        self.completed_mask.zero_()
        self.current_step_idx.zero_()
        self.is_plan_active.fill_(False)

    def should_formulate_plan(self, h: torch.Tensor) -> Tuple[bool, float]:
        """Evaluates Context Engine to determine if task requires planning."""
        h_pool = h.mean(dim=1) if h.ndim == 3 else h
        gate_score = torch.sigmoid(self.w_plan_gate(h_pool)).mean().item()
        needs_plan = gate_score >= self.plan_threshold
        return needs_plan, gate_score

    def formulate_plan(self, h: torch.Tensor) -> torch.Tensor:
        """
        Formulates a compact latent plan of K waypoints (<2 KB VRAM).
        """
        h_pool = h.mean(dim=1) if h.ndim == 3 else h
        B = h_pool.shape[0]
        waypoints = self.w_plan_encoder(h_pool).reshape(B, self.num_waypoints, self.d_plan)
        waypoints = self.norm_plan(waypoints)
        
        # Cache active plan
        self.active_plan = waypoints.detach()
        self.lost_plan.zero_()
        self.completed_mask.zero_()
        self.current_step_idx.zero_()
        self.is_plan_active.fill_(True)
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
            # Transition to LostPlan with decay
            completed_waypoint = self.active_plan[:, curr:curr+1, :]
            # Decay existing lost plans
            self.lost_plan = self.lost_plan * self.decay_factor
            # Add new completed waypoint into lost plan register
            self.lost_plan[:, curr:curr+1, :] = completed_waypoint * 0.90
            self.current_step_idx.fill_(curr + 1)
            
        if self.current_step_idx.item() >= self.num_waypoints:
            self.is_plan_active.fill_(False)

    def verify_goal_satisfaction(self, h_goal: torch.Tensor) -> Tuple[bool, float]:
        """
        Cross-checks accumulated LostPlan against original goal representation.
        """
        h_pool = h_goal.mean(dim=1) if h_goal.ndim == 3 else h_goal
        # Reconstruct executed trajectory from LostPlan
        accumulated_trace = self.lost_plan.sum(dim=1)  # (1, d_plan)
        reconstructed_goal = self.w_plan_guidance(accumulated_trace)  # (1, d_model)
        
        sim = F.cosine_similarity(h_pool, reconstructed_goal).mean().item()
        # If similarity is above 0.40, plan execution aligns with goal
        is_satisfied = sim >= 0.40
        return is_satisfied, sim

    def forward(
        self,
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Applies latent plan guidance if planning is active or warranted.
        
        Returns:
            delta_plan: Latent planning modulation vector (B, L, D) or (B, D)
            info_dict: Metadata including plan_active, gate_score, and step_idx
        """
        orig_ndim = h.ndim
        if orig_ndim == 2:
            h = h.unsqueeze(1)
            
        B, L, D = h.shape
        needs_plan, gate_score = self.should_formulate_plan(h)
        
        # If plan is not currently active, but complexity is high, formulate now!
        if not self.is_plan_active and needs_plan:
            self.formulate_plan(h)
            
        delta_plan = torch.zeros(B, L, D, device=h.device, dtype=h.dtype)
        if self.is_plan_active:
            curr_step = min(int(self.current_step_idx.item()), self.num_waypoints - 1)
            target_waypoint = self.active_plan[:, curr_step, :]  # (B, d_plan)
            # Re-expand to sequence
            guidance = self.w_plan_guidance(target_waypoint).unsqueeze(1)  # (B, 1, D)
            delta_plan = torch.tanh(self.alpha_plan) * guidance.expand(B, L, D)
            
        if orig_ndim == 2:
            delta_plan = delta_plan.squeeze(1)
            
        info = {
            "is_plan_active": bool(self.is_plan_active.item()),
            "gate_score": float(gate_score),
            "current_step_idx": int(self.current_step_idx.item()),
            "completed_count": int(self.completed_mask.sum().item())
        }
        return delta_plan, info
