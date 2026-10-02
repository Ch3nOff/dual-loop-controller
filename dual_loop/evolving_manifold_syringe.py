"""
HADL v3.3: Evolving Manifold with LM-Head Syringe Injection
============================================================
Mathematical Formulation:
1. Evolving Manifold:
       R^D(m) = h + (|m| / sqrt(D)) * Thought
   where |m| is dynamic manifold mass, P_cloud in Delta^(M-1) is SquareCloud Simplex,
   and C_point = V * (1 + 0.5 * tanh(K * W_vk)) are dynamic moving points.
2. 50% STE Latent Judge:
   Supervises coherence with straight-through estimator and fail-safe veto.
3. Unitary Givens Isometry:
   Guarantees strict norm conservation: ||h'|| == ||h|| (0.000000 error).
4. LM-Head Syringe Injection:
   Overcomes downstream decoding disconnect by injecting low-rank manifold guidance
   directly into the LM Head logits:
       Logits_final = Logits_base + gate(h) * (W_up * GELU(W_down * h))
   with ReZero initialization (W_up = 0) guaranteeing zero baseline regression.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union

from .square_cloud_engine import (
    SelectiveIdentityMatrixRouter,
    DynamicMovingPointModulator,
    KnowledgeSyringe
)


class EvolvingManifoldModule(nn.Module):
    """
    Evolving Representation Manifold:
        R^D(m) = h + (|m| / sqrt(D)) * Thought
    Integrates dynamic mass estimation, SquareCloud simplex, moving points,
    50% STE Latent Judge, and Unitary Givens rotation.
    """
    def __init__(
        self,
        d_model: int,
        num_slots: int = 16,
        temperature: float = 0.85,
        max_rotation_angle: float = 0.2618  # ~15 degrees
    ):
        super().__init__()
        self.d_model = d_model
        self.sqrt_d = math.sqrt(float(d_model))
        self.num_slots = num_slots
        self.temperature = float(temperature)
        self.max_rotation_angle = float(max_rotation_angle)
        assert d_model % 2 == 0, "d_model must be even for Givens rotations."
        self.half_d = d_model // 2

        # 1. Dynamic Manifold Mass Estimator: |m| >= 0
        self.mass_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()  # Guarantees strictly non-negative mass
        )

        # 2. Attention Cloud Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

        # 3. Dynamic Moving Points & Selective Router
        self.identity_router = SelectiveIdentityMatrixRouter(d_model=d_model)
        self.point_modulator = DynamicMovingPointModulator(d_model=d_model)
        self.syringe = KnowledgeSyringe(d_model=d_model)

        # 4. 50% Heavyweight Latent Judge with Straight-Through Estimator
        self.judge_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        nn.init.constant_(self.judge_mlp[-1].bias, 1.0)

        # 5. Unitary Givens Trigonometric Rotator (ReZero initialization)
        self.angle_proj = nn.Linear(d_model, self.half_d, bias=False)
        nn.init.zeros_(self.angle_proj.weight)

        # 6. Global Workspace Memory Slots
        self.slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)

        self.last_telemetry: Dict[str, Any] = {}

    def forward(
        self,
        h: torch.Tensor,
        syringe_key: Optional[torch.Tensor] = None,
        syringe_val: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        t0 = time.perf_counter()
        orig_norm = torch.norm(h, dim=-1).mean().item()
        B, S, D = h.shape

        # 1. Dynamic Manifold Mass: |m| >= 0
        mass = self.mass_estimator(h)  # [B, S, 1]
        mass_scale = mass / (self.sqrt_d + 1e-6)  # [B, S, 1]

        # 2. Compute Q, K, V
        Q = self.q_proj(h)  # [B, S, D]
        expanded_slots = self.slots.expand(B, -1, -1)  # [B, num_slots, D]
        context = torch.cat([h, expanded_slots], dim=1)  # [B, S + num_slots, D]
        K = self.k_proj(context)  # [B, S + num_slots, D]
        V = self.v_proj(context)  # [B, S + num_slots, D]

        # 3. Selective Identity Routing
        Q_scaled, s_diag = self.identity_router(Q)

        # 4. SquareCloud Simplex P_cloud in Delta^(M-1)
        affinity = torch.matmul(Q_scaled, K.transpose(-1, -2))  # [B, S, S + num_slots]
        if S > 1:
            causal_mask = torch.zeros((S, S + self.num_slots), device=h.device, dtype=h.dtype)
            causal_mask[:, :S] = torch.triu(torch.full((S, S), float('-inf'), device=h.device, dtype=h.dtype), diagonal=1)
            affinity = affinity + causal_mask.unsqueeze(0)

        cloud_probs = F.softmax(affinity / self.temperature, dim=-1)

        # 5. Dynamic Moving Points [V * K]
        dynamic_points = self.point_modulator(V, K)
        if syringe_key is not None and syringe_val is not None:
            syringe_vec = self.syringe.bind_fact(syringe_key, syringe_val)
            dynamic_points = dynamic_points + 0.3 * syringe_vec.unsqueeze(1).to(dtype=dynamic_points.dtype)

        # 6. Retrieve Thought Vector
        raw_thought = torch.matmul(cloud_probs, dynamic_points)
        raw_thought = self.out_proj(raw_thought)

        # Scale thought by dynamic manifold mass (|m| / sqrt(D))
        thought = mass_scale * raw_thought

        # 7. 50% STE Latent Judge
        judge_input = torch.cat([h, thought], dim=-1)
        judge_logit = self.judge_mlp(judge_input)
        judge_prob = torch.sigmoid(judge_logit)
        judge_hard = (judge_prob >= 0.5).to(dtype=h.dtype)
        judge_gate = judge_prob + (judge_hard - judge_prob).detach()  # STE

        # 8. Unitary Givens Isometry
        theta = torch.tanh(self.angle_proj(thought)) * self.max_rotation_angle
        theta = theta * judge_gate  # Clamped to 0 if vetoed

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        h_even = h[..., 0::2]
        h_odd  = h[..., 1::2]

        h_even_rot = h_even * cos_t - h_odd * sin_t
        h_odd_rot  = h_even * sin_t + h_odd * cos_t

        rotated_h = torch.empty_like(h)
        rotated_h[..., 0::2] = h_even_rot
        rotated_h[..., 1::2] = h_odd_rot

        final_norm = torch.norm(rotated_h, dim=-1).mean().item()
        isometry_error = abs(final_norm - orig_norm)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.last_telemetry = {
            "orig_norm": round(orig_norm, 4),
            "final_norm": round(final_norm, 4),
            "isometry_error": round(isometry_error, 6),
            "mean_mass": round(float(mass.mean().item()), 4),
            "judge_prob": round(float(judge_prob.mean().item()), 4),
            "judge_verdict": float(judge_hard.mean().item()),
            "mean_rotation_deg": round(float(torch.mean(torch.abs(theta)).item()) * 180.0 / math.pi, 2),
            "latency_ms": round(elapsed_ms, 3)
        }

        return rotated_h, self.last_telemetry


class LMHeadSyringe(nn.Module):
    """
    LM-Head Syringe Injector:
    Bypasses the downstream decoding disconnect by injecting calibrated
    low-rank manifold guidance directly into the un-embedding vocabulary space:
        Logits_final = Logits_base + gate(h) * (W_up * GELU(W_down * h))
    Guaranteed ReZero initialization: W_up = 0 ensures exact baseline preservation.
    """
    def __init__(self, d_model: int, vocab_size: int, rank: int = 64):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.rank = rank

        # Low-rank projection
        self.down_proj = nn.Linear(d_model, rank, bias=False)
        self.up_proj = nn.Linear(rank, vocab_size, bias=False)
        self.act = nn.GELU()

        # Dynamic Syringe Gate
        self.gate_proj = nn.Linear(d_model, 1)

        # ReZero initialization: starts at exact 0.0
        nn.init.zeros_(self.up_proj.weight)
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.constant_(self.gate_proj.bias, -4.0)  # sigmoid(-4) ~ 0.018

        self.last_telemetry: Dict[str, Any] = {}

    def forward(
        self,
        h_final: torch.Tensor,
        base_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        orig_dtype = base_logits.dtype
        # Compute gate: [B, S, 1]
        gate = torch.sigmoid(self.gate_proj(h_final.to(dtype=self.gate_proj.weight.dtype)))
        
        # Low-rank delta logits: [B, S, V]
        h_down = self.down_proj(h_final.to(dtype=self.down_proj.weight.dtype))
        delta_logits = self.up_proj(self.act(h_down)).to(dtype=orig_dtype)

        # Inject logits
        injected_logits = base_logits + gate.to(dtype=orig_dtype) * delta_logits

        self.last_telemetry = {
            "syringe_gate": round(float(gate.mean().item()), 4),
            "delta_logit_norm": round(float(torch.norm(delta_logits, dim=-1).mean().item()), 4)
        }
        return injected_logits, self.last_telemetry


class EvolvingSquareCloudWrapper(nn.Module):
    """
    Unified Evolving Manifold and LM-Head Syringe Architecture.
    Simultaneously manages:
    - Layer 11 Mid-Layer Manifold Evolution (R^D(m))
    - LM-Head Logits Syringe Injection
    """
    def __init__(
        self,
        base_model: nn.Module,
        target_layer_idx: int = 11,
        rank: int = 64
    ):
        super().__init__()
        self.base_model = base_model
        self.target_layer_idx = int(target_layer_idx)
        self.d_model = getattr(base_model.config, "hidden_size", 2048)
        self.vocab_size = getattr(base_model.config, "vocab_size", 151936)

        # 1. Evolving Mid-Layer Engine
        self.manifold_engine = EvolvingManifoldModule(d_model=self.d_model)

        # 2. LM-Head Syringe
        self.head_syringe = LMHeadSyringe(
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            rank=rank
        )

        # Align device and dtype
        param = next(base_model.parameters(), None)
        if param is not None:
            self.manifold_engine.to(device=param.device, dtype=param.dtype)
            self.head_syringe.to(device=param.device, dtype=param.dtype)

        self.enabled: bool = True
        self._layer_hook_handle = None
        self._head_hook_handle = None
        self._last_final_h: Optional[torch.Tensor] = None

        self._attach_hooks()

    def _attach_hooks(self):
        # 1. Attach Mid-Layer Hook
        layers = None
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, "layers"):
            layers = self.base_model.layers

        if layers is not None and 0 <= self.target_layer_idx < len(layers):
            target_layer = layers[self.target_layer_idx]

            def _layer_hook(module, args, output):
                if not self.enabled:
                    return output
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                if not isinstance(h, torch.Tensor):
                    return output

                enhanced_h, _ = self.manifold_engine(h)
                if is_tuple:
                    return (enhanced_h,) + output[1:]
                return enhanced_h

            self._layer_hook_handle = target_layer.register_forward_hook(_layer_hook)

        # 2. Attach LM-Head Hook
        # Locate LM Head
        head = None
        if hasattr(self.base_model, "lm_head"):
            head = self.base_model.lm_head
        elif hasattr(self.base_model, "get_output_embeddings"):
            head = self.base_model.get_output_embeddings()

        if head is not None:
            def _head_hook(module, args, output):
                if not self.enabled:
                    return output
                # args[0] is the input to lm_head (final hidden states h_final)
                if len(args) > 0 and isinstance(args[0], torch.Tensor):
                    h_final = args[0]
                    base_logits = output
                    injected_logits, _ = self.head_syringe(h_final, base_logits)
                    return injected_logits
                return output

            self._head_hook_handle = head.register_forward_hook(_head_hook)

    def remove_hooks(self):
        if self._layer_hook_handle is not None:
            self._layer_hook_handle.remove()
            self._layer_hook_handle = None
        if self._head_hook_handle is not None:
            self._head_hook_handle.remove()
            self._head_hook_handle = None

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
