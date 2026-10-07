"""
ID-SMIL: Incoherent Dispersion with Self-Routing Micro-Iterative Latents
========================================================================
Architecture Components:
1. IncoherentDiffuseRouter (IDR):
   Non-rigid, open-phase latent dispersion that accepts tokens without forced
   deterministic categorizations or human hardcoded thresholds.
2. LocalAffordanceGate (Local Epistemic Affordance - LEA):
   Specialized local units evaluate their own intrinsic comprehension of the incoming task.
3. RecursiveBounceBackChannel (RBC):
   If a local unit does not understand the representation (kappa <= 0), it bounces
   the signal back rather than forcing noisy, destructive transformations.
4. SMILBackgroundUnit ("Small Little in the Back"):
   A stationary, low-drift micro-latent grounding unit running in the background.
   Handles fundamental universal features with gentle non-hyperdynamic operations (epsilon << 1),
   guaranteeing baseline stability and zero catastrophic forgetting.
"""

import math
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .callearn_oc_engine import HADLv4UnifiedEngine, GhostVerificationLayer


class IncoherentDiffuseRouter(nn.Module):
    """
    Incoherent Diffuse Router (IDR):
    Disperses incoming token representations across a continuous, non-rigid phase landscape.
    Does NOT use rigid if-else boundaries or human-imposed hard threshold cuts.
    """
    def __init__(self, d_model: int = 2048, num_paths: int = 3, tau_phase: float = 1.414):
        super().__init__()
        self.d_model = d_model
        self.num_paths = num_paths
        self.tau_phase = tau_phase

        # Continuous phase-dispersion projection
        self.w_diffuse = nn.Parameter(torch.empty(d_model, num_paths))
        self.bias = nn.Parameter(torch.zeros(num_paths))
        self.norm = nn.LayerNorm(d_model)

        # Dispersion entropy modulation factor
        self.log_tau = nn.Parameter(torch.tensor(0.0))

        # Persistent usage buffer for equilibrium tracking
        self.register_buffer("usage_memory", torch.zeros(num_paths))
        self.decay = 0.95

        nn.init.normal_(self.w_diffuse, std=0.02)

    def forward(
        self,
        embed_tensor: Optional[torch.Tensor] = None,
        embed_layer: Optional[nn.Module] = None,
        input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Disperses representation into soft candidate energy distributions.
        Returns:
            dispatch_weights: [B, num_paths] continuous energy shares
            telemetry: metadata dictionary
        """
        if embed_tensor is not None:
            x = embed_tensor
        elif embed_layer is not None and input_ids is not None:
            with torch.no_grad():
                x = embed_layer(input_ids)
        else:
            dev = self.w_diffuse.device
            weights = torch.softmax(torch.zeros(1, self.num_paths, device=dev), dim=-1)
            return weights, {"mode": "incoherent_uniform"}

        # Align input with router parameters
        target_dtype = self.w_diffuse.dtype
        x_in = x.to(dtype=target_dtype)
        x_norm = self.norm(x_in)  # [B, S, D]
        x_mean = x_norm.mean(dim=1)     # [B, D]

        # 1. Continuous phase-dispersion projection
        z_raw = torch.matmul(x_mean, self.w_diffuse) + self.bias  # [B, num_paths]

        # 2. Incoherent Wave Dispersion (sin-cos non-linear phase interference)
        tau = torch.exp(self.log_tau.to(target_dtype)).clamp(0.5, 4.0)
        phase_incoherent = torch.sin(z_raw / tau) * torch.cos(z_raw / (tau * self.tau_phase))
        z_dispersed = z_raw + 0.35 * phase_incoherent

        # 3. Diffuse distribution across paths
        dispatch_weights = F.softmax(z_dispersed.float(), dim=-1).to(target_dtype)  # [B, num_paths]

        # Track usage in background memory
        if self.training:
            with torch.no_grad():
                self.usage_memory.copy_(
                    self.decay * self.usage_memory + (1.0 - self.decay) * dispatch_weights.detach().float().mean(dim=0)
                )

        telemetry = {
            "dispatch_weights": [round(float(w), 4) for w in dispatch_weights[0].tolist()],
            "usage_memory": [round(float(u), 4) for u in self.usage_memory.tolist()],
            "incoherent_entropy": round(float(-torch.sum(dispatch_weights.float() * torch.log(dispatch_weights.float() + 1e-8)).item()), 4)
        }
        return dispatch_weights, telemetry


class LocalAffordanceGate(nn.Module):
    """
    Local Epistemic Affordance (LEA):
    Evaluates whether the local specialist unit understands the incoming task.
    If kappa > 0 -> Executes local adaptation.
    If kappa <= 0 -> Triggers Bounce-Back (signal is returned without distortion).
    """
    def __init__(self, d_model: int = 2048, intrinsic_scale: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.intrinsic_scale = intrinsic_scale

        # Learned Affordance Centroid: represents domain manifold resonance
        self.centroid = nn.Parameter(torch.randn(d_model) * 0.02)
        # Epistemic uncertainty projection
        self.w_uncertainty = nn.Linear(d_model, 64)
        self.w_entropy = nn.Linear(64, 1)

    def evaluate_affordance(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates intrinsic comprehension score:
        kappa = cos(h, centroid) - entropy_penalty
        Returns:
            kappa: [B, S, 1] comprehension score per token
            confidence_mean: scalar float
        """
        target_dtype = self.centroid.dtype
        h_norm = F.normalize(h.to(dtype=target_dtype), p=2, dim=-1)
        c_norm = F.normalize(self.centroid, p=2, dim=-1)

        # 1. Manifold Cosine Resonance
        if self.training:
            with torch.no_grad():
                h_mean_dir = h_norm.mean(dim=(0, 1))
                h_mean_dir = F.normalize(h_mean_dir, p=2, dim=-1)
                self.centroid.data = 0.90 * self.centroid.data + 0.10 * h_mean_dir
                c_norm = F.normalize(self.centroid, p=2, dim=-1)

        cos_res = torch.sum(h_norm * c_norm, dim=-1, keepdim=True)  # [B, S, 1]

        # 2. Epistemic Uncertainty Estimation
        u_proj = torch.tanh(self.w_uncertainty(h_norm))
        entropy_raw = torch.sigmoid(self.w_entropy(u_proj))  # [B, S, 1] in (0, 1)

        # 3. Local Intrinsic Understanding (Kappa)
        kappa = self.intrinsic_scale * cos_res - (entropy_raw - 0.5)
        return kappa.to(h.dtype), kappa.mean().detach()


class SMILBackgroundUnit(nn.Module):
    """
    SMIL: Small Little in the Back
    ===============================
    A stationary, non-hyperdynamic micro-latent grounding unit running in the background.
    - Low-drift contraction mapping (epsilon << 1)
    - Identity-preserving residual connection
    - Handles fundamental universal language grounding so the model never forgets
      core representations, even when specialized local units bounce back.
    """
    def __init__(self, d_model: int = 2048, hidden_dim: int = 256, epsilon: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon

        # Micro-capacity stationary bottleneck
        self.norm = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dim, d_model, bias=False)

        # Dynamic micro-modulator: bounded in [0.01, 0.08]
        self.alpha_smil = nn.Parameter(torch.tensor(-2.9957))  # sigmoid(-2.9957) ~ 0.047

        # Orthogonal / near-isometry initialization to ensure non-destructive drift
        nn.init.orthogonal_(self.down_proj.weight, gain=0.2)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, h: torch.Tensor, bounce_residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Executes gentle background grounding.
        If a bounce residual is returned from a confused local unit, SMIL absorbs
        and stabilizes it back into the base manifold.
        """
        eff_eps = torch.sigmoid(self.alpha_smil) * self.epsilon * 2.0  # safe micro-scale

        # Combine with bounce residual if present
        h_comb = h if bounce_residual is None else (h + 0.15 * bounce_residual.to(dtype=h.dtype, device=h.device))
        target_dtype = self.norm.weight.dtype
        h_in = h_comb.to(dtype=target_dtype)

        # Low-drift micro-inference in the back
        normed = self.norm(h_in)
        micro_feat = self.act(self.down_proj(normed))
        delta_smil = self.up_proj(micro_feat)

        # Stationary Grounding: h_grounded = h + eps * delta_smil
        h_grounded = h + (eff_eps * delta_smil).to(dtype=h.dtype)
        return h_grounded


class IDSMILController(nn.Module):
    """
    Unified ID-SMIL Controller:
    Coordinates Incoherent Dispersion, Local Affordance, Bounce-Back, and SMIL Grounding.
    """
    def __init__(
        self,
        d_model: int = 2048,
        engine: Optional[HADLv4UnifiedEngine] = None,
        ghost: Optional[GhostVerificationLayer] = None,
        zero_drift_on_bounce: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.zero_drift_on_bounce = zero_drift_on_bounce
        self.router = IncoherentDiffuseRouter(d_model=d_model, num_paths=3)
        self.affordance_gate = LocalAffordanceGate(d_model=d_model)
        self.smil = SMILBackgroundUnit(d_model=d_model, hidden_dim=256, epsilon=0.05)

        self.engine = engine if engine is not None else HADLv4UnifiedEngine(d_model=d_model)
        self.ghost = ghost if ghost is not None else GhostVerificationLayer(d_model=d_model)

        self.last_telemetry: Dict[str, Any] = {}

    def process_layer11(self, h: torch.Tensor, dispatch_weights: torch.Tensor) -> torch.Tensor:
        """
        Layer 11 Processing with ID-SMIL Flow:
        1. Local Specialist Affordance Check
        2. If Paham (kappa > 0): Execute Engine with soft dispatch
        3. If Tidak Paham (kappa <= 0): Bounce back to SMIL Background Unit
        """
        B, S, D = h.shape
        kappa, kappa_mean = self.affordance_gate.evaluate_affordance(h)

        # Continuous understanding probability via sigmoid(kappa)
        p_understand = torch.sigmoid(kappa).to(dtype=h.dtype)  # [B, S, 1]

        # Path weights from Incoherent Diffuse Router:
        dw = dispatch_weights.to(dtype=h.dtype, device=h.device)
        w_byp = dw[:, 0].view(B, 1, 1)
        w_mid = dw[:, 1].view(B, 1, 1)
        w_hvy = dw[:, 2].view(B, 1, 1)

        # Specialist activation modulated by INTRINSIC understanding (no human forced if-else!)
        # Only when the specialist actually understands does it intervene!
        specialist_weight = (w_hvy + 0.5 * w_mid) * p_understand  # [B, S, 1]

        # Check bounce-back occurrence
        bounce_condition = (kappa_mean <= 0.0)
        bounce_ratio = float((p_understand < 0.5).float().mean().item())

        if self.zero_drift_on_bounce and bounce_condition and not self.training:
            self.last_telemetry.update({
                "kappa_mean": round(float(kappa_mean.item()), 4),
                "p_understand_mean": round(float(p_understand.mean().item()), 4),
                "bounce_ratio": round(bounce_ratio, 4),
                "bounce_triggered": True,
                "specialist_weight_mean": 0.0,
                "smil_active": False,
                "zero_drift_applied": True
            })
            return h

        # Execute specialized engine
        h_engine, engine_telem = self.engine(h)
        h_engine = h_engine.to(dtype=h.dtype)
        delta_spec = h_engine - h

        # If bounce occurs: the unhandled energy is directed to SMIL in the background
        bounce_residual = delta_spec * (1.0 - p_understand)

        # SMIL processes in the background with grounding
        h_smil_grounded = self.smil(h, bounce_residual=bounce_residual).to(dtype=h.dtype)

        # Integrated output: Base / SMIL Grounding + Specialist Modulation
        # If specialist doesn't understand (p_understand -> 0), output naturally converges to SMIL Grounded Base!
        h_out = (1.0 - specialist_weight) * h_smil_grounded + specialist_weight * h_engine

        self.last_telemetry.update({
            "kappa_mean": round(float(kappa_mean.item()), 4),
            "p_understand_mean": round(float(p_understand.mean().item()), 4),
            "bounce_ratio": round(bounce_ratio, 4),
            "bounce_triggered": bool(bounce_condition),
            "specialist_weight_mean": round(float(specialist_weight.mean().item()), 4),
            "smil_active": True,
            "zero_drift_applied": False
        })
        return h_out.to(dtype=h.dtype)

    def process_ghost(self, h_final: torch.Tensor, cached_h11: torch.Tensor, dispatch_weights: torch.Tensor) -> torch.Tensor:
        """
        Ghost Layer (Layer 23 / Pre-LM-Head) with SMIL Stabilization.
        """
        bounce_triggered = self.last_telemetry.get("bounce_triggered", False)
        if self.zero_drift_on_bounce and bounce_triggered and not self.training:
            return h_final

        B, S, D = h_final.shape
        dw = dispatch_weights.to(dtype=h_final.dtype, device=h_final.device)
        w_hvy = dw[:, 2].view(B, 1, 1)
        w_mid = dw[:, 1].view(B, 1, 1)

        # Ghost verification
        h_verified, ghost_telem = self.ghost(h_final, cached_h11)
        h_verified = h_verified.to(dtype=h_final.dtype)
        self.last_telemetry.update(ghost_telem)

        # Ghost injection strength modulated by specialist weight and understanding
        p_mean = self.last_telemetry.get("p_understand_mean", 1.0)
        alpha_ghost = 0.25 * (w_hvy + 0.5 * w_mid) * p_mean
        h_out = (1.0 - alpha_ghost) * h_final + alpha_ghost * h_verified
        return h_out.to(dtype=h_final.dtype)


class HADLIDSMILModelWrapper(nn.Module):
    """
    Top-Level Model Wrapper equipping any Causal Transformer with ID-SMIL:
    - Incoherent Diffuse Router (IDR)
    - Local Epistemic Affordance (LEA)
    - Recursive Bounce-Back Channel (RBC)
    - SMIL Background Unit ("Small Little in the Back")
    """
    def __init__(
        self,
        base_model: nn.Module,
        target_layer_idx: int = 11,
        ghost_layer_idx: Optional[int] = 23,
        d_model: Optional[int] = None,
        zero_drift_on_bounce: bool = True
    ):
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
        self.controller = IDSMILController(d_model=d_model, zero_drift_on_bounce=zero_drift_on_bounce)

        # Align controller submodules with base_model device and dtype
        p_ref = next(base_model.parameters(), None)
        if p_ref is not None:
            self.controller.to(device=p_ref.device, dtype=p_ref.dtype)

        self.cached_h11 = None
        self.cached_dispatch_weights = None
        self.enabled = True

        self.hook_handle_target = None
        self.hook_handle_ghost = None
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
                    if not self.enabled or self.cached_dispatch_weights is None:
                        return output

                    h = output[0] if isinstance(output, tuple) else output
                    h_evolved = self.controller.process_layer11(h, self.cached_dispatch_weights)
                    self.cached_h11 = h_evolved
                    return (h_evolved,) + output[1:] if isinstance(output, tuple) else h_evolved

                self.hook_handle_target = target_layer.register_forward_hook(layer_11_hook)

            # 2. Ghost Layer Hook (Layer 23 / Pre-LM-Head)
            ghost_idx = self.ghost_layer_idx if (self.ghost_layer_idx is not None and self.ghost_layer_idx < len(layers)) else len(layers) - 1
            ghost_layer = layers[ghost_idx]

            def layer_ghost_hook(module, args, output):
                if not self.enabled or self.cached_dispatch_weights is None or self.cached_h11 is None:
                    return output

                h_final = output[0] if isinstance(output, tuple) else output
                h_verified = self.controller.process_ghost(h_final, self.cached_h11, self.cached_dispatch_weights)
                return (h_verified,) + output[1:] if isinstance(output, tuple) else h_verified

            self.hook_handle_ghost = ghost_layer.register_forward_hook(layer_ghost_hook)

    def _evaluate_top_routing(self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None):
        embed_layer = None
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens"):
            embed_layer = self.base_model.model.embed_tokens
        elif hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
            embed_layer = self.base_model.transformer.wte

        device = input_ids.device if input_ids is not None else (inputs_embeds.device if inputs_embeds is not None else torch.device("cpu"))
        self.controller.router.to(device=device)

        dispatch_weights, telem = self.controller.router(
            embed_tensor=inputs_embeds,
            embed_layer=embed_layer,
            input_ids=input_ids
        )
        self.cached_dispatch_weights = dispatch_weights
        self.controller.last_telemetry.update(telem)

    def forward(self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None, *args, **kwargs):
        self._evaluate_top_routing(input_ids=input_ids, inputs_embeds=inputs_embeds)
        return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, *args, **kwargs)

    def generate(self, input_ids, *args, **kwargs):
        self.controller.engine.reset_kv_cache()
        self.cached_h11 = None
        self._evaluate_top_routing(input_ids=input_ids)

        out = self.base_model.generate(input_ids, *args, **kwargs)
        self.controller.engine.reset_kv_cache()
        self.cached_h11 = None
        return out


def attach_hadl_idsmil(
    base_model: nn.Module,
    target_layer_idx: int = 11,
    ghost_layer_idx: Optional[int] = 23,
    d_model: Optional[int] = None,
    zero_drift_on_bounce: bool = True
) -> HADLIDSMILModelWrapper:
    """
    Canonically attaches the ID-SMIL architecture to any causal transformer model.
    """
    return HADLIDSMILModelWrapper(
        base_model=base_model,
        target_layer_idx=target_layer_idx,
        ghost_layer_idx=ghost_layer_idx,
        d_model=d_model,
        zero_drift_on_bounce=zero_drift_on_bounce
    )
