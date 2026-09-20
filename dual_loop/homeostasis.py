import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List

class HomeostaticDriveEngine(nn.Module):
    """
    Biological Homeostatic Drive Engine (Drive-Reduction Theory).
    
    Maintains a continuous dynamic Internal Physiological State Vector:
        S_t = [S_1, S_2, S_3, S_4]^T in R^4
    where:
        S_1 : Compute Budget / Energy remaining in [0, 1] (depletes with deliberation steps, recovers on bypass)
        S_2 : Epistemic Entropy / Uncertainty vacuity u(x) in [0, 1] (from Dirichlet decomposition)
        S_3 : Semantic Coherence / Drift distance from core anchor ||h_t - h_anchor|| / ||h_anchor||
        S_4 : Working Memory Saturation ratio in [0, 1] (occupied slots / capacity)
        
    Ideal Homeostatic Setpoint:
        S* = [1.0, 0.05, 0.0, 0.25]^T
        
    Drive Distance Function:
        D(S_t) = sum_{i=1}^4 omega_i * |S_{t, i} - S_i^*|^p
        
    Intrinsic Motivation (Free Energy Reduction):
        R_internal = D(S_t) - D(S_{t+1})
        The agent is intrinsically motivated to act (ponder) ONLY when doing so
        reduces overall drive error towards S*.
    """
    def __init__(
        self,
        p_norm: float = 2.0,
        weights: Optional[List[float]] = None,
        energy_cost_per_step: float = 0.15,
        energy_recovery_rate: float = 0.05,
        entropy_reduction_rate: float = 0.60
    ):
        super().__init__()
        self.p_norm = float(p_norm)
        # Default weights prioritize Epistemic Entropy (S_2) and Semantic Drift (S_3) over Energy (S_1)
        w = weights if weights is not None else [1.0, 2.5, 2.0, 0.5]
        self.register_buffer("weights", torch.tensor(w, dtype=torch.float32))
        self.register_buffer("S_star", torch.tensor([1.0, 0.05, 0.0, 0.25], dtype=torch.float32))
        
        self.energy_cost_per_step = energy_cost_per_step
        self.energy_recovery_rate = energy_recovery_rate
        self.entropy_reduction_rate = entropy_reduction_rate
        
        # Stateful energy tracker for streaming inference
        self.current_energy: float = 1.0

    def reset_physiological_state(self):
        """Resets dynamic internal state to full homeostasis."""
        self.current_energy = 1.0

    def compute_drive(self, S: torch.Tensor) -> torch.Tensor:
        """
        Computes scalar drive penalty D(S).
        Args:
            S: [..., 4] Physiological state vector.
        Returns:
            D: [...] Scalar drive distance.
        """
        w = self.weights.to(device=S.device, dtype=S.dtype)
        s_star = self.S_star.to(device=S.device, dtype=S.dtype)
        diff = torch.abs(S - s_star)
        if self.p_norm == 2.0:
            return torch.sum(w * (diff ** 2), dim=-1)
        else:
            return torch.sum(w * (diff ** self.p_norm), dim=-1)

    def evaluate_drive_reduction(
        self,
        current_u: float,
        current_drift: float = 0.0,
        memory_saturation: float = 0.25,
        candidate_k_steps: int = 2
    ) -> Tuple[float, float, float]:
        """
        Simulates the homeostatic drive reduction Delta D for a prospective deliberation step.
        Returns:
            delta_D: D(S_now) - D(S_prospective). Positive means deliberation is advantageous.
            D_now: Current drive penalty.
            D_prospective: Prospective drive penalty after k steps.
        """
        device = self.weights.device
        dtype = self.weights.dtype
        
        # 1. State now
        S_now = torch.tensor([self.current_energy, current_u, current_drift, memory_saturation], device=device, dtype=dtype)
        D_now = float(self.compute_drive(S_now).item())
        
        # 2. Prospective state if candidate_k_steps are taken
        if candidate_k_steps == 0:
            # Bypass mode: recover energy slightly, entropy remains unchanged
            prospective_energy = min(1.0, self.current_energy + self.energy_recovery_rate)
            prospective_u = current_u
            prospective_drift = current_drift
        else:
            # Deliberation mode: consumes energy, but actively reduces epistemic uncertainty and drift
            prospective_energy = max(0.0, self.current_energy - (self.energy_cost_per_step * candidate_k_steps))
            reduction_factor = (1.0 - self.entropy_reduction_rate) ** candidate_k_steps
            prospective_u = current_u * reduction_factor
            prospective_drift = current_drift * 0.5
            
        S_prospective = torch.tensor([prospective_energy, prospective_u, prospective_drift, memory_saturation], device=device, dtype=dtype)
        D_prospective = float(self.compute_drive(S_prospective).item())
        
        delta_D = D_now - D_prospective
        return delta_D, D_now, D_prospective

    def update_state_after_action(self, k_steps_taken: int):
        """Updates internal energy level based on actual action taken."""
        if k_steps_taken == 0:
            self.current_energy = min(1.0, self.current_energy + self.energy_recovery_rate)
        else:
            self.current_energy = max(0.0, self.current_energy - (self.energy_cost_per_step * k_steps_taken))


class ActiveInferencePolicyRouter(nn.Module):
    """
    Active Inference & Expected Free Energy Policy Router (Friston Principle).
    
    Given the current hidden state h and epistemic uncertainty u, evaluates candidate policies:
        pi_0: System 1 Bypass (k=0, instant execution, preserves syntax & energy)
        pi_1: System 2 Focused Deliberation (k=1..2, resolves ambiguity in reasoning)
        pi_2: The Brain Sandbox (k=3..4, multi-stage mental simulation for complex coding/scripts)
        
    Expected Free Energy:
        G(pi) = - E_Q[ln P(o_tau | C)] (Pragmatic Value: Homeostasis & Constraints)
                - E_Q[ln Q(s_tau | o_tau, pi) - ln Q(s_tau | pi)] (Epistemic Value: Information Gain)
    """
    def __init__(
        self,
        d_model: int,
        tau_syntax_certainty: float = 0.20,
        tau_script_trigger: float = 0.65
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_syntax_certainty = float(tau_syntax_certainty)
        self.tau_script_trigger = float(tau_script_trigger)
        
        self.homeostasis = HomeostaticDriveEngine()
        
        # Lightweight domain classifier: [Text/Reasoning, Complex Code/Script]
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2)
        )
        # Small scale init for stable classifier
        nn.init.normal_(self.domain_classifier[-1].weight, std=0.01)
        nn.init.zeros_(self.domain_classifier[-1].bias)

    def select_policy(
        self,
        h_current: torch.Tensor,
        vacuity_u: Optional[float] = None,
        is_token_streaming: bool = False,
        base_k_steps: int = 2
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Selects optimal deliberation steps k* by minimizing Expected Free Energy and reducing Drive.
        
        Args:
            h_current: [B, S, D] Current hidden states.
            vacuity_u: Scalar epistemic vacuity in [0, 1].
            is_token_streaming: True if decoding a single token (S == 1) during autoregressive generation.
            base_k_steps: Configured default deliberation steps (e.g. 2).
            
        Returns:
            optimal_k: Selected deliberation steps in {0, 1, 2, 3, 4}.
            telemetry: Detailed physiological metrics.
        """
        u_val = float(vacuity_u) if vacuity_u is not None else 0.50
        
        # 1. STREAMING TOKEN GENERATION RULE:
        # In autoregressive decoding, deliberative planning was already executed at the prompt phase.
        # Intermediate token generation proceeds via fast, fluent System 1 (k=0) unless an extreme
        # anomaly occurs (vacuity u >= 0.85). This completely preserves syntax deterministic fluency
        # and eliminates CPU latency penalties on code syntax (;, {}, div).
        if is_token_streaming and u_val < 0.85:
            self.homeostasis.update_state_after_action(0)
            return 0, {
                "policy": "pi_0_syntax_bypass",
                "optimal_k": 0,
                "reason": "fluent_streaming_mode",
                "energy": self.homeostasis.current_energy,
                "vacuity_u": u_val,
                "is_streaming": True
            }
            
        # 2. Evaluate script detection on anchor/prompt tokens
        h_anchor = h_current[:, -1, :] if h_current.dim() == 3 else h_current
        with torch.no_grad():
            logits = self.domain_classifier(h_anchor)
            probs = F.softmax(logits, dim=-1)
            p_script = float(probs[0, 1].item())
            
        is_script_task = (p_script >= self.tau_script_trigger) or (not is_token_streaming and u_val >= self.tau_script_trigger)
        
        # 3. Evaluate Homeostatic Drive Reduction for prospective policies
        delta_D_delib, d_now, d_delib = self.homeostasis.evaluate_drive_reduction(
            current_u=u_val,
            candidate_k_steps=base_k_steps
        )
        
        delta_D_sandbox, _, d_sandbox = self.homeostasis.evaluate_drive_reduction(
            current_u=u_val,
            candidate_k_steps=3
        )
        
        # 4. Policy Selection based on Expected Free Energy & Drive Reduction
        if is_script_task and not is_token_streaming:
            # Deep script planning: The Brain Sandbox (k=3)
            optimal_k = 3
            policy_name = "pi_2_brain_sandbox"
        elif delta_D_delib > 0.05:
            # Pondering reliably resolves high ambiguity
            optimal_k = base_k_steps
            policy_name = "pi_1_focused_deliberation"
        else:
            # Pondering costs more energy than it gains in entropy reduction: Bypass!
            optimal_k = 0
            policy_name = "pi_0_homeostatic_bypass"
            
        self.homeostasis.update_state_after_action(optimal_k)
        
        telemetry = {
            "policy": policy_name,
            "optimal_k": optimal_k,
            "p_script": p_script,
            "is_script_task": is_script_task,
            "delta_D_delib": delta_D_delib,
            "delta_D_sandbox": delta_D_sandbox,
            "D_now": d_now,
            "energy_remaining": self.homeostasis.current_energy,
            "vacuity_u": u_val,
            "is_streaming": is_token_streaming
        }
        return optimal_k, telemetry
