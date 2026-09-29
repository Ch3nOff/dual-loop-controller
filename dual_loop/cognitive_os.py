"""
Unified Cognitive Operating System (HADL v3.0 Core Engine)
=========================================================
Synthesizes Mathematics, Cybernetics, and Hardware Computer Architecture into
the 5 Computational Brain Organs:

1. Multi-Time-Scale Dynamics (Dual-Speed Inference & Cognitive Conflict Gate)
2. Allostasis & Active Inference (Vital State Vector S_t & Expected Free Energy G(pi))
3. Global Workspace Theory (GWT L2 SRAM spotlight vs VRAM passive memory)
4. Fast Synaptic Plasticity & Sleep Consolidation (Hebbian fast weights -> SVD distillation)
5. Sheaf-Theoretic Invariant Firewall (Sub-0.05ms Prefrontal Executive Inhibition)
"""

import time
import math
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union

from .adapters.universal_adapter import UniversalDualLoopAdapter, CANONICAL_DIM
from .allostasis import AllostaticEnergyModulator
from .curiosity_daemon import AutonomousDaemonController
from .sleep_consolidation import SleepPhaseConsolidationEngine
from .firewall import SheafInvariantFirewall


class CognitiveConflictEvaluator(nn.Module):
    """
    Organ 1: Multi-Time-Scale Dynamics Dispatcher.
    Evaluates logit margin between top-1 and top-2 predictions.
    When margin is large, System 1 proceeds with sub-microsecond bypass.
    When margin is narrow (ambiguity / dilemma), System 2 latent deliberation is triggered.
    """
    def __init__(self, tau_margin: float = 1.20, tau_entropy: float = 1.80):
        super().__init__()
        self.tau_margin = tau_margin
        self.tau_entropy = tau_entropy

    def evaluate_conflict(self, logits: torch.Tensor) -> Tuple[bool, Dict[str, float]]:
        """
        Args:
            logits: [B, Vocab] or [B, S, Vocab]
        Returns:
            should_deliberate: boolean flag (True -> activate System 2)
            telemetry: margin and entropy values
        """
        # Focus on last token
        if logits.dim() == 3:
            cur_logits = logits[:, -1, :]
        else:
            cur_logits = logits

        top2_vals, _ = torch.topk(cur_logits, k=min(2, cur_logits.shape[-1]), dim=-1)
        margin = (top2_vals[:, 0] - top2_vals[:, 1]).mean().item()
        
        # Softmax entropy
        probs = torch.softmax(cur_logits, dim=-1)
        log_probs = torch.log_softmax(cur_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()

        # Trigger System 2 when margin is low OR entropy is high
        should_deliberate = (margin < self.tau_margin) or (entropy > self.tau_entropy)

        return should_deliberate, {
            "logit_margin": float(margin),
            "entropy": float(entropy),
            "tau_margin": float(self.tau_margin),
            "tau_entropy": float(self.tau_entropy)
        }


class VitalStateVector:
    """
    Organ 2: Vital State Vector S_t = [Energy, Entropy u, Drift kappa, Memory Saturation]^T.
    Governs biological allostasis and intrinsic drives (curiosity vs rest).
    """
    def __init__(
        self,
        energy: float = 1.0,
        entropy: float = 0.20,
        drift: float = 0.0,
        saturation: float = 0.0
    ):
        self.energy = energy            # E in [0, 1]
        self.entropy = entropy          # u in [0, 1]
        self.drift = drift              # kappa >= 0
        self.saturation = saturation    # M in [0, 1]

    def update(
        self,
        energy_drain: float = 0.02,
        new_entropy: Optional[float] = None,
        observed_drift: float = 0.0,
        added_memory_items: int = 0,
        max_items: int = 100
    ):
        self.energy = max(0.05, min(1.0, self.energy - energy_drain))
        if new_entropy is not None:
            self.entropy = 0.85 * self.entropy + 0.15 * new_entropy
        self.drift = 0.90 * self.drift + 0.10 * observed_drift
        self.saturation = min(1.0, self.saturation + (added_memory_items / max_items))

    def recharge_during_sleep(self):
        """Restores energy and flushes memory saturation during sleep phase."""
        self.energy = 1.0
        self.saturation = 0.0
        self.entropy = 0.10

    def compute_expected_free_energy(self, action_cost: float = 0.1) -> float:
        """
        G(pi) = Pragmatic (Survival) + Epistemic (Curiosity).
        Lower is better.
        """
        pragmatic_risk = (1.0 - self.energy) + self.drift
        epistemic_value = self.entropy * 0.5  # Higher entropy drives curiosity
        g_val = pragmatic_risk - epistemic_value + action_cost
        return float(g_val)

    def to_dict(self) -> Dict[str, float]:
        return {
            "energy": self.energy,
            "entropy": self.entropy,
            "drift": self.drift,
            "memory_saturation": self.saturation,
            "expected_free_energy": self.compute_expected_free_energy()
        }


class UnifiedCognitiveOS(nn.Module):
    """
    Unified Cognitive Operating System (HADL v3.0).
    Orchestrates the 5 Organs across any base Transformer model.
    """
    def __init__(
        self,
        d_native: int,
        d_canonical: int = CANONICAL_DIM,
        num_slots: int = 16,
        max_ponder_steps: int = 3,
        tau_conflict_margin: float = 1.20
    ):
        super().__init__()
        self.d_native = d_native
        self.d_canonical = d_canonical
        
        # Organ 1: Multi-Time-Scale Dynamics
        self.conflict_evaluator = CognitiveConflictEvaluator(tau_margin=tau_conflict_margin)
        
        # Organ 2: Allostasis & Active Inference
        self.vital_state = VitalStateVector()
        self.allostatic_modulator = AllostaticEnergyModulator(d_model=d_canonical)
        
        # Organ 3 & 4: Universal Canonical System 2 Adapter with GWT & Sleep Replay
        self.universal_adapter = UniversalDualLoopAdapter(
            d_native=d_native,
            d_canonical=d_canonical,
            num_slots=num_slots,
            max_ponder_steps=max_ponder_steps,
            enable_plasticity=True,
            enable_firewall=True,
            enable_sleep_consolidation=True
        )
        
        # Organ 5: Sheaf Invariant Firewall
        self.firewall = self.universal_adapter.firewall

        # Autonomous Curiosity Daemon for background contemplation
        self.curiosity_daemon = AutonomousDaemonController(d_model=d_canonical)

    def execute_cognitive_step(
        self,
        h_native: torch.Tensor,
        current_logits: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes an integrated cognitive inference step.
        Dispatches between Fast System 1 and Slow System 2 based on Cognitive Conflict.
        """
        t0 = time.perf_counter()
        
        # 1. Multi-Time-Scale Conflict Check (Organ 1)
        should_deliberate = True
        conflict_telem = {}
        if current_logits is not None:
            should_deliberate, conflict_telem = self.conflict_evaluator.evaluate_conflict(current_logits)

        # Update vital state
        self.vital_state.update(
            energy_drain=0.01 if should_deliberate else 0.001,
            new_entropy=conflict_telem.get("entropy", 0.20)
        )

        # 2. Latent Deliberation Dispatch
        k_steps = self.universal_adapter.max_ponder_steps if should_deliberate else 0
        enhanced_h, adapter_telem = self.universal_adapter(h_native, k_steps=k_steps, context=context)

        # Check if sleep cycle is needed due to memory saturation
        sleep_triggered = False
        sleep_telem = {}
        if self.vital_state.saturation >= 0.90:
            sleep_telem = self.trigger_sleep_consolidation()
            sleep_triggered = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        telemetry = {
            "mode": "SLOW_SYSTEM_2" if should_deliberate else "FAST_SYSTEM_1_BYPASS",
            "k_steps": k_steps,
            "conflict": conflict_telem,
            "vital_state": self.vital_state.to_dict(),
            "adapter": adapter_telem,
            "sleep_cycle_executed": sleep_triggered,
            "sleep_telemetry": sleep_telem,
            "total_latency_ms": elapsed_ms
        }

        return enhanced_h, telemetry

    def trigger_sleep_consolidation(self) -> Dict[str, Any]:
        """Organ 4: Executes complete offline sleep-phase consolidation."""
        telem = self.universal_adapter.trigger_sleep_cycle()
        self.vital_state.recharge_during_sleep()
        return telem
