"""
Probabilistic Cognitive Judge & Hierarchical 2x-Think Controller
===============================================================
Implements:
1. Hierarchical 2x-Think Gating (System 1 Intuition -> Common-Sense Check -> System 2 Escalation).
2. Probabilistic Soft Belief Revision (Prevents 100% hard-locks; allows belief update upon overwhelming evidence).
3. Constraint & Implausibility Scoring (-score penalty for refuted/contradictory hypotheses).
"""

from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from .directional_reservoir import ContextDirectionalRouter, CompactCommonSenseReservoir

class ProbabilisticCognitiveJudge:
    """
    Evaluates candidate hypotheses with soft belief penalties, common-sense margin detection,
    and adaptive deliberation allocation (2x-Think).
    """
    def __init__(
        self,
        cs_margin_threshold: float = 0.35,
        base_lambda: float = 0.85,
        intuitive_lambda: float = 0.20,
        soft_penalty_weight: float = 4.5,
        allow_belief_revision: bool = True,
        polynomial_power: float = 2.0,
        inversion_conviction_threshold: float = 2.2,
        use_directional_reservoir: bool = True
    ):
        self.cs_margin_threshold = cs_margin_threshold
        self.base_lambda = base_lambda
        self.intuitive_lambda = intuitive_lambda
        self.soft_penalty_weight = soft_penalty_weight
        self.allow_belief_revision = allow_belief_revision
        self.polynomial_power = polynomial_power
        self.inversion_conviction_threshold = inversion_conviction_threshold
        self.use_directional_reservoir = use_directional_reservoir
        self.directional_router = ContextDirectionalRouter()
        self.cs_reservoir = CompactCommonSenseReservoir()

    def assess_common_sense_margin(
        self,
        scores_base: Union[List[float], np.ndarray],
        banned_indices: Optional[List[int]] = None
    ) -> Tuple[float, int, bool]:
        """
        Calculates the common-sense margin on eligible (non-banned) candidates.

        Returns:
            margin (float): Difference between top-1 and top-2 unbanned candidates.
            top_idx (int): Index of the highest scoring candidate.
            is_decisive (bool): True if margin >= cs_margin_threshold (intuitive answer is clear).
        """
        s_arr = np.array(scores_base, dtype=np.float32)
        n = len(s_arr)
        banned_set = set(banned_indices or [])
        unbanned_indices = [i for i in range(n) if i not in banned_set]

        if not unbanned_indices:
            # Fallback if all were banned: unban the highest score
            best_overall = int(np.argmax(s_arr))
            return 0.0, best_overall, False

        unbanned_scores = sorted([s_arr[i] for i in unbanned_indices], reverse=True)
        top_unbanned_idx = max(unbanned_indices, key=lambda i: s_arr[i])

        if len(unbanned_scores) == 1:
            margin = 999.0
        else:
            margin = float(unbanned_scores[0] - unbanned_scores[1])

        is_decisive = margin >= self.cs_margin_threshold
        return margin, top_unbanned_idx, is_decisive

    def compute_adaptive_lambda(
        self,
        margin: float,
        is_decisive: bool,
        deliberation_entropy: Optional[float] = None
    ) -> float:
        """
        Dynamically computes the deliberation weight lambda using polynomial modulation:
        Decays smoothly with a polynomial curve as margin grows.
        """
        ratio = float(np.clip(margin / max(1e-4, self.cs_margin_threshold), 0.0, 1.0))
        decay = float((1.0 - ratio) ** self.polynomial_power)
        eff_lambda = self.intuitive_lambda + decay * (self.base_lambda - self.intuitive_lambda)
        return float(eff_lambda)

    def judge_and_fuse(
        self,
        scores_base: Union[List[float], np.ndarray],
        scores_delib: Union[List[float], np.ndarray],
        labels: List[str],
        banned_labels: Optional[List[str]] = None,
        confidence_map: Optional[Dict[str, float]] = None,
        prompt: Optional[str] = None,
        choices: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes hierarchical judging, polynomial modulation, inverse deliberative fallback,
        directional context routing, common-sense grounding (f o g), and soft belief score fusion.
        """
        s_base = np.array(scores_base, dtype=np.float32)
        s_delib = np.array(scores_delib, dtype=np.float32)
        n = len(labels)

        banned_labels_set = set(banned_labels or [])
        banned_indices = [i for i, lbl in enumerate(labels) if lbl in banned_labels_set]

        # Stage 1: Assess Common-Sense Margin on unbanned candidates
        margin, top_idx, is_decisive = self.assess_common_sense_margin(s_base, banned_indices)

        # Stage 2: Directional Context Routing (Graph Direction: UP vs DOWN)
        direction = "UP_SCIENTIFIC"
        rho_direction = 1.0
        alpha_cs = 0.0
        cs_deltas = np.zeros(n, dtype=np.float32)

        if prompt is not None and self.use_directional_reservoir:
            route_info = self.directional_router.route_context(prompt)
            rho_direction = route_info["rho"]
            direction = route_info["direction"]
            alpha_cs = route_info["alpha_cs"]

            if (direction == "DOWN_COMMONSENSE" or not is_decisive) and choices is not None:
                raw_deltas = self.cs_reservoir.compute_grounding_deltas(prompt, choices)
                cs_deltas = np.array(raw_deltas, dtype=np.float32)

        # Stage 3: Polynomial Adaptive Lambda Modulation
        eff_lambda = self.compute_adaptive_lambda(margin, is_decisive)

        # Stage 4: Deliberative Inversion Fallback (Sistem Invers Balik ke Asisten)
        # If the Assistant (Deliberation) has an overwhelming conviction on a different candidate:
        unbanned_indices = [i for i in range(n) if i not in banned_indices]
        inversion_triggered = False
        if unbanned_indices:
            top_delib_idx = max(unbanned_indices, key=lambda i: s_delib[i])
            if top_delib_idx != top_idx:
                delib_conviction = float(s_delib[top_delib_idx] - s_delib[top_idx])
                conv_thresh = self.inversion_conviction_threshold if direction == "UP_SCIENTIFIC" else self.inversion_conviction_threshold * 1.5
                if delib_conviction >= conv_thresh:
                    eff_lambda = self.base_lambda
                    inversion_triggered = True

        # Stage 5: Soft Penalties & Score Fusion with Directional Grounding
        fused = np.zeros(n, dtype=np.float32)
        for i in range(n):
            lbl = labels[i]
            base_val = s_base[i]
            delib_val = s_delib[i]
            c_val = (1.0 - eff_lambda) * base_val + eff_lambda * delib_val

            # Inject Common-Sense Grounding Delta (f o g) if directed downwards
            if self.use_directional_reservoir and alpha_cs > 0.0:
                c_val += alpha_cs * cs_deltas[i]

            # Apply soft penalty if label was logged as wrong
            if lbl in banned_labels_set:
                penalty = self.soft_penalty_weight
                if confidence_map and lbl in confidence_map:
                    penalty *= confidence_map[lbl]
                c_val -= penalty

            fused[i] = c_val

        # Stage 6: Probabilistic Decision (No 100% hard-locks; decisions flow from fused scores)
        pred_idx = int(np.argmax(fused))
        is_belief_revision = labels[pred_idx] in banned_labels_set

        return {
            "pred_idx": pred_idx,
            "pred_label": labels[pred_idx],
            "margin": margin,
            "is_decisive": is_decisive,
            "effective_lambda": eff_lambda,
            "is_belief_revision": is_belief_revision,
            "direction": direction,
            "rho": rho_direction,
            "alpha_cs": alpha_cs,
            "cs_deltas": cs_deltas.tolist(),
            "fused_scores": fused.tolist()
        }
