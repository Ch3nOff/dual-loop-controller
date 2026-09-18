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
        allow_belief_revision: bool = True
    ):
        self.cs_margin_threshold = cs_margin_threshold
        self.base_lambda = base_lambda
        self.intuitive_lambda = intuitive_lambda
        self.soft_penalty_weight = soft_penalty_weight
        self.allow_belief_revision = allow_belief_revision

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
        Dynamically computes the deliberation weight lambda:
        - If intuitive common-sense is decisive (e.g. margin >= threshold), set low lambda
          to avoid overthinking on simple common-sense questions.
        - If margin is narrow (ambiguity or multihop science reasoning), escalate to high lambda.
        """
        if is_decisive:
            return self.intuitive_lambda

        # Smooth scaling between intuitive_lambda and base_lambda based on margin
        scale = np.clip(1.0 - (margin / max(1e-4, self.cs_margin_threshold)), 0.0, 1.0)
        eff_lambda = self.intuitive_lambda + scale * (self.base_lambda - self.intuitive_lambda)
        return float(eff_lambda)

    def judge_and_fuse(
        self,
        scores_base: Union[List[float], np.ndarray],
        scores_delib: Union[List[float], np.ndarray],
        labels: List[str],
        banned_labels: Optional[List[str]] = None,
        confidence_map: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Executes hierarchical judging, soft belief penalty application, and score fusion.

        Args:
            scores_base: Raw likelihoods from System 1.
            scores_delib: Deliberation likelihoods from System 2.
            labels: List of candidate labels (['A', 'B', 'C', 'D']).
            banned_labels: Labels recorded in wrong log bank.
            confidence_map: Optional per-label penalty weights (defaults to soft_penalty_weight).

        Returns:
            Dict containing predicted index, predicted label, effective lambda, and fused scores.
        """
        s_base = np.array(scores_base, dtype=np.float32)
        s_delib = np.array(scores_delib, dtype=np.float32)
        n = len(labels)

        banned_labels_set = set(banned_labels or [])
        banned_indices = [i for i, lbl in enumerate(labels) if lbl in banned_labels_set]

        # Stage 1: Assess Common-Sense Margin on unbanned candidates
        margin, top_idx, is_decisive = self.assess_common_sense_margin(s_base, banned_indices)

        # Stage 2: Determine Adaptive Deliberation Lambda (2x-Think Gating)
        eff_lambda = self.compute_adaptive_lambda(margin, is_decisive)

        # Stage 3: Soft Penalties & Score Fusion (Anti-Hardlock)
        fused = np.zeros(n, dtype=np.float32)
        for i in range(n):
            lbl = labels[i]
            base_val = s_base[i]
            delib_val = s_delib[i]
            c_val = (1.0 - eff_lambda) * base_val + eff_lambda * delib_val

            # Apply soft penalty if label was logged as wrong
            if lbl in banned_labels_set:
                penalty = self.soft_penalty_weight
                if confidence_map and lbl in confidence_map:
                    penalty *= confidence_map[lbl]
                c_val -= penalty

            fused[i] = c_val

        # Stage 4: Probabilistic Decision (No 100% hard-locks; decisions flow from fused scores)
        pred_idx = int(np.argmax(fused))
        is_belief_revision = labels[pred_idx] in banned_labels_set

        return {
            "pred_idx": pred_idx,
            "pred_label": labels[pred_idx],
            "margin": margin,
            "is_decisive": is_decisive,
            "effective_lambda": eff_lambda,
            "is_belief_revision": is_belief_revision,
            "fused_scores": fused.tolist()
        }
