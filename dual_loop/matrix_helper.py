"""
Cognitive Matrix Question Helper (Elimination-by-Aspects & Subspace Pruning)
===========================================================================
Implements the 2-Bench Cognitive Refiner:
- Bench 1 (Raw Screening): Computes candidate log-likelihoods, derives softmax confidence
  percentages, and logs distractor candidates into an Evidence & Elimination Matrix.
- Bench 2 (Dual Loop Deliberation): Reads the Matrix Question Helper, calculates and prunes
  the 'wrong logs' (distractors with prob < tau_elim), and focuses System 2 latent
  cross-attention exclusively on surviving contenders.

Mathematical Foundations:
1. Amos Tversky's Elimination-by-Aspects (EBA) Model (Psychological Review, 1972).
2. Subspace Attention Sharpening: Prevents attention dilution over multi-choice distractors.
3. Evidential Residual Gating: Balances prior confidence with deliberated proof conviction.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

class CognitiveMatrixHelper:
    """
    Evidence Matrix & Distractor Log Eliminator for Multi-Choice Reasoning.
    """
    def __init__(
        self,
        default_temperature: float = 0.50,
        elimination_threshold: float = 0.12,
        min_survivors: int = 2
    ):
        self.default_temperature = default_temperature
        self.elimination_threshold = elimination_threshold
        self.min_survivors = min_survivors

    def build_evidence_matrix(
        self,
        scores_base: Union[List[float], np.ndarray, torch.Tensor],
        labels: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        elim_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Builds the Evidence & Elimination Matrix from Bench 1 raw log-likelihood scores.

        Args:
            scores_base: Raw log-likelihood scores for each candidate choice.
            labels: Optional choice labels (e.g. ['A', 'B', 'C', 'D']).
            temperature: Softmax temperature for probability calculation.
            elim_threshold: Minimum probability to avoid elimination.

        Returns:
            Dict containing:
                - 'scores': Raw scores array.
                - 'probs': Softmax probabilities array.
                - 'elim_mask': Boolean array where True indicates eliminated distractor.
                - 'survivors': Integer indices of non-eliminated candidates.
                - 'eliminated_labels': List of labels for eliminated candidates.
                - 'survivor_labels': List of labels for surviving candidates.
                - 'top1_idx': Index of predicted choice in Bench 1.
                - 'margin': Score gap between top-1 and top-2 candidate.
                - 'entropy': Predictive entropy of candidate distribution.
        """
        if isinstance(scores_base, torch.Tensor):
            s_arr = scores_base.detach().cpu().float().numpy()
        else:
            s_arr = np.array(scores_base, dtype=np.float32)

        n_cands = len(s_arr)
        if labels is None:
            labels = [chr(65 + i) for i in range(n_cands)]

        temp = temperature if temperature is not None else self.default_temperature
        tau_elim = elim_threshold if elim_threshold is not None else self.elimination_threshold

        # Compute stable softmax probabilities
        shifted = (s_arr - np.max(s_arr)) / max(1e-5, temp)
        exp_s = np.exp(shifted)
        probs = exp_s / np.sum(exp_s)

        # Predictive Entropy
        safe_probs = np.clip(probs, 1e-12, 1.0)
        entropy = float(-np.sum(safe_probs * np.log(safe_probs)))

        # Sorted metrics
        sorted_indices = np.argsort(s_arr)[::-1]
        sorted_scores = s_arr[sorted_indices]
        margin = float(sorted_scores[0] - sorted_scores[1]) if n_cands > 1 else 999.0

        # Elimination logic (Eliminate wrong logs)
        if n_cands > self.min_survivors:
            # Ensure at least min_survivors survive
            second_highest_prob = probs[sorted_indices[self.min_survivors - 1]]
            effective_cutoff = min(tau_elim, float(second_highest_prob))
            elim_mask = probs < effective_cutoff
            survivors = np.where(~elim_mask)[0]
            if len(survivors) < self.min_survivors:
                survivors = sorted_indices[:self.min_survivors]
                elim_mask = np.ones(n_cands, dtype=bool)
                elim_mask[survivors] = False
        else:
            elim_mask = np.zeros(n_cands, dtype=bool)
            survivors = np.arange(n_cands)

        survivors = np.sort(survivors)
        eliminated_indices = np.where(elim_mask)[0]

        return {
            "scores": s_arr,
            "probs": probs,
            "elim_mask": elim_mask,
            "survivors": survivors,
            "eliminated_indices": eliminated_indices,
            "eliminated_labels": [labels[i] for i in eliminated_indices],
            "survivor_labels": [labels[i] for i in survivors],
            "top1_idx": int(sorted_indices[0]),
            "margin": margin,
            "entropy": entropy,
            "n_cands": n_cands
        }

    def filter_candidate_embeddings(
        self,
        candidate_embeds: torch.Tensor,
        survivor_indices: np.ndarray
    ) -> torch.Tensor:
        """
        Extracts embedding vectors strictly for surviving candidates.
        
        Args:
            candidate_embeds: Tensor of shape [B, n_cands, L_c, D] or [B, n_cands, D].
            survivor_indices: Indices of non-eliminated candidates.
            
        Returns:
            Filtered candidate embeddings tensor for System 2 cross-attention.
        """
        idx_tensor = torch.tensor(survivor_indices, device=candidate_embeds.device, dtype=torch.long)
        return torch.index_select(candidate_embeds, dim=1, index=idx_tensor)

    def fuse_scores(
        self,
        scores_base: Union[List[float], np.ndarray],
        scores_delib_survivors: Union[List[float], np.ndarray],
        survivor_indices: np.ndarray,
        lambda_delib: float = 0.85,
        eliminated_fill_value: float = -1e9
    ) -> np.ndarray:
        """
        Merges Bench 1 log-likelihoods with Bench 2 deliberation scores on surviving candidates.
        Eliminated candidates are assigned eliminated_fill_value (-inf).
        """
        s_base = np.array(scores_base, dtype=np.float32)
        s_delib_surv = np.array(scores_delib_survivors, dtype=np.float32)

        final_scores = np.full_like(s_base, fill_value=eliminated_fill_value)

        for local_idx, orig_idx in enumerate(survivor_indices):
            # Convex combination on surviving candidates
            final_scores[orig_idx] = (1.0 - lambda_delib) * s_base[orig_idx] + lambda_delib * s_delib_surv[local_idx]

        return final_scores
