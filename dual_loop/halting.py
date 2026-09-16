import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class EntropyHaltingUnit(nn.Module):
    """
    Predictive Entropy-Based Halting Unit.
    
    Evaluates Shannon entropy:
    H(p_k) = - sum_v P(v | h_k) * log P(v | h_k)
    
    Supports:
    1. Absolute thresholding (requires empirical calibration against actual model entropy distribution).
    2. Relative entropy drop: Halts when entropy delta |H_k - H_{k-1}| < delta_stop (convergence).
    3. Calibration utility to set threshold dynamically based on validation logits.
    """
    def __init__(self, entropy_threshold: float = 1.30, delta_threshold: float = 0.02):
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.delta_threshold = delta_threshold

    def calculate_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates Shannon entropy in nats from unnormalized logits.
        Args:
            logits: [B, VocabSize]
        Returns:
            entropy: [B] scalar entropy per sample
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def calibrate_threshold(self, sample_logits: torch.Tensor, percentile: float = 75.0):
        """
        Calibrates the entropy threshold to match the model's actual empirical
        predictive distribution (defaults to 75th percentile / upper quartile of validation entropy).
        """
        with torch.no_grad():
            entropies = self.calculate_entropy(sample_logits)
            calibrated_val = torch.quantile(entropies, percentile / 100.0).item()
            self.entropy_threshold = calibrated_val
            return calibrated_val

    def should_halt(
        self,
        logits: torch.Tensor,
        prev_entropy: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates whether sequences in the batch should halt based on calibrated threshold
        or relative convergence.
        """
        thresh = self.entropy_threshold if threshold is None else threshold
        entropy = self.calculate_entropy(logits)
        
        # Criterion 1: Confidence threshold
        confident_mask = entropy <= thresh
        
        # Criterion 2: Relative convergence (entropy has stopped changing significantly)
        if prev_entropy is not None:
            delta = torch.abs(prev_entropy - entropy)
            converged_mask = delta <= self.delta_threshold
            halt_mask = confident_mask | converged_mask
        else:
            halt_mask = confident_mask
            
        return halt_mask, entropy
