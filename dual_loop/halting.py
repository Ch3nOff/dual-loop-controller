import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class EntropyHaltingUnit(nn.Module):
    """
    Predictive Entropy-Based Halting Unit.
    
    Replaces Alex Graves' fragile Adaptive Computation Time (ACT) ponder cost
    penalty with an information-theoretic halting criterion:
    H(p_k) = - sum_v P(v | h_k) * log P(v | h_k)
    
    If predictive entropy falls below the threshold tau (the model is confident),
    pondering halts immediately.
    """
    def __init__(self, entropy_threshold: float = 0.5):
        super().__init__()
        self.entropy_threshold = entropy_threshold

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
        # H(p) = - sum(p * log p)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def should_halt(self, logits: torch.Tensor, threshold: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates whether each sequence in the batch has reached the confidence threshold.
        Args:
            logits: [B, VocabSize]
            threshold: Optional override for entropy threshold.
        Returns:
            halt_mask: [B] boolean tensor (True = halt, False = continue)
            entropy: [B] entropy values
        """
        thresh = self.entropy_threshold if threshold is None else threshold
        entropy = self.calculate_entropy(logits)
        halt_mask = entropy <= thresh
        return halt_mask, entropy
