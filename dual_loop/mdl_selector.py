import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional

class NeuroSymbolicMDLSelector(nn.Module):
    """
    Neuro-Symbolic Minimum Description Length (MDL) Selector.
    
    Evaluates candidate thought trajectories or latent plans using the MDL principle:
        Score(p*) = Length(p*) + lambda * Error(Constraints | p*)
        
    Where:
        Length(p*): Structural complexity of the proposed latent plan
                    (measured via effective latent rank / sparsity / entropy).
        Error(Constraints | p*): Reconstruction discrepancy against grounded context slots.
        
    Penalizes bloated, repetitive, or over-engineered code architectures,
    favoring the most compact, elegant, and parsimonious solution (Occam's Razor).
    """
    def __init__(
        self,
        d_model: int,
        lambda_error: float = 2.0,
        complexity_weight: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.lambda_error = float(lambda_error)
        self.complexity_weight = float(complexity_weight)
        
        # Discrepancy evaluation projection
        self.eval_proj = nn.Linear(d_model, d_model)

    def compute_plan_complexity(self, thought_plan: torch.Tensor) -> torch.Tensor:
        """
        Computes description length / structural complexity of thought representation.
        Args:
            thought_plan: [B, L, D]
        Returns:
            complexity: [B] Scalar complexity score.
        """
        # 1. Sparsity / L1 energy norm
        l1_cost = torch.norm(thought_plan, p=1, dim=-1).mean(dim=-1) # [B]
        
        # 2. Latent variance / diversity across thought tokens (penalizes chaotic high-entropy wandering)
        variance_cost = torch.var(thought_plan, dim=1).mean(dim=-1) # [B]
        
        return (l1_cost * 0.05 + variance_cost * 1.5) * self.complexity_weight

    def compute_constraint_error(
        self,
        thought_plan: torch.Tensor,
        context_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes discrepancy error between candidate plan and context constraints.
        Args:
            thought_plan: [B, L, D]
            context_memory: [B, M, D]
        Returns:
            error: [B] Scalar constraint violation error.
        """
        # Cross-attention matching between thought plan and context slots
        B, L, D = thought_plan.shape
        _, M, _ = context_memory.shape
        
        proj_t = self.eval_proj(thought_plan) # [B, L, D]
        sim = torch.bmm(proj_t, context_memory.transpose(1, 2)) / (D ** 0.5) # [B, L, M]
        attn = F.softmax(sim, dim=-1)
        reconstructed = torch.bmm(attn, context_memory) # [B, L, D]
        
        # Discrepancy norm
        error = torch.norm(thought_plan - reconstructed, dim=-1).mean(dim=-1) # [B]
        return error

    def score_candidate(
        self,
        thought_plan: torch.Tensor,
        context_memory: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes total MDL score for a single thought plan.
        Returns:
            total_mdl: [B] Scalar score (lower is better).
            telemetry: Detailed metrics.
        """
        complexity = self.compute_plan_complexity(thought_plan)
        error = self.compute_constraint_error(thought_plan, context_memory)
        total_mdl = complexity + (self.lambda_error * error)
        
        telemetry = {
            "complexity": float(complexity.mean().item()),
            "constraint_error": float(error.mean().item()),
            "total_mdl": float(total_mdl.mean().item())
        }
        return total_mdl, telemetry

    def select_best_candidate(
        self,
        candidate_plans: List[torch.Tensor],
        context_memory: torch.Tensor
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Selects the most parsimonious plan from a list of candidate rollouts.
        Args:
            candidate_plans: List of [B, L, D] candidate thought plans.
            context_memory: [B, M, D] Grounded context slots.
        Returns:
            best_plan: [B, L, D]
            best_idx: Index of best candidate.
            telemetry: Selection metrics.
        """
        scores = []
        all_telem = []
        for i, plan in enumerate(candidate_plans):
            score, telem = self.score_candidate(plan, context_memory)
            scores.append(score)
            all_telem.append(telem)
            
        stacked_scores = torch.stack(scores, dim=0) # [NumCandidates, B]
        best_idx = int(torch.argmin(stacked_scores.mean(dim=-1)).item())
        best_plan = candidate_plans[best_idx]
        
        telemetry = {
            "selected_index": best_idx,
            "best_mdl": float(stacked_scores[best_idx].mean().item()),
            "all_scores": [float(s.mean().item()) for s in scores],
            "candidate_telemetries": all_telem
        }
        return best_plan, best_idx, telemetry
