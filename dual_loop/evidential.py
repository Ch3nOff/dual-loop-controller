import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Any

class EvidentialEpistemicGate(nn.Module):
    """
    Evidential Epistemic Self-Recognition Gate (Subjective Logic / Dirichlet Prior).
    
    Equips the cognitive controller with intrinsic metacognitive awareness:
    distinguishing between ordinary aleatoric uncertainty (statistical spread among known options)
    and epistemic ignorance / vacuity of evidence (unprecedented, out-of-protocol, or black swan events).
    
    Mathematical Formulation:
        Given latent hidden state h in R^D or candidate logits z in R^M:
        alpha_m = softplus(W_evid h) + 1.0  (Dirichlet concentration parameter)
        S = sum_{m=1}^M alpha_m             (Dirichlet strength)
        
        b_m = (alpha_m - 1) / S             (Belief mass in known paradigm m)
        u = M / S                           (Vacuity of evidence / Epistemic uncertainty in (0, 1])
        
        sum_{m=1}^M b_m + u = 1.0           (Subjective Logic Conservation Law)
        
    Behavior:
        - When familiar/settled context: evidence is abundant -> S >> M -> u -> 0, belief b -> 1.
        - When unexpected/nir-protokol context: evidence is near zero -> S -> M -> u -> 1.0.
          Automatically detects that the situation is out-of-distribution without human supervision.
    """
    def __init__(
        self,
        d_model: int,
        num_evidence_classes: int = 16,
        tau_novelty: float = 0.40,
        tau_unseen: float = 0.65
    ):
        super().__init__()
        self.d_model = d_model
        self.num_evidence_classes = num_evidence_classes
        self.tau_novelty = float(tau_novelty)
        self.tau_unseen = float(tau_unseen)
        
        # Evidential evidence projection head: R^D -> R^M
        self.evidence_projector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, num_evidence_classes)
        )
        
        # Initialize evidence projector to produce small initial evidence
        nn.init.normal_(self.evidence_projector[-1].weight, std=0.02)
        nn.init.constant_(self.evidence_projector[-1].bias, 0.0)

    def compute_epistemic_uncertainty(
        self,
        h: torch.Tensor,
        logits_external: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes evidential decomposition into beliefs and vacuity of evidence.
        
        Args:
            h: [B, D] or [B, L, D] Hidden representation of the current context / query.
            logits_external: Optional [B, M] external logits (e.g. from candidate scoring).
            
        Returns:
            vacuity_u: [B] or [B, L] Epistemic uncertainty (0 = fully confident, 1 = completely unknown).
            beliefs: [B, M] or [B, L, M] Belief distribution over known paradigms.
            total_evidence: [B] or [B, L] Total accumulated evidence S - M.
        """
        if logits_external is not None:
            # Derive evidence directly from candidate logits
            evidence = F.softplus(logits_external)
            M = logits_external.size(-1)
        else:
            # Project through internal evidential probe
            evidence = F.softplus(self.evidence_projector(h))
            M = self.num_evidence_classes
            
        alpha = evidence + 1.0  # Dirichlet parameters alpha_m >= 1.0
        S = torch.sum(alpha, dim=-1, keepdim=False).clamp(min=1e-6)  # Dirichlet strength
        
        # Vacuity of evidence (epistemic uncertainty): u = M / S
        vacuity_u = (float(M) / S).clamp(min=0.0, max=1.0)
        
        # Belief masses: b_m = (alpha_m - 1) / S
        beliefs = evidence / S.unsqueeze(-1)
        
        # Total evidence collected
        total_evidence = torch.sum(evidence, dim=-1)
        
        return vacuity_u, beliefs, total_evidence

    def forward(
        self,
        h: torch.Tensor,
        logits_external: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Evaluates input state and returns detailed epistemic telemetry.
        """
        # If 3D [B, L, D], compute for primary token (L=0 or last token)
        if h.dim() == 3:
            h_eval = h[:, -1, :]
        else:
            h_eval = h
            
        vacuity_u, beliefs, total_evidence = self.compute_epistemic_uncertainty(
            h_eval, logits_external=logits_external
        )
        
        is_novel = vacuity_u >= self.tau_novelty
        is_unseen = vacuity_u >= self.tau_unseen
        
        return {
            "vacuity_u": vacuity_u,                    # [B] in [0, 1]
            "beliefs": beliefs,                        # [B, M]
            "total_evidence": total_evidence,          # [B]
            "is_novel": is_novel,                      # [B] bool mask (triggers plastic fast-weights)
            "is_unseen": is_unseen,                    # [B] bool mask (triggers open-concept synthesis)
            "mean_vacuity": float(vacuity_u.mean().item())
        }
