import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class HypothesisVerificationGate(nn.Module):
    """
    Counterfactual Hypothesis-Testing & Conservative Verification Gate.
    
    Treats recurrent latent deliberation as a *controlled mental trial* (hypothesis testing)
    rather than open-loop overthinking. Evaluates whether candidate deliberated thoughts
    exhibit statistically and semantically meaningful evidence gain over the initial
    System 1 query prior before approving residual injection into the LLM backbone.
    
    Mathematical Formulation:
    1. Epistemic Prior Certainty:
       c_prior = sigmoid(w_prior^T q + b_prior) in [0, 1]
       Measures whether the query is already an obvious, settled intuition.
       
    2. Evidence Gain:
       sim_trial = max_m cos(h_primary, memory_m)
       sim_prior = max_m cos(q, memory_m)
       delta_evidence = sim_trial - sim_prior
       
    3. Trust Region Drift:
       drift = 1 - cos(h_primary, q)
       
    4. Acceptance Probability beta in [0, 1]:
       features = [delta_evidence, drift, c_prior]
       beta = sigmoid(W_verify @ features + b_verify)
       
    5. Conservative Verification:
       If deliberation degrades evidence alignment on a settled intuition,
       beta is suppressed, falling back to the intuitive prior and preventing overthinking.
    """
    def __init__(
        self,
        d_model: int,
        tau_fallback: float = 0.25,
        evidence_margin: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_fallback = tau_fallback
        self.evidence_margin = evidence_margin
        
        # Evaluates initial query certainty (is System 1 already confident?)
        self.prior_evaluator = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 4)),
            nn.GELU(),
            nn.Linear(max(16, d_model // 4), 1)
        )
        
        # Features: [evidence_gain, drift, prior_certainty] -> logit
        self.verification_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        # Safe initialization: starts with positive acceptance bias so normal deliberation
        # is preserved, but trains to reject overthinking when evidence_gain is negative.
        nn.init.constant_(self.verification_net[-1].bias, 1.5)
        nn.init.normal_(self.verification_net[-1].weight, std=0.02)
        nn.init.constant_(self.prior_evaluator[-1].bias, 0.0)

    def forward(
        self,
        thoughts: torch.Tensor,
        query_rep: torch.Tensor,
        memory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            thoughts: [B, L, D] Candidate thoughts after K deliberation steps.
            query_rep: [B, D] Intuitive System 1 query anchor representation.
            memory: [B, M, D] Working memory context buffer.
            
        Returns:
            verified_thoughts: [B, L, D] Gated thoughts.
            acceptance_beta: [B, 1, 1] Scaling factor in [0, 1].
            telemetry: Dict containing diagnostic verification signals.
        """
        B, L, D = thoughts.shape
        device = thoughts.device
        
        h_primary = thoughts[:, 0, :]  # Primary thought token [B, D]
        
        # 1. Epistemic Prior Certainty c_prior: [B]
        c_prior = torch.sigmoid(self.prior_evaluator(query_rep)).squeeze(-1)
        
        # 2. Evidence Gain: does deliberated thought ground more tightly with context than raw query?
        h_norm = F.normalize(h_primary, p=2, dim=-1)  # [B, D]
        q_norm = F.normalize(query_rep, p=2, dim=-1)  # [B, D]
        m_norm = F.normalize(memory, p=2, dim=-1)     # [B, M, D]
        
        # Best alignment with any memory slot: [B, 1, D] @ [B, D, M] -> [B, 1, M]
        sim_trial = torch.bmm(h_norm.unsqueeze(1), m_norm.transpose(1, 2)).squeeze(1).max(dim=-1).values
        sim_prior = torch.bmm(q_norm.unsqueeze(1), m_norm.transpose(1, 2)).squeeze(1).max(dim=-1).values
        evidence_gain = sim_trial - sim_prior  # [B]
        
        # 3. Trust Region Drift: how far has deliberation pulled thought from query anchor?
        cos_anchor = F.cosine_similarity(h_primary, query_rep, dim=-1)  # [B]
        drift = torch.clamp(1.0 - cos_anchor, min=0.0, max=2.0)
        
        # 4. Multi-signal verification features: [B, 3]
        features = torch.stack([evidence_gain, drift, c_prior], dim=-1)
        learned_logit = self.verification_net(features).squeeze(-1)  # [B]
        
        # Analytic Evidence Factor (Bayesian Likelihood Ratio proxy):
        # If deliberation finds strong evidence in context: evidence_gate -> 1.0
        # If deliberation drifts with negative evidence: evidence_gate -> 0.0 (suppresses overthinking)
        evidence_gate = torch.sigmoid(evidence_gain / 0.05)  # [B]
        
        # Combined acceptance probability: learned neural modulation * analytic evidence factor
        beta_eff = torch.sigmoid(learned_logit) * evidence_gate  # [B] in (0, 1)
        
        # Conservative threshold: if beta is below tau_fallback, smoothly damp to zero
        beta_eff = torch.where(beta_eff < self.tau_fallback, beta_eff * (beta_eff / self.tau_fallback), beta_eff)
        
        acceptance_beta = beta_eff.view(B, 1, 1)  # [B, 1, 1]
        
        # Verified thoughts: convex blend between deliberated thought and initial query
        q_expanded = query_rep.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        verified_thoughts = acceptance_beta * thoughts + (1.0 - acceptance_beta) * q_expanded
        
        telemetry = {
            "evidence_gain": evidence_gain.detach(),
            "drift": drift.detach(),
            "c_prior": c_prior.detach(),
            "acceptance_beta": beta_eff.detach()
        }
        
        return verified_thoughts, acceptance_beta, telemetry
