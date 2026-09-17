import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, Any, Union, List

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


class UncertaintySurpriseGate(nn.Module):
    """
    Task-Aware Uncertainty & Surprise Gate based on Jensen-Shannon Divergence.
    
    Prevents overthinking on commonsense tasks (PIQA, OpenBookQA) where the base
    model's pre-trained System 1 intuition is already confident, while preserving
    deliberation on multi-hop reasoning tasks where the base model is uncertain.
    
    Mathematical Formulation:
        S = D_JS[ p_base || p_delib ] = 0.5 * D_KL[ p_base || m ] + 0.5 * D_KL[ p_delib || m ]
        where m = 0.5 * (p_base + p_delib).
        
    Gating function:
        gate = sigmoid( (S - tau_surprise) / temperature ) in [0, 1]
        
    When S < tau_surprise (deliberation didn't meaningfully change predictions or
    base model is confident): gate -> 0, suppressing delta and preserving base answers.
    When S >= tau_surprise: gate -> 1, approving full deliberation delta.
    """
    def __init__(
        self,
        d_model: int,
        tau_surprise: float = 0.01,
        temperature: float = 0.05,
        vocab_size: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_surprise = float(tau_surprise)
        self.temperature = float(temperature)
        object.__setattr__(self, "_probe", None)
        
        # Lightweight internal probe fallback when external lm_head is not bound
        probe_dim = min(vocab_size if vocab_size else 256, 256)
        self.internal_probe = nn.Linear(d_model, probe_dim)
        nn.init.normal_(self.internal_probe.weight, std=0.02)
        nn.init.zeros_(self.internal_probe.bias)

    @property
    def probe(self) -> Optional[nn.Module]:
        return getattr(self, "_probe", None)

    def set_lm_head(self, lm_head: nn.Module):
        """Bind output vocabulary projection (e.g. lm_head) with zero parameter overhead."""
        object.__setattr__(self, "_probe", lm_head)

    def compute_surprise_gate(
        self,
        h_base: torch.Tensor,
        h_delib: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates JSD surprise and returns smooth modulation gate in [0, 1].
        
        Args:
            h_base: [B, D] Initial query representation before deliberation.
            h_delib: [B, D] Deliberated representation (h_base + candidate delta).
            
        Returns:
            gate: [B, 1] Per-sample modulation factor in [0, 1].
            jsd: [B] Jensen-Shannon Divergence values.
        """
        probe = self._probe if self._probe is not None else self.internal_probe
        if probe is None:
            B = h_base.size(0)
            return torch.ones(B, 1, device=h_base.device), torch.zeros(B, device=h_base.device)

        orig_device = h_base.device
        with torch.no_grad():
            if hasattr(probe, 'weight') and probe.weight is not None:
                if h_base.dtype != probe.weight.dtype or h_base.device != probe.weight.device:
                    h_base = h_base.to(device=probe.weight.device, dtype=probe.weight.dtype)
                    h_delib = h_delib.to(device=probe.weight.device, dtype=probe.weight.dtype)

            logits_base = probe(h_base)     # [B, V]
            logits_delib = probe(h_delib)   # [B, V]

            p_base = F.softmax(logits_base, dim=-1)
            p_delib = F.softmax(logits_delib, dim=-1)

            # Jensen-Shannon Divergence (symmetric, bounded [0, ln(2)])
            m = torch.clamp(0.5 * (p_base + p_delib), min=1e-12)
            kl_bm = F.kl_div(m.log(), p_base, reduction='none').sum(-1)
            kl_dm = F.kl_div(m.log(), p_delib, reduction='none').sum(-1)
            jsd = (0.5 * (kl_bm + kl_dm)).to(device=orig_device, dtype=torch.float32) # [B]

        # Smooth gate: sigmoid((JSD - tau) / temp)
        gate = torch.sigmoid((jsd - self.tau_surprise) / self.temperature).unsqueeze(-1) # [B, 1]
        return gate, jsd


class ContrastiveEvidenceAccumulator(nn.Module):
    """
    Contrastive Evidence Accumulator across candidate options in latent space.
    
    Replaces undirected deliberation delta with targeted cross-attentive evidence
    scoring across multiple-choice candidates {c_1, ..., c_n}.
    
    Converts 'undirected overthinking' on distractor-heavy benchmarks (OpenBookQA, PIQA)
    into 'focused contrastive comparison' in latent space:
        e_i = MHA(h_thought, Enc(c_i), Enc(c_i))
        s_i = softmax( cos(e_i, h_thought) / tau_c )
        delta_contrastive = sum_i s_i * e_i
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        tau_contrast: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_contrast = float(tau_contrast)
        self.evidence_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.evidence_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.evidence_proj.weight, std=0.01)
        nn.init.zeros_(self.evidence_proj.bias)

    def forward(
        self,
        thought: torch.Tensor,
        candidate_embeds: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            thought: [B, D] or [B, 1, D] Primary deliberated thought token.
            candidate_embeds: Either:
                - List of [B, L_i, D] tensors, OR
                - Tensor of shape [B, n_cands, L_c, D] or [B, n_cands, D].
                
        Returns:
            directed_delta: [B, D] Contrastive deliberation delta.
            scores: [B, n_cands] Contrastive evidence softmax scores.
        """
        if thought.dim() == 2:
            h_prim = thought
            q = thought.unsqueeze(1) # [B, 1, D]
        else:
            h_prim = thought[:, 0, :]
            q = thought[:, :1, :]

        B, D = h_prim.shape

        # Normalize candidate_embeds to list of [B, L_i, D]
        if isinstance(candidate_embeds, torch.Tensor):
            if candidate_embeds.dim() == 3: # [B, n_cands, D]
                cands_list = [candidate_embeds[:, i:i+1, :] for i in range(candidate_embeds.size(1))]
            elif candidate_embeds.dim() == 4: # [B, n_cands, L_c, D]
                cands_list = [candidate_embeds[:, i, :, :] for i in range(candidate_embeds.size(1))]
            else:
                raise ValueError(f"Unexpected candidate_embeds tensor dimension {candidate_embeds.dim()}")
        elif isinstance(candidate_embeds, (list, tuple)):
            cands_list = list(candidate_embeds)
        else:
            raise TypeError(f"candidate_embeds must be Tensor or List of Tensors, got {type(candidate_embeds)}")

        evidence_vectors = []
        for cand_emb in cands_list:
            cand_emb = cand_emb.to(device=q.device, dtype=q.dtype)
            if cand_emb.dim() == 2:
                cand_emb = cand_emb.unsqueeze(1) # [B, 1, D]
            # Cross-attend thought query against candidate tokens
            ev, _ = self.evidence_attn(q, cand_emb, cand_emb) # [B, 1, D]
            evidence_vectors.append(ev.squeeze(1)) # [B, D]

        evidence = torch.stack(evidence_vectors, dim=1) # [B, n_cands, D]

        # Contrastive scoring via cosine similarity against thought
        thought_expanded = h_prim.unsqueeze(1).expand_as(evidence) # [B, n_cands, D]
        cos_sims = F.cosine_similarity(evidence, thought_expanded, dim=-1) # [B, n_cands]
        scores = F.softmax(cos_sims / self.tau_contrast, dim=-1) # [B, n_cands]

        # Weight evidence vectors by contrastive scores -> directed delta
        directed_delta = (scores.unsqueeze(-1) * evidence).sum(dim=1) # [B, D]
        directed_delta = self.evidence_proj(directed_delta)

        return directed_delta, scores


class DirectionalSafetyProjection(nn.Module):
    """
    Monotonic Safety Projection for deliberation deltas.
    
    Projects out the component of the deliberation delta that would flip
    the base model's top-1 prediction, guaranteeing:
        margin_new >= margin_base  (monotonic ranking preservation)
    
    Solves the BBH Logical Deduction regression where deliberation
    engages with context (positive evidence_gain) but resolves incorrectly,
    flipping confident-and-correct base answers to wrong ones.
    
    Mathematical formulation:
        d = W_lm[top1] - W_lm[top2]  (discriminant direction in vocab space)
        margin_shift = d^T @ delta    (how delta affects the margin)
        If margin_shift < 0:          (delta would flip top-1)
            delta_safe = delta + |margin_shift| / ||d||^2 * d  (project out harmful component)
        Else:
            delta_safe = delta         (delta reinforces top-1, keep it)
    
    Zero trainable parameters. ~0.05ms overhead (2 lm_head probes + 1 dot product).
    """
    def __init__(self):
        super().__init__()
        # No trainable parameters — pure geometric projection
    
    def forward(
        self,
        delta: torch.Tensor,
        h_base: torch.Tensor,
        lm_head: nn.Module,
        margin_floor: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            delta: [B, D] candidate deliberation delta.
            h_base: [B, D] original hidden state before deliberation.
            lm_head: nn.Linear or equivalent vocabulary projection.
            margin_floor: minimum base margin to protect (0 = protect all).
            
        Returns:
            delta_safe: [B, D] safety-projected delta that cannot flip top-1.
            telemetry: diagnostic dict with margin_shift and correction_magnitude.
        """
        B, D = delta.shape
        
        with torch.no_grad():
            # Compute base logits and identify the top-2 token indices
            base_cast = h_base.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
            logits_base = lm_head(base_cast)  # [B, V]
            topk = torch.topk(logits_base, k=2, dim=-1)
            top1_idx = topk.indices[:, 0]  # [B]
            top2_idx = topk.indices[:, 1]  # [B]
            base_margin = topk.values[:, 0] - topk.values[:, 1]  # [B]
            
            # Discriminant direction in hidden space: d = W[top1] - W[top2]
            W = lm_head.weight  # [V, D]
            d = (W[top1_idx] - W[top2_idx]).to(device=delta.device, dtype=delta.dtype)  # [B, D]
            d_norm_sq = (d * d).sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
        
        # How much does delta shift the margin? (positive = reinforces top-1, negative = harms)
        margin_shift = (d.detach() * delta).sum(dim=-1, keepdim=True)  # [B, 1]
        
        # Only correct when margin would decrease (margin_shift < 0)
        # AND base model has sufficient margin to protect
        needs_correction = (margin_shift < 0) & (base_margin.unsqueeze(-1).to(delta.device) > margin_floor)
        harmful_magnitude = torch.clamp(-margin_shift, min=0.0)  # [B, 1]
        
        # Project out the harmful component along discriminant direction
        correction = (harmful_magnitude / d_norm_sq.detach()) * d.detach()  # [B, D]
        correction = torch.where(needs_correction.expand_as(correction), correction, torch.zeros_like(correction))
        delta_safe = delta + correction
        
        telemetry = {
            "margin_shift": margin_shift.detach().squeeze(-1),
            "base_margin": base_margin.to(delta.device),
            "correction_magnitude": correction.norm(dim=-1).detach()
        }
        
        return delta_safe, telemetry

    @staticmethod
    def project_choice_scores(
        scores_base: np.ndarray,
        scores_delib: np.ndarray,
        base_margin: float,
        confidence_threshold: float = 0.35
    ) -> np.ndarray:
        """
        Directional safety projection on candidate choice space.
        
        Guarantees monotonic ranking safety:
        If the base model has an established confidence margin (>= confidence_threshold),
        deliberation cannot flip top-1 to runner-up due to distractor drift.
        If base model is uncertain (< confidence_threshold), deliberation is free to re-rank and rescue.
        
        Mathematically:
            Projects scores_delib onto the half-space where the confident top-1 is preserved.
        """
        if len(scores_base) <= 1:
            return scores_delib
            
        top1_idx = int(np.argmax(scores_base))
        sorted_indices = np.argsort(scores_base)[::-1]
        top2_idx = int(sorted_indices[1])
        
        delib_margin = scores_delib[top1_idx] - scores_delib[top2_idx]
        
        # If base model is confident and deliberation flips the margin:
        if base_margin >= confidence_threshold and delib_margin < 0.0:
            safe_scores = scores_delib.copy()
            # Restore top-1 above runner-up with preserved base margin fraction
            safe_scores[top1_idx] = safe_scores[top2_idx] + min(0.05, base_margin * 0.1)
            return safe_scores
            
        return scores_delib


class AdaptiveSurpriseThreshold(nn.Module):
    """
    Computes per-sample adaptive surprise gate thresholds based on
    the base model's predictive entropy.
    
    High base confidence (low entropy) → higher threshold → harder to override
    Low base confidence (high entropy) → lower threshold → easier to override
    
    Replaces the fixed global tau_surprise=0.01 with:
        tau_effective(x) = tau_base + gamma * max(0, H_ref - H(p_base(x)))
    
    where H_ref is a reference entropy level (calibrated from typical model entropy).
    
    This automatically protects:
    - BBH Logical Deduction (base=68%, low H) → restrictive gate
    - BBH Date Understanding (base=50%, medium H) → permissive gate  
    - ARC-Challenge (base=50%, high H) → very permissive gate
    
    Zero trainable parameters.
    """
    def __init__(
        self,
        tau_base: float = 0.005,
        gamma: float = 0.05,
        H_ref: float = 3.0
    ):
        super().__init__()
        self.tau_base = tau_base
        self.gamma = gamma
        self.H_ref = H_ref
    
    def compute_adaptive_tau(
        self,
        h_base: torch.Tensor,
        probe: nn.Module
    ) -> torch.Tensor:
        """
        Returns per-sample adaptive surprise threshold [B].
        
        Args:
            h_base: [B, D] base hidden states.
            probe: nn.Linear vocabulary projection (lm_head).
        """
        with torch.no_grad():
            base_cast = h_base.to(device=probe.weight.device, dtype=probe.weight.dtype)
            logits = probe(base_cast)  # [B, V]
            p = F.softmax(logits, dim=-1)
            entropy = -torch.sum(p * F.log_softmax(logits, dim=-1), dim=-1)  # [B]
            
            # Higher base confidence (lower entropy) → higher tau → more restrictive
            confidence_excess = torch.clamp(self.H_ref - entropy, min=0.0)
            tau_adaptive = self.tau_base + self.gamma * confidence_excess  # [B]
        
        return tau_adaptive.to(device=h_base.device, dtype=torch.float32)

