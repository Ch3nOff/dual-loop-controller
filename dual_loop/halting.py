import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

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


class LearnedHaltingGate(nn.Module):
    """
    PonderNet-inspired learned halting gate with geometric prior regularization.
    
    Replaces heuristic entropy thresholding with a differentiable, task-adaptive
    stopping criterion that consumes multi-signal convergence features:
    1. Cosine similarity drift of primary thought token (directional convergence)
    2. Metacognitive discrepancy / error magnitude (inconsistency between thoughts and memory)
    3. Relative L2 state delta across thought tokens
    4. Optional entropy drop (if audit probe exists)
    
    Training: Computes geometric halting distribution p(N=k) and KL loss against Geometric(lambda_prior).
    Inference: Deterministic early-exit when lambda_k > tau_halt.
    """
    def __init__(
        self,
        d_model: int,
        lambda_prior: float = 0.5,
        beta_kl: float = 0.01,
        tau_halt: float = 0.75,
        vocab_size: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.lambda_prior = lambda_prior
        self.beta_kl = beta_kl
        self.tau_halt = tau_halt
        self.vocab_size = vocab_size

        # Multi-signal feature projection:
        # Features: [cos_drift, discrepancy_norm, rel_l2_delta, entropy_delta] -> scalar logit
        self.halt_proj = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        # Direct projection fallback from primary thought representation
        self.thought_proj = nn.Linear(d_model, 1)

        # Optional probe for predictive entropy
        self.probe = nn.Linear(d_model, vocab_size) if vocab_size is not None else None

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy in nats from unnormalized logits."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -torch.sum(probs * log_probs, dim=-1)

    def compute_lambda(
        self,
        h_curr: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        H_curr: Optional[torch.Tensor] = None,
        H_prev: Optional[torch.Tensor] = None,
        discrepancy_norm: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        logits_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes per-sample halting probability lambda_k in (0, 1).
        
        Args:
            h_curr: [B, D] primary thought token at step k.
            h_prev: [B, D] primary thought token at step k-1 (None for step 0).
            H_curr: [B, L, D] full thought matrix at step k.
            H_prev: [B, L, D] full thought matrix at step k-1.
            discrepancy_norm: [B] scalar metacognitive error norm.
            logits: Optional [B, VocabSize] current audit/model logits.
            logits_prev: Optional [B, VocabSize] previous audit/model logits.
        """
        B = h_curr.size(0)
        device = h_curr.device

        if h_prev is not None:
            # Signal 1: Cosine drift (0 = identical/converged, 2 = maximally opposite)
            cos_sim = F.cosine_similarity(h_curr, h_prev, dim=-1)  # [B]
            cos_drift = torch.clamp(1.0 - cos_sim, min=0.0, max=2.0)

            # Signal 2: Discrepancy norm (error signal from critique unit)
            if discrepancy_norm is None:
                disc_signal = torch.zeros(B, device=device)
            else:
                disc_signal = discrepancy_norm.clamp(0.0, 10.0)

            # Signal 3: Relative Frobenius delta of thought matrix
            if H_curr is not None and H_prev is not None:
                diff_norm = (H_curr - H_prev).norm(dim=[-2, -1])
                base_norm = H_prev.norm(dim=[-2, -1]) + 1e-6
                rel_delta = (diff_norm / base_norm).clamp(0.0, 5.0)
            else:
                rel_delta = ((h_curr - h_prev).norm(dim=-1) / (h_prev.norm(dim=-1) + 1e-6)).clamp(0.0, 5.0)

            # Signal 4: Entropy drop (positive = model became more certain)
            if logits is not None and logits_prev is not None:
                ent_curr = self._entropy(logits)
                ent_prev = self._entropy(logits_prev)
                ent_drop = (ent_prev - ent_curr).clamp(-5.0, 5.0)
            elif self.probe is not None:
                ent_curr = self._entropy(self.probe(h_curr))
                ent_prev = self._entropy(self.probe(h_prev))
                ent_drop = (ent_prev - ent_curr).clamp(-5.0, 5.0)
            else:
                ent_drop = torch.zeros(B, device=device)

            features = torch.stack([cos_drift, disc_signal, rel_delta, ent_drop], dim=-1)  # [B, 4]
            logit = self.halt_proj(features).squeeze(-1)  # [B]
        else:
            # Step 0: project directly from initial thought vector
            logit = self.thought_proj(h_curr).squeeze(-1)  # [B]

        return torch.sigmoid(logit)  # [B] in (0, 1)

    def geometric_kl_loss(self, lambdas: List[torch.Tensor]) -> torch.Tensor:
        """
        KL divergence between learned halting distribution and geometric prior:
        D_KL[ p(N) || p_G(N; lambda_prior) ]
        """
        if not lambdas:
            return torch.tensor(0.0)

        K = len(lambdas)
        device = lambdas[0].device
        B = lambdas[0].size(0)

        # Compute p(N=k) = lambda_k * prod_{j<k} (1 - lambda_j)
        p_halt = []
        log_survival = torch.zeros(B, device=device)
        for k in range(K):
            lam_k = lambdas[k].clamp(1e-6, 1.0 - 1e-6)
            log_p_k = torch.log(lam_k) + log_survival
            p_halt.append(log_p_k.exp())
            log_survival = log_survival + torch.log(1.0 - lam_k)

        # Remaining mass assigned to last step
        p_halt[-1] = p_halt[-1] + log_survival.exp()

        p_model = torch.stack(p_halt, dim=-1)  # [B, K]
        p_model = p_model / (p_model.sum(dim=-1, keepdim=True) + 1e-8)

        # Geometric prior: p_G(N=k) = lambda_G * (1 - lambda_G)^(k-1)
        p_prior = []
        for k in range(K):
            if k < K - 1:
                p_prior.append(self.lambda_prior * ((1.0 - self.lambda_prior) ** k))
            else:
                p_prior.append((1.0 - self.lambda_prior) ** k)
        p_prior = torch.tensor(p_prior, device=device, dtype=torch.float32)

        kl = F.kl_div(
            (p_model + 1e-8).log(),
            p_prior.unsqueeze(0).expand_as(p_model),
            reduction='batchmean',
            log_target=False
        )
        return self.beta_kl * kl

    def compute_halting_distribution(self, lambdas: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes p(N=k) for k in 0..K-1.
        Returns:
            p_model: [B, K] normalized halting probability distribution.
        """
        if not lambdas:
            return torch.empty(0)
        K = len(lambdas)
        device = lambdas[0].device
        B = lambdas[0].size(0)

        p_halt = []
        log_survival = torch.zeros(B, device=device)
        for k in range(K):
            lam_k = lambdas[k].clamp(1e-6, 1.0 - 1e-6)
            log_p_k = torch.log(lam_k) + log_survival
            p_halt.append(log_p_k.exp())
            log_survival = log_survival + torch.log(1.0 - lam_k)

        p_halt[-1] = p_halt[-1] + log_survival.exp()
        p_model = torch.stack(p_halt, dim=-1)
        return p_model / (p_model.sum(dim=-1, keepdim=True) + 1e-8)

    def compute_pondernet_loss(
        self,
        lambdas: List[torch.Tensor],
        step_logits: List[torch.Tensor],
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes total PonderNet loss:
        L_total = E_{k ~ p(N)} [ L_task(step_logits[k], target) ] + beta_kl * D_KL[ p(N) || p_G ]
        
        Returns:
            total_loss, p_model, kl_loss, weighted_rec_loss
        """
        p_model = self.compute_halting_distribution(lambdas)  # [B, K]
        kl_loss = self.geometric_kl_loss(lambdas)

        K = len(step_logits)
        rec_losses = []
        for k in range(K):
            loss_k = F.cross_entropy(step_logits[k], target, reduction='none')  # [B]
            rec_losses.append(loss_k)
        rec_matrix = torch.stack(rec_losses, dim=-1)  # [B, K]
        weighted_rec = (p_model * rec_matrix).sum(dim=-1).mean()
        total_loss = weighted_rec + kl_loss
        return total_loss, p_model, kl_loss, weighted_rec

    @torch.no_grad()
    def should_halt_inference(self, lambda_k: torch.Tensor) -> torch.Tensor:
        """Deterministic halting mask for inference (True for samples that halt)."""
        return lambda_k > self.tau_halt


class DriftDiffusionHalting(nn.Module):
    """
    Drift-Diffusion Model (DDM) inspired halting criterion.
    
    Grounded in the Sequential Probability Ratio Test (Wald, 1945) and cognitive
    neuroscience's evidence accumulation models (Ratcliff & McKoon, 2008).
    
    Maps the log-likelihood ratio (evidence margin) between top-1 and top-2 predictions:
        x_k = logit_{top1}^{(k)} - logit_{top2}^{(k)}
    to an evidence accumulation process with collapsing decision boundaries:
        theta_k = theta_0 * (1 - k / K_max)^gamma
        
    For commonsense tasks (PIQA, OpenBookQA):
        Evidence accumulates rapidly (high drift) -> |x_1| > theta_1 -> early halt at k=1.
        Protects pre-trained intuitive representations from epistemic drift / overthinking.
        
    For complex reasoning tasks (AA-LCR, ARC-Challenge):
        Evidence accumulates slowly (low drift) -> |x_k| < theta_k -> deliberates full steps.
    """
    def __init__(
        self,
        theta_0: float = 3.0,
        gamma: float = 0.5,
        k_max: int = 4,
        min_theta: float = 0.5
    ):
        super().__init__()
        self.theta_0 = nn.Parameter(torch.tensor(float(theta_0), dtype=torch.float32))
        self.gamma = float(gamma)
        self.k_max = int(k_max)
        self.min_theta = float(min_theta)

    def decision_boundary(self, k: int) -> torch.Tensor:
        """
        Collapsing decision boundary: starts strict, relaxes over time
        as cognitive urgency increases.
        """
        progress = min(float(k), float(self.k_max)) / float(max(1, self.k_max))
        urgency_factor = max(0.0, 1.0 - progress) ** self.gamma
        boundary = torch.clamp(self.theta_0 * urgency_factor, min=self.min_theta)
        return boundary

    def compute_evidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Log-likelihood ratio (margin) between top-1 and top-2 predictions.
        
        Args:
            logits: [B, V] Prediction logits.
        Returns:
            evidence: [B] Evidence margin (positive: decisive, near-zero: uncertain).
        """
        if logits.size(-1) < 2:
            return torch.zeros(logits.size(0), device=logits.device)
        topk = torch.topk(logits, k=2, dim=-1)
        evidence = topk.values[:, 0] - topk.values[:, 1]
        return evidence

    def should_halt(self, logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates whether evidence has crossed the decision boundary at step k.
        
        Args:
            logits: [B, V]
            k: Current deliberation step index (1-indexed)
            
        Returns:
            halt_mask: [B] Boolean mask (True for decisive samples)
            evidence: [B] Evidence margin
            boundary: scalar boundary tensor
        """
        evidence = self.compute_evidence(logits)
        boundary = self.decision_boundary(k)
        halt_mask = evidence.abs() > boundary
        return halt_mask, evidence, boundary
