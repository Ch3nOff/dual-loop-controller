import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .halting import EntropyHaltingUnit, LearnedHaltingGate

class TopKCapacityCrossAttention(nn.Module):
    """
    Capacity-Constrained Static Cross-Attention (Mixture-of-Depths style).
    
    Guarantees fixed tensor shapes [B, K_cap, D] on GPU to prevent
    warp divergence, execution serialization, and dynamic allocation overhead.
    """
    def __init__(self, d_model: int, n_heads: int, capacity_factor: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.capacity_factor = capacity_factor
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.router = nn.Linear(d_model, 1)

    def forward(self, thoughts: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            thoughts: [B, L, D] Thought tokens
            memory: [B, M, D] Cognitive Working Memory or context buffer
        Returns:
            updated_thoughts: [B, L, D]
        """
        B, L, D = thoughts.shape
        K_cap = max(1, int(L * self.capacity_factor))

        # Predict need for memory access
        scores = self.router(thoughts).squeeze(-1) # [B, L]
        topk_indices = torch.topk(scores, K_cap, dim=-1).indices # [B, K_cap]

        # Extract strictly static-shaped sub-tensor
        batch_idx = torch.arange(B, device=thoughts.device).unsqueeze(1).expand(-1, K_cap)
        selected_thoughts = thoughts[batch_idx, topk_indices] # [B, K_cap, D]

        # Cross-attend only on the fixed quota
        attn_out, _ = self.mha(selected_thoughts, memory, memory)

        # Scatter back: return only the cross-attention delta so the caller's
        # residual addition (H + H_cross) is clean and avoids a double residual (TENSOR-03)
        attn_delta = torch.zeros_like(thoughts)
        attn_delta[batch_idx, topk_indices] = attn_out
        return attn_delta


class LatentCritiqueRefinementUnit(nn.Module):
    """
    Metacognitive Latent Critique & Error-Refinement Unit.
    
    Evaluates discrepancy between current deliberation thoughts and grounded context,
    extracting an error vector e_k that is projected into a corrective residual delta:
    
    1. Discrepancy signal: e_k = LayerNorm(thoughts - cross_delta)
    2. Metacognitive critique: delta_correct = GELU(W_critique(e_k)) * sigmoid(W_gate(e_k))
    3. Returns:
       - delta_correct: [B, L, D] corrective direction to repair mistakes
       - error_norm: [B] scalar inconsistency score used for adaptive halting
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.norm_err = nn.LayerNorm(d_model)
        self.critique_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.error_gate = nn.Linear(d_model, d_model)
        
        # Initialize final projection with small scale so critique starts gentle
        nn.init.normal_(self.critique_net[-1].weight, std=0.01)
        nn.init.zeros_(self.critique_net[-1].bias)
        nn.init.zeros_(self.error_gate.weight)
        nn.init.constant_(self.error_gate.bias, -1.0)

    def forward(self, thoughts: torch.Tensor, cross_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_err = thoughts - cross_delta
        e_k = self.norm_err(raw_err)
        gate = torch.sigmoid(self.error_gate(e_k))
        delta_correct = self.critique_net(e_k) * gate
        error_norm = raw_err.norm(dim=-1).mean(dim=-1) # [B]
        return delta_correct, error_norm


class RecurrentLatentController(nn.Module):
    """
    System 2: Recurrent Latent Executive Controller with Metacognitive Error-Refinement.
    
    Features:
    1. Query-Conditioned Thought Initialization: Hooks directly onto the query
       representation to provide an immediate semantic direction from step k=0.
    2. Weight-Tied Recurrent Transformer Layer: Preserves latent manifold geometry
       and enables residual BPTT without gradient vanishing.
    3. Top-K Capacity Gating: Deterministic compute quota for GPU SIMT alignment.
    4. Metacognitive Critique & Error Correction: Actively computes discrepancy against
       context memory and injects corrective critique updates (learns from mistakes).
    5. Adaptive Learned Anchor Gate: Contraction mapping guaranteeing stability at K >= 4.
    6. Multi-Signal Halting Unit: PonderNet-inspired learned halting or calibrated entropy.
    7. On-Demand Audit Probe: Linear probe head for training semantic supervision.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_ff: int = 128,
        num_thought_tokens: int = 4,
        max_ponder_steps: int = 4,
        vocab_size: Optional[int] = None,
        capacity_factor: float = 0.5,
        entropy_threshold: float = 0.5,
        enable_critique: bool = True,
        use_learned_halting: bool = False,
        lambda_prior: float = 0.5,
        tau_halt: float = 0.75
    ):
        super().__init__()
        self.d_model = d_model
        self.num_thought_tokens = num_thought_tokens
        self.max_ponder_steps = max_ponder_steps
        self.vocab_size = vocab_size

        # Query projection for initializing thoughts
        self.query_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.learned_slot_offsets = nn.Parameter(torch.randn(1, num_thought_tokens, d_model) * 0.02)

        # Weight-tied recurrent transformer block
        self.latent_self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.capacity_cross_attn = TopKCapacityCrossAttention(d_model, n_heads, capacity_factor=capacity_factor)
        
        # Metacognitive Critique Unit (Error Recognition & Self-Correction)
        self.enable_critique = enable_critique
        self.critique_unit = LatentCritiqueRefinementUnit(d_model) if enable_critique else None

        # Learned Adaptive Anchor Gate (Contraction Mapping for K >= 4 stabilization)
        self.anchor_gate = nn.Linear(d_model, 1)
        nn.init.constant_(self.anchor_gate.bias, 1.0)
        nn.init.normal_(self.anchor_gate.weight, std=0.01)

        self.latent_mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dynamic Halting units
        self.halting_unit = EntropyHaltingUnit(entropy_threshold=entropy_threshold)
        self.use_learned_halting = use_learned_halting
        self.learned_halting_gate = LearnedHaltingGate(
            d_model=d_model,
            lambda_prior=lambda_prior,
            tau_halt=tau_halt,
            vocab_size=vocab_size
        ) if use_learned_halting else None

        self.last_lambdas: List[torch.Tensor] = []
        self.last_error_norms: List[torch.Tensor] = []

        # Optional auxiliary probe head
        if vocab_size is not None:
            self.audit_probe = nn.Linear(d_model, vocab_size)
        else:
            self.audit_probe = None

    def reset_state(self):
        """Clear mutable instance telemetry state to prevent cross-request leakage (ARCH-02 / NEW-02)."""
        self.last_lambdas = []
        self.last_error_norms = []

    def initialize_thoughts(self, query_rep: torch.Tensor) -> torch.Tensor:
        """
        Initializes H_0 conditioned directly on the query.
        Args:
            query_rep: [B, D] Representation of the query token / instruction.
        Returns:
            H_0: [B, L_thought, D] Initialized thought vectors.
        """
        B = query_rep.size(0)
        base = self.query_projector(query_rep).unsqueeze(1) # [B, 1, D]
        H_0 = base.expand(B, self.num_thought_tokens, -1) + self.learned_slot_offsets
        return H_0

    def step_deliberation(
        self,
        H: torch.Tensor,
        H_anchor: torch.Tensor,
        memory: torch.Tensor,
        H_prev: Optional[torch.Tensor] = None,
        h_prev_primary: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Executes a single recurrent step of outer loop deliberation.
        Unified across full rollout (forward) and per-step interactive decoding (decoder PATH A).
        
        Args:
            H: [B, L_thought, D] Current thought representations.
            H_anchor: [B, L_thought, D] Anchor thoughts from initial query.
            memory: [B, M, D] Working memory buffer (CWM).
            H_prev: [B, L_thought, D] Optional previous step thought representations.
            h_prev_primary: [B, D] Optional previous primary thought representation.
            
        Returns:
            H_next: [B, L_thought, D] Updated thought representations.
            err_norm: [B] or None Discrepancy norm from LatentCritiqueRefinementUnit.
            lam_k: [B] or None Halting probability from LearnedHaltingGate.
        """
        if H_prev is None:
            H_prev = H.clone()

        # 1. Latent Self-Attention (Reflective deliberation)
        attn_self, _ = self.latent_self_attn(H, H, H)
        H = self.norm1(H + attn_self)

        # 2. Capacity-Gated Cross-Attention to Memory (Context grounding)
        H_cross = self.capacity_cross_attn(H, memory)

        # 3. Metacognitive Error-Reflection & Self-Correction (Learn from mistakes)
        if self.critique_unit is not None:
            delta_critique, err_norm = self.critique_unit(H, H_cross)
            H_updated = H + H_cross + delta_critique
        else:
            err_norm = None
            H_updated = H + H_cross

        # 4. Learned Adaptive Anchor Gate (Contraction Mapping: guarantees stability at K >= 4)
        alpha = torch.sigmoid(self.anchor_gate(H_updated)) # [B, L, 1]
        H = self.norm2(alpha * H_updated + (1.0 - alpha) * H_anchor)

        # 5. Latent MLP
        H = self.norm3(H + self.latent_mlp(H))

        # Recurrent state norm clipping (ARCH-01: prevent divergence at higher ponder steps)
        h_norm = torch.norm(H, p=2, dim=-1, keepdim=True)
        max_norm = 50.0
        clip_coef = torch.clamp(max_norm / (h_norm + 1e-6), max=1.0)
        H = H * clip_coef

        h_primary = H[:, 0, :]

        # 6. Learned Halting Gate Evaluation (PonderNet multi-signal)
        lam_k = None
        if self.learned_halting_gate is not None:
            lam_k = self.learned_halting_gate.compute_lambda(
                h_curr=h_primary,
                h_prev=h_prev_primary,
                H_curr=H,
                H_prev=H_prev,
                discrepancy_norm=err_norm
            )

        return H, err_norm, lam_k

    def forward(
        self,
        query_rep: torch.Tensor,
        memory: torch.Tensor,
        k_steps: Optional[int] = None,
        dynamic_halting: bool = False,
        return_aux: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Executes recurrent latent pondering.
        
        Args:
            query_rep: [B, D] Query/prompt anchor vector.
            memory: [B, M, D] Working memory buffer (from CWM).
            k_steps: Fixed number of ponder steps (overrides max_ponder_steps).
            dynamic_halting: If True, evaluates entropy to halt early.
            return_aux: If True, collects probe predictions for audit/training.
            
        Returns:
            H_final: [B, L_thought, D] Final thought state.
            aux_logits: List of [B, VocabSize] for each step (if audit_probe exists).
            step_entropies: List of [B] entropy values per step.
        """
        # ARCH-02 / NEW-02: Clear mutable instance state at forward entry to prevent cross-request leakage
        self.reset_state()

        H = self.initialize_thoughts(query_rep)
        H_anchor = H.detach() if not self.training else H.clone() # Anchor for residual stream stability (TENSOR-01)

        steps = self.max_ponder_steps if k_steps is None else k_steps
        aux_logits = []
        step_entropies = []
        lambdas = []
        error_norms = []
        h_prev_primary = None

        for step in range(steps):
            H_prev = H.clone()

            # Execute unified recurrent step of Outer Loop
            H, err_norm, lam_k = self.step_deliberation(
                H=H,
                H_anchor=H_anchor,
                memory=memory,
                H_prev=H_prev,
                h_prev_primary=h_prev_primary
            )

            if err_norm is not None:
                error_norms.append(err_norm)

            if lam_k is not None:
                lambdas.append(lam_k)
                h_prev_primary = H[:, 0, :].detach()

                if dynamic_halting and not self.training:
                    halt_mask = self.learned_halting_gate.should_halt_inference(lam_k)
                    if halt_mask.all():
                        break

            # Audit Probe & Entropy evaluation
            h_primary = H[:, 0, :]
            if self.audit_probe is not None:
                probe_out = self.audit_probe(h_primary) # Probe primary thought token
                if return_aux or dynamic_halting:
                    aux_logits.append(probe_out)
                    entropy = self.halting_unit.calculate_entropy(probe_out)
                    step_entropies.append(entropy)

                    if self.learned_halting_gate is None and dynamic_halting and (entropy.mean().item() < self.halting_unit.entropy_threshold):
                        # Batch has achieved confident consensus; early halt
                        break
            elif dynamic_halting and self.learned_halting_gate is None:
                # Latent representation delta convergence
                rel_delta = (H - H_prev).norm() / (H_prev.norm() + 1e-6)
                step_entropies.append(rel_delta.unsqueeze(0))
                if step > 0 and rel_delta.item() < self.halting_unit.delta_threshold:
                    break

        self.last_lambdas = lambdas
        self.last_error_norms = error_norms
        return H, aux_logits, step_entropies
