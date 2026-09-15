import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .halting import EntropyHaltingUnit

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

        # Scatter back to the thoughts matrix
        updated_thoughts = thoughts.clone()
        updated_thoughts[batch_idx, topk_indices] = thoughts[batch_idx, topk_indices] + attn_out
        return updated_thoughts


class RecurrentLatentController(nn.Module):
    """
    System 2: Recurrent Latent Executive Controller.
    
    Features:
    1. Query-Conditioned Thought Initialization: Hooks directly onto the query
       representation to provide an immediate semantic direction from step k=0.
    2. Weight-Tied Recurrent Transformer Layer: Preserves latent manifold geometry
       and enables residual BPTT without gradient vanishing.
    3. Top-K Capacity Gating: Deterministic compute quota for GPU SIMT alignment.
    4. On-Demand Audit Probe: Linear probe head for training semantic supervision
       and regulatory compliance audit logging.
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
        entropy_threshold: float = 0.5
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
        self.latent_mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dynamic Halting unit
        self.halting_unit = EntropyHaltingUnit(entropy_threshold=entropy_threshold)

        # Optional auxiliary probe head
        if vocab_size is not None:
            self.audit_probe = nn.Linear(d_model, vocab_size)
        else:
            self.audit_probe = None

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
        H = self.initialize_thoughts(query_rep)
        H_anchor = H.clone() # Anchor for residual stream stability

        steps = self.max_ponder_steps if k_steps is None else k_steps
        aux_logits = []
        step_entropies = []

        for step in range(steps):
            # 1. Latent Self-Attention (Reflective deliberation)
            attn_self, _ = self.latent_self_attn(H, H, H)
            H = self.norm1(H + attn_self)

            # 2. Capacity-Gated Cross-Attention ke Memory (Context grounding)
            H_cross = self.capacity_cross_attn(H, memory)
            H = self.norm2(H + H_cross + 0.1 * H_anchor)

            # 3. Latent MLP
            H = self.norm3(H + self.latent_mlp(H))

            # Audit Probe & Entropy evaluation
            if self.audit_probe is not None:
                probe_out = self.audit_probe(H[:, 0, :]) # Probe primary thought token
                if return_aux or dynamic_halting:
                    aux_logits.append(probe_out)
                    entropy = self.halting_unit.calculate_entropy(probe_out)
                    step_entropies.append(entropy)

                    if dynamic_halting and (entropy.mean().item() < self.halting_unit.entropy_threshold):
                        # Batch has achieved confident consensus; early halt
                        break

        return H, aux_logits, step_entropies
