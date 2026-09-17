import torch
import torch.nn as nn
from typing import Optional

class CognitiveWorkingMemory(nn.Module):
    """
    Cognitive Working Memory (CWM) Compressor.
    
    Compresses long context representations [B, N, D] into a compact,
    fixed-size working memory buffer [B, M, D] (where M << N, e.g. M=16..32).
    
    Hardware Rationale:
    Storing M compact slots allows recurrent latent cross-attention in the
    Outer Loop to fit completely inside GPU SRAM / L2 cache, eliminating
    the massive memory-bandwidth bottleneck of repeatedly fetching full
    KV-caches from VRAM (HBM) on each ponder step k.
    """
    def __init__(self, d_model: int, num_slots: int = 16, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.n_heads = n_heads
        
        # Learned memory query slots that summarize the context
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_mlp = nn.LayerNorm(d_model)

    def forward(self, context_emb: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            context_emb: [B, N, D] Full input prompt/context representations.
            key_padding_mask: Optional [B, N] boolean mask (True for padded positions).
        Returns:
            cwm: [B, M, D] Compressed working memory slots in SRAM-friendly size.
        """
        B = context_emb.size(0)
        q = self.slot_queries.expand(B, -1, -1) # [B, M, D]
        
        # Compress context into M memory slots
        attn_out, _ = self.cross_attn(
            query=q,
            key=context_emb,
            value=context_emb,
            key_padding_mask=key_padding_mask
        )
        slots = self.norm(q + attn_out)
        slots = self.norm_mlp(slots + self.mlp(slots))
        return slots


class EpisodicMemoryBuffer(nn.Module):
    """
    Episodic Meta-Cognitive Memory Buffer (Hippocampal Consolidation).
    
    Maintains persistent memory traces across multiple encounters/trials:
    - Stores key queries, final thought vectors, epistemic uncertainty u(x), decision margins,
      associative fast-weight matrices, and settled status.
    - Settled Anchors: Confident, validated decisions are locked into memory (`is_settled=True`).
      On subsequent encounters, they are instantly recalled and reinforced, eliminating
      second-guessing and unnecessary token/compute expenditure.
    """
    def __init__(self, d_model: int, capacity: int = 512, sim_threshold: float = 0.70):
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity
        self.sim_threshold = sim_threshold
        
        self.keys: list = []
        self.thoughts: list = []
        self.vacuities: list = []
        self.margins: list = []
        self.fast_weights: list = []
        self.is_settled: list = []
        self.confidences: list = []
        self.metadata: list = []

    def store(
        self,
        key: torch.Tensor,
        thought: torch.Tensor,
        vacuity_u: float,
        margin: float,
        m_fast: Optional[torch.Tensor] = None,
        meta: Optional[dict] = None,
        is_settled: bool = False,
        confidence: float = 0.0
    ):
        """Stores a completed deliberation episode into the episodic bank."""
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.thoughts.pop(0)
            self.vacuities.pop(0)
            self.margins.pop(0)
            self.fast_weights.pop(0)
            self.is_settled.pop(0)
            self.confidences.pop(0)
            self.metadata.pop(0)
            
        k_rep = (key.detach().squeeze(0) if key.dim() > 1 else key.detach()).clone().cpu()
        t_rep = (thought.detach().squeeze(0) if thought.dim() > 1 else thought.detach()).clone().cpu()
        m_snap = m_fast.detach().clone().cpu() if m_fast is not None else None
        
        self.keys.append(k_rep)
        self.thoughts.append(t_rep)
        self.vacuities.append(float(vacuity_u))
        self.margins.append(float(margin))
        self.fast_weights.append(m_snap)
        self.is_settled.append(bool(is_settled))
        self.confidences.append(float(confidence))
        self.metadata.append(meta or {})

    def recall(
        self,
        query: torch.Tensor,
        top_k: int = 1
    ) -> list:
        """
        Recalls the most relevant past episodic deliberation traces for the given query.
        Returns list of matched episodes with cosine similarity above threshold.
        """
        if not self.keys:
            return []
            
        q_rep = query.detach().squeeze(0) if query.dim() > 1 else query.detach()
        q_norm = torch.nn.functional.normalize(q_rep.unsqueeze(0).float().cpu(), p=2, dim=-1)
        
        k_stack = torch.stack(self.keys, dim=0).float() # [N, D]
        k_norm = torch.nn.functional.normalize(k_stack, p=2, dim=-1)
        
        sims = torch.mm(q_norm, k_norm.t()).squeeze(0) # [N]
        topk_vals, topk_inds = torch.topk(sims, k=min(top_k, len(self.keys)), dim=-1)
        
        results = []
        inds = [topk_inds.item()] if topk_inds.dim() == 0 else topk_inds.tolist()
        vals = [topk_vals.item()] if topk_vals.dim() == 0 else topk_vals.tolist()
        for val, idx in zip(vals, inds):
            if val >= self.sim_threshold:
                results.append({
                    "similarity": val,
                    "thought": self.thoughts[idx].to(device=query.device, dtype=query.dtype),
                    "vacuity_u": self.vacuities[idx],
                    "margin": self.margins[idx],
                    "m_fast": self.fast_weights[idx].to(device=query.device) if self.fast_weights[idx] is not None else None,
                    "is_settled": self.is_settled[idx],
                    "confidence": self.confidences[idx],
                    "metadata": self.metadata[idx],
                    "index": idx
                })
        return results

    def recall_settled(
        self,
        query: torch.Tensor,
        sim_threshold: float = 0.95
    ) -> Optional[dict]:
        """
        Quickly checks if this exact or near-identical problem has already been solved
        and consolidated as a Settled Anchor. If found, returns the settled episode.
        """
        recalled = self.recall(query, top_k=1)
        if recalled:
            top_match = recalled[0]
            if top_match["similarity"] >= sim_threshold and top_match["is_settled"]:
                return top_match
        return None

    def mark_settled(self, idx: int, is_settled: bool = True, confidence: Optional[float] = None):
        """Marks an existing episode as a settled anchor."""
        if 0 <= idx < len(self.is_settled):
            self.is_settled[idx] = is_settled
            if confidence is not None:
                self.confidences[idx] = float(confidence)

    def consolidate(self, decay_factor: float = 0.95):
        """Decays fast-weights of non-settled traces while preserving settled anchors."""
        for i in range(len(self.fast_weights)):
            if self.fast_weights[i] is not None:
                # Settled memories decay much slower (99% retention), exploring memories decay faster
                rate = 0.99 if self.is_settled[i] else decay_factor
                self.fast_weights[i] = self.fast_weights[i] * rate

    def clear(self):
        """Completely clears the episodic memory buffer."""
        self.keys.clear()
        self.thoughts.clear()
        self.vacuities.clear()
        self.margins.clear()
        self.fast_weights.clear()
        self.is_settled.clear()
        self.confidences.clear()
        self.metadata.clear()

    def __len__(self) -> int:
        return len(self.keys)

