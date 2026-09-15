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
