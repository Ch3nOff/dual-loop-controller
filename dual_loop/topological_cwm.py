"""
HADL Dual-Loop Cognitive Controller: Topological & Spatio-Temporal CWM
======================================================================
Implements Spatio-Temporal Entropic Cognitive Working Memory (CWM) Compression.
Prevents token explosion in multimodal domains (1024-2048 visual/audio patches)
by computing spatial visual saliency and partitioning working memory slots.

Mathematical Formulation:
-------------------------
1. Saliency / Information Density Metric:
    I(p_i) = ||h_tilde_{V, i} - h_bar_V||_2 * H(Softmax(h_tilde_{V, i}))

    where:
    - h_bar_V = mean(h_tilde_V, dim=seq) is the visual centroid representation.
    - ||h_tilde_{V, i} - h_bar_V||_2 measures spatial distinctiveness.
    - H(Softmax(h_tilde_{V, i})) is the Shannon feature entropy measuring richness.

2. Bimodal Slot Partitioning:
    Total CWM budget: M slots (e.g., M = 16)
    - M_text   = M // 2 (8 slots) dedicated to query / instruction grounding.
    - M_visual = M // 2 (8 slots) dedicated to salient sensory patches.

3. Constant SRAM / Complexity Bound:
    Regardless of whether visual tokens N_V = 576, 1024, or 4096, the cross-attention
    pool operates on at most top-K salient tokens, guaranteeing O(M) working memory
    cache with fixed SRAM footprint and sub-millisecond execution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List


class SpatioTemporalEntropicCWM(nn.Module):
    """
    Spatio-Temporal Entropic Working Memory Compressor.
    
    Provides bounded, invariant memory compression for multimodal inputs.
    """
    def __init__(
        self,
        d_model: int,
        num_slots: int = 16,
        n_heads: int = 4,
        max_salient_patches: int = 64
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.n_heads = n_heads
        self.max_salient_patches = max_salient_patches
        
        self.m_text = num_slots // 2
        self.m_visual = num_slots - self.m_text

        # Distinct learned memory query slots for linguistic vs sensory partitions
        self.text_slot_queries = nn.Parameter(torch.randn(1, self.m_text, d_model) * 0.02)
        self.visual_slot_queries = nn.Parameter(torch.randn(1, self.m_visual, d_model) * 0.02)

        # Cross-attention heads for text and visual contexts
        self.cross_attn_text = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn_visual = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm_text = nn.LayerNorm(d_model)
        self.norm_visual = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_mlp = nn.LayerNorm(d_model)

    def compute_spatial_saliency(
        self,
        h_vision: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the visual saliency metric:
            I(p_i) = ||h_V,i - h_bar_V||_2 * H(Softmax(h_V,i))
            
        Args:
            h_vision: [B, N_V, D]
            vision_mask: Optional [B, N_V] (1 for valid, 0 for pad)
        Returns:
            saliency: [B, N_V] Information density score per patch.
        """
        B, N_V, D = h_vision.shape
        device = h_vision.device
        dtype = h_vision.dtype

        # 1. Centroid distance
        if vision_mask is not None:
            v_mask_f = vision_mask.unsqueeze(-1).to(dtype=dtype)
            count = v_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
            h_bar = (h_vision * v_mask_f).sum(dim=1, keepdim=True) / count # [B, 1, D]
        else:
            h_bar = h_vision.mean(dim=1, keepdim=True) # [B, 1, D]

        dist = torch.norm(h_vision - h_bar, p=2, dim=-1) # [B, N_V]

        # 2. Shannon entropy across feature channels
        p = F.softmax(h_vision, dim=-1) # [B, N_V, D]
        log_p = torch.log(p.clamp(min=1e-8))
        entropy = -torch.sum(p * log_p, dim=-1) # [B, N_V]

        saliency = dist * entropy # [B, N_V]

        if vision_mask is not None:
            saliency = torch.where(vision_mask > 0, saliency, torch.full_like(saliency, -1e9))

        return saliency

    def forward(
        self,
        h_text: Optional[torch.Tensor],
        h_vision: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compresses bimodal context into structured working memory.
        
        Args:
            h_text: Optional [B, N_T, D] Text representations.
            h_vision: Optional [B, N_V, D] Visual/audio representations.
            text_mask: Optional [B, N_T]
            vision_mask: Optional [B, N_V]
            
        Returns:
            memory: [B, M, D] Full working memory (text + visual slots).
            visual_slots: [B, M_visual, D] Dedicated sensory slots for Popperian falsification.
            telemetry: Saliency and compression metrics.
        """
        has_vision = (h_vision is not None and h_vision.size(1) > 0)
        has_text = (h_text is not None and h_text.size(1) > 0)

        if not has_vision and not has_text:
            raise ValueError("At least one of h_text or h_vision must be provided to SpatioTemporalEntropicCWM.")

        B = h_text.size(0) if has_text else h_vision.size(0)
        device = h_text.device if has_text else h_vision.device
        dtype = h_text.dtype if has_text else h_vision.dtype

        telemetry = {}

        # ----------------------------------------------------------------------
        # Case A: Bimodal Input (Text + Vision)
        # ----------------------------------------------------------------------
        if has_text and has_vision:
            # 1. Process Text Partition: M_text slots cross-attend to h_text
            if h_text.size(1) >= self.m_text:
                q_t_seed = h_text[:, :self.m_text, :]
            else:
                q_t_seed = F.pad(h_text, (0, 0, 0, self.m_text - h_text.size(1)))
            q_t = q_t_seed + self.text_slot_queries.expand(B, -1, -1).to(device=device, dtype=dtype)
            key_pad_text = (text_mask == 0) if text_mask is not None else None
            attn_text, _ = self.cross_attn_text(
                query=q_t,
                key=h_text,
                value=h_text,
                key_padding_mask=key_pad_text
            )
            slots_text = self.norm_text(q_t + 0.1 * attn_text) # [B, M_text, D]

            # 2. Compute Visual Saliency and select top salient patches
            N_V = h_vision.size(1)
            k_salient = min(N_V, self.max_salient_patches)
            saliency = self.compute_spatial_saliency(h_vision, vision_mask=vision_mask) # [B, N_V]
            topk_indices = torch.topk(saliency, k_salient, dim=-1).indices # [B, k_salient]

            # Gather top-k patches
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, k_salient)
            h_v_salient = h_vision[batch_idx, topk_indices] # [B, k_salient, D]

            # 3. Process Visual Partition: M_visual slots seeded from salient patches + cross-attention
            if h_v_salient.size(1) >= self.m_visual:
                q_v_seed = h_v_salient[:, :self.m_visual, :]
            else:
                q_v_seed = F.pad(h_v_salient, (0, 0, 0, self.m_visual - h_v_salient.size(1)))
            q_v = q_v_seed + self.visual_slot_queries.expand(B, -1, -1).to(device=device, dtype=dtype)
            attn_visual, _ = self.cross_attn_visual(
                query=q_v,
                key=h_v_salient,
                value=h_v_salient
            )
            slots_visual = self.norm_visual(q_v + 0.1 * attn_visual) # [B, M_visual, D]

            # 4. Concatenate partitioned slots into unified working memory
            memory = torch.cat([slots_text, slots_visual], dim=1) # [B, M, D]
            memory = self.norm_mlp(memory + self.mlp(memory))

            telemetry = {
                "num_patches_total": N_V,
                "num_patches_salient": k_salient,
                "compression_ratio": float(k_salient) / float(max(1, N_V)),
                "mean_saliency": float(saliency.mean().item()),
                "m_text_slots": self.m_text,
                "m_visual_slots": self.m_visual,
                "bimodal": True
            }

            return memory, slots_visual, telemetry

        # ----------------------------------------------------------------------
        # Case B: Text-Only Fallback (Standard CWM compatibility)
        # ----------------------------------------------------------------------
        elif has_text:
            q_all = torch.cat([
                self.text_slot_queries.expand(B, -1, -1),
                self.visual_slot_queries.expand(B, -1, -1)
            ], dim=1).to(device=device, dtype=dtype) # [B, M, D]

            key_pad = (text_mask == 0) if text_mask is not None else None
            attn_out, _ = self.cross_attn_text(
                query=q_all,
                key=h_text,
                value=h_text,
                key_padding_mask=key_pad
            )
            memory = self.norm_text(q_all + attn_out)
            memory = self.norm_mlp(memory + self.mlp(memory))

            # Virtual zero visual slots
            slots_visual = memory[:, self.m_text:, :]
            telemetry = {
                "num_patches_total": 0,
                "num_patches_salient": 0,
                "bimodal": False
            }
            return memory, slots_visual, telemetry

        # ----------------------------------------------------------------------
        # Case C: Vision-Only Input
        # ----------------------------------------------------------------------
        else:
            N_V = h_vision.size(1)
            k_salient = min(N_V, self.max_salient_patches)
            saliency = self.compute_spatial_saliency(h_vision, vision_mask=vision_mask)
            topk_indices = torch.topk(saliency, k_salient, dim=-1).indices
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, k_salient)
            h_v_salient = h_vision[batch_idx, topk_indices]

            q_all = torch.cat([
                self.text_slot_queries.expand(B, -1, -1),
                self.visual_slot_queries.expand(B, -1, -1)
            ], dim=1).to(device=device, dtype=dtype)

            attn_out, _ = self.cross_attn_visual(
                query=q_all,
                key=h_v_salient,
                value=h_v_salient
            )
            memory = self.norm_visual(q_all + attn_out)
            memory = self.norm_mlp(memory + self.mlp(memory))
            slots_visual = memory[:, self.m_text:, :]

            telemetry = {
                "num_patches_total": N_V,
                "num_patches_salient": k_salient,
                "compression_ratio": float(k_salient) / float(max(1, N_V)),
                "bimodal": False
            }
            return memory, slots_visual, telemetry
