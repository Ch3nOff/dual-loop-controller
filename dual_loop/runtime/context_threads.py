"""
Latent Context Threads & Dual-Store Memory Engine (HADL v4.5+)
==============================================================
Implements:
1. OC-Dim Chunked Context Thread Engine:
   Expands intermediate tokens to an orthogonal subspace (D -> 2D) and condenses
   them into self-contained "Latent Macro-Thread Vectors", preventing attention dispersion.
2. Dual-Store Memory Architecture:
   Separates Fast Short-Term Working Buffer from Long-Term Plastic Difference Matrix
   with non-saturating adaptive decay lambda(t).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any


class LatentContextThreadEngine(nn.Module):
    """
    OC-Dim Latent Context Thread Engine.
    
    Transforms sequential token representations into compressed, self-organizing
    macro-thread vectors via orthogonal dimension expansion (D -> 2D) and bounded pooling.
    This eliminates attention dispersion over long-horizon sequences.
    """
    def __init__(self, d_model: int = 2048, chunk_size: int = 16, expansion_factor: int = 2):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.d_expanded = d_model * expansion_factor
        
        # 1. OC-Dim Expansion: D -> 2D
        self.w_expand = nn.Linear(d_model, self.d_expanded, bias=False)
        self.norm_expanded = nn.LayerNorm(self.d_expanded)
        
        # 2. Self-Supervised Latent Thread Aggregator
        self.thread_gate = nn.Linear(self.d_expanded, 1, bias=True)
        self.thread_compress = nn.Linear(self.d_expanded, d_model, bias=False)
        self.norm_thread = nn.LayerNorm(d_model)
        
        # Initialize projections near identity / small variance
        nn.init.orthogonal_(self.thread_compress.weight)
        
    def chunk_and_compress(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compresses historical sequence (B, T, D) into latent thread vectors (B, N_chunks, D).
        Returns:
            macro_threads: (B, N_chunks, D) compressed context threads
            residual_tokens: (B, T_recent, D) unchunked recent tokens
        """
        B, T, D = hidden_states.shape
        if T < self.chunk_size:
            # Not enough tokens to form a full thread
            return torch.empty(B, 0, D, device=hidden_states.device, dtype=hidden_states.dtype), hidden_states
            
        num_full_chunks = T // self.chunk_size
        chunked_len = num_full_chunks * self.chunk_size
        
        # Separate full chunks from trailing residual tokens
        to_chunk = hidden_states[:, :chunked_len, :]  # (B, chunked_len, D)
        residual = hidden_states[:, chunked_len:, :]   # (B, T - chunked_len, D)
        
        # Reshape to (B * num_chunks, chunk_size, D)
        chunks = to_chunk.reshape(B * num_full_chunks, self.chunk_size, D)
        
        # OC-Dim Expansion: (B * N, chunk_size, 2D)
        h_exp = F.gelu(self.norm_expanded(self.w_expand(chunks)))
        
        # Softmax Attention Pooling within chunk
        scores = F.softmax(self.thread_gate(h_exp), dim=1)  # (B * N, chunk_size, 1)
        pooled_exp = torch.sum(scores * h_exp, dim=1)        # (B * N, 2D)
        
        # Compress back to canonical model dimension
        threads = self.norm_thread(self.thread_compress(pooled_exp))  # (B * N, D)
        macro_threads = threads.reshape(B, num_full_chunks, D)        # (B, num_chunks, D)
        
        return macro_threads, residual


class DualStoreMemory(nn.Module):
    """
    Dual-Store Cognitive Memory Architecture.
    
    1. Short-Term Store: Fast sliding window buffer of recent token representations.
    2. Long-Term Store: Plastic Context Difference Matrix M_long in R^{D x D}
       that tracks deviation vectors Delta v_t = v_t - mu_context with adaptive
       decay lambda(t) to prevent eigenvalue saturation on long horizons.
    """
    def __init__(
        self,
        d_model: int = 2048,
        short_term_capacity: int = 256,
        rank: int = 32,
        base_lr: float = 0.10,
        base_decay: float = 0.02
    ):
        super().__init__()
        self.d_model = d_model
        self.capacity = short_term_capacity
        self.rank = rank
        self.base_lr = base_lr
        self.base_decay = base_decay
        
        # Low-rank associative projections: R^{D} -> R^{rank}
        self.proj_key = nn.Linear(d_model, rank, bias=False)
        self.proj_val = nn.Linear(d_model, rank, bias=False)
        self.proj_out = nn.Linear(rank, d_model, bias=False)
        
        # Running centroid buffer for difference vector calculation
        self.register_buffer("running_centroid", torch.zeros(d_model))
        self.register_buffer("centroid_count", torch.zeros(1))
        
        # State: Long-term memory trace in low-rank space (B, rank, rank)
        self.m_long: Optional[torch.Tensor] = None
        self.step_counter: int = 0
        
    def reset_state(self):
        """Clears memory states between distinct inference sessions."""
        self.m_long = None
        self.step_counter = 0
        self.running_centroid.zero_()
        self.centroid_count.zero_()
        
    def forward(
        self,
        query: torch.Tensor,
        current_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Current hidden representation (B, L, D)
            current_state: Current hidden states (B, L, D) to assimilate into memory
        Returns:
            recalled_delta: Retrieved associative context delta (B, L, D)
            m_norm: Frobenius norm of active long-term memory
        """
        B, L, D = current_state.shape
        device = current_state.device
        dtype = current_state.dtype
        
        # Update running centroid
        with torch.no_grad():
            if self.running_centroid.device != device or self.running_centroid.dtype != dtype:
                self.running_centroid = self.running_centroid.to(device=device, dtype=dtype)
            batch_mean = current_state.mean(dim=(0, 1))
            self.running_centroid.copy_(0.95 * self.running_centroid + 0.05 * batch_mean)
            self.centroid_count.add_(1.0)
            
        # 1. Compute Dynamic Difference Vector Delta v = v_t - centroid
        diff_vectors = current_state - self.running_centroid.unsqueeze(0).unsqueeze(0)
        
        # 2. Project into low-rank factor space
        k_factors = self.proj_key(current_state)  # (B, L, rank)
        v_factors = self.proj_val(diff_vectors)    # (B, L, rank)
        
        # Outer-product associative update: (B, rank, rank)
        delta_m = torch.matmul(v_factors.transpose(1, 2), k_factors) / max(1, L)
        
        # 3. Adaptive Sliding-Window Decay lambda(t)
        # Prevents Hebbian eigenvalue explosion on long horizons
        self.step_counter += L
        adaptive_decay = self.base_decay * (1.0 + 0.5 * math.log1p(self.step_counter / float(self.capacity)))
        adaptive_decay = min(0.25, adaptive_decay)
        
        if self.m_long is None or self.m_long.shape[0] != B:
            self.m_long = torch.zeros(B, self.rank, self.rank, device=device, dtype=dtype)
            
        # Update long-term trace
        self.m_long = (1.0 - adaptive_decay) * self.m_long + self.base_lr * delta_m
        
        # 4. Recall: Query long-term associative matrix
        q_factors = self.proj_key(query)  # (B, L, rank)
        # (B, L, rank) @ (B, rank, rank)^T
        recalled_latent = torch.matmul(q_factors, self.m_long.transpose(1, 2))
        
        # Project back to native dimension with hyperbolic tangent bounding
        recalled_delta = 0.15 * torch.tanh(self.proj_out(recalled_latent))  # (B, L, D)
        
        m_norm = torch.norm(self.m_long, p="fro")
        return recalled_delta, m_norm
