import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .memory import CognitiveWorkingMemory
from .controller import RecurrentLatentController

class DualLoopTransformer(nn.Module):
    """
    Complete Dual-Loop Cognitive Controller Model.
    
    Integrates:
    - Input Embedding & Context Encoding
    - Cognitive Working Memory (CWM) Compression (SRAM optimization)
    - System 2: Recurrent Latent Executive Controller (Outer Loop)
    - System 1: Soft-Prefix Conditioned Autoregressive Decoder (Inner Loop)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        num_decoder_layers: int = 2,
        num_thought_tokens: int = 4,
        num_cwm_slots: int = 16,
        max_ponder_steps: int = 3,
        capacity_factor: float = 0.5,
        entropy_threshold: float = 1.30
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 1. Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        
        # 2. Context Encoder (Shallow representation)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 3. Cognitive Working Memory
        self.cwm = CognitiveWorkingMemory(d_model=d_model, num_slots=num_cwm_slots, n_heads=n_heads)
        
        # 4. System 2: Outer Loop
        self.outer_loop = RecurrentLatentController(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_thought_tokens=num_thought_tokens,
            max_ponder_steps=max_ponder_steps,
            vocab_size=vocab_size,
            capacity_factor=capacity_factor,
            entropy_threshold=entropy_threshold
        )
        
        # 5. System 1: Inner Loop Decoder
        dec_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.inner_decoder = nn.TransformerEncoder(dec_layer, num_layers=num_decoder_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def calibrate_halting(self, sample_inputs: torch.Tensor, percentile: float = 50.0):
        """
        Dynamically calibrates the halting threshold against the model's actual
        entropy distribution on real validation samples.
        """
        self.eval()
        with torch.no_grad():
            logits, info = self.forward(sample_inputs, return_aux=True)
            if info["aux_logits"]:
                probe_logits = info["aux_logits"][-1]
            else:
                probe_logits = logits
            calibrated = self.outer_loop.halting_unit.calibrate_threshold(probe_logits, percentile=percentile)
            return calibrated

    def forward(
        self,
        input_ids: torch.Tensor,
        k_steps: Optional[int] = None,
        query_token_pos: int = -2,
        dynamic_halting: bool = False,
        return_aux: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with dual-loop cognition.
        
        Args:
            input_ids: [B, SeqLen] input tokens
            k_steps: Optional override for outer loop ponder steps
            query_token_pos: Index of the query token in prompt (-2 by default)
            dynamic_halting: Enable predictive entropy early halting
            return_aux: Return intermediate probe outputs and entropies
            
        Returns:
            logits: [B, VocabSize] prediction on final target token
            info: Dictionary containing auxiliary outputs and telemetry
        """
        B, S = input_ids.shape
        x_emb = self.embedding(input_ids) + self.pos_emb[:, :S, :]
        ctx = self.context_encoder(x_emb) # [B, S, D]
        
        # Extract query token representation for query-conditioning
        query_rep = ctx[:, query_token_pos, :] # [B, D]
        
        # 1. Compress context into Cognitive Working Memory (SRAM cache)
        cwm_memory = self.cwm(ctx) # [B, M, D]
        
        # 2. Outer Loop: System 2 Pondering in Continuous Latent Space
        h_thought, aux_logits, step_entropies = self.outer_loop(
            query_rep=query_rep,
            memory=cwm_memory,
            k_steps=k_steps,
            dynamic_halting=dynamic_halting,
            return_aux=return_aux
        ) # [B, L_thought, D]
        
        # 3. Fuse Soft-Prefix: Prepend thoughts directly to context
        # Shape: [B, L_thought + S, D]
        fused_sequence = torch.cat([h_thought, ctx], dim=1)
        
        # 4. Inner Loop: System 1 Language Generator
        decoded = self.inner_decoder(fused_sequence)
        
        # 5. LM Head prediction from the final sequence position
        final_logits = self.lm_head(decoded[:, -1, :])
        
        info = {
            "aux_logits": aux_logits,
            "step_entropies": step_entropies,
            "num_thoughts": h_thought.shape[1],
            "effective_k": len(step_entropies) if step_entropies else (self.outer_loop.max_ponder_steps if k_steps is None else k_steps)
        }
        return final_logits, info
