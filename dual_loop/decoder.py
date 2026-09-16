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

    def calibrate_halting(
        self,
        sample_inputs: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
        percentile: Optional[float] = None,
        candidate_percentiles: Tuple[float, ...] = (25.0, 35.0, 50.0, 65.0, 75.0, 85.0),
        verbose: bool = False
    ) -> float:
        """
        Dynamically calibrates the halting threshold against the model's actual
        decoder predictive entropy distribution on validation samples.

        Modes:
        1. Fixed Quantile (default: 75th percentile / upper quartile):
           If `percentile` is specified (or when `target_labels is None`), sets the threshold
           directly to the specified quantile of step-1 predictive entropy.
        2. Automated Pareto Grid-Search:
           If `target_labels` is provided and `percentile is None`, sweeps across candidate
           percentiles to find the Pareto-optimal threshold balancing high accuracy with reduced
           pondering steps (effective K).
        """
        self.eval()
        with torch.no_grad():
            logits_k1, _ = self.forward(sample_inputs, k_steps=1, dynamic_halting=False)
            ent_k1 = self.outer_loop.halting_unit.calculate_entropy(logits_k1)

            if target_labels is not None and percentile is None:
                best_score = -float('inf')
                best_thresh = None
                best_p = 75.0

                for p in candidate_percentiles:
                    cand_thresh = torch.quantile(ent_k1, p / 100.0).item()
                    self.outer_loop.halting_unit.entropy_threshold = cand_thresh
                    logits_dyn, info = self.forward(sample_inputs, dynamic_halting=True)
                    acc = (logits_dyn.argmax(dim=-1) == target_labels).float().mean().item() * 100.0
                    eff_k = info["effective_k"]

                    # Pareto scoring: maximize accuracy, reward compute savings
                    max_k = float(self.outer_loop.max_ponder_steps)
                    compute_saving = (max_k - eff_k) / max_k if max_k > 0 else 0.0
                    score = acc + (compute_saving * 5.0)

                    if verbose:
                        print(f"[Calibration Search] P={p:4.1f}% -> Thresh={cand_thresh:.3f} | Acc={acc:5.1f}% | Avg K={eff_k:.2f} | Score={score:.2f}")

                    if score > best_score:
                        best_score = score
                        best_thresh = cand_thresh
                        best_p = p

                self.outer_loop.halting_unit.entropy_threshold = best_thresh
                if verbose:
                    print(f"[Calibration Search] Selected Pareto-optimal Percentile={best_p}% (Threshold={best_thresh:.3f} nats)")
                return best_thresh
            else:
                p = 75.0 if percentile is None else percentile
                calibrated = torch.quantile(ent_k1, p / 100.0).item()
                self.outer_loop.halting_unit.entropy_threshold = calibrated
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
        Forward pass with dual-loop cognition and per-sample dynamic halting.
        """
        B, S = input_ids.shape
        x_emb = self.embedding(input_ids) + self.pos_emb[:, :S, :]
        ctx = self.context_encoder(x_emb) # [B, S, D]
        
        # Extract query token representation for query-conditioning
        query_rep = ctx[:, query_token_pos, :] # [B, D]
        
        # 1. Compress context into Cognitive Working Memory (SRAM cache)
        cwm_memory = self.cwm(ctx) # [B, M, D]
        
        max_k = self.outer_loop.max_ponder_steps if k_steps is None else k_steps
        
        # =====================================================================
        # PATH A: Per-Sample Dynamic Halting (Validated from halting_audit.py)
        # =====================================================================
        if dynamic_halting:
            thresh = self.outer_loop.halting_unit.entropy_threshold
            active_mask = torch.ones(B, dtype=torch.bool, device=input_ids.device)
            steps_taken = torch.full((B,), float(max_k), dtype=torch.float, device=input_ids.device)
            final_logits = torch.zeros(B, self.vocab_size, device=input_ids.device)
            
            # Initialize latent thoughts
            H = self.outer_loop.initialize_thoughts(query_rep)
            H_anchor = H.clone()
            
            aux_logits_list = []
            step_entropies_list = []
            
            for k in range(1, max_k + 1):
                # 1. Execute single recurrent step of Outer Loop
                attn_self, _ = self.outer_loop.latent_self_attn(H, H, H)
                H = self.outer_loop.norm1(H + attn_self)
                H_cross = self.outer_loop.capacity_cross_attn(H, cwm_memory)
                H = self.outer_loop.norm2(H + H_cross + 0.1 * H_anchor)
                H = self.outer_loop.norm3(H + self.outer_loop.latent_mlp(H))
                
                # 2. Fuse soft prefix and decode through Inner Loop
                fused = torch.cat([H, ctx], dim=1)
                dec = self.inner_decoder(fused)
                logits_k = self.lm_head(dec[:, -1, :]) # [B, VocabSize]
                
                # 3. Compute predictive entropy per sample from actual decoder logits
                ent_k = self.outer_loop.halting_unit.calculate_entropy(logits_k) # [B]
                step_entropies_list.append(ent_k)
                
                if self.outer_loop.audit_probe is not None:
                    aux_logits_list.append(self.outer_loop.audit_probe(H[:, 0, :]))
                
                # 4. Per-sample halting: freeze predictions for confident sequences
                newly_halted = active_mask & (ent_k <= thresh)
                if newly_halted.any():
                    final_logits[newly_halted] = logits_k[newly_halted]
                    steps_taken[newly_halted] = float(k)
                    active_mask[newly_halted] = False
                
                # Early exit if 100% of samples in the batch have confident consensus
                if not active_mask.any():
                    break
                    
            # Any remaining unhalted samples take the final step's prediction
            if active_mask.any():
                final_logits[active_mask] = logits_k[active_mask]
                steps_taken[active_mask] = float(max_k)
                
            info = {
                "aux_logits": aux_logits_list,
                "step_entropies": step_entropies_list,
                "num_thoughts": H.shape[1],
                "steps_taken": steps_taken,
                "effective_k": steps_taken.mean().item()
            }
            return final_logits, info

        # =====================================================================
        # PATH B: Standard Static Pondering (Zero-Overhead for Training / Fixed K)
        # =====================================================================
        h_thought, aux_logits, step_entropies = self.outer_loop(
            query_rep=query_rep,
            memory=cwm_memory,
            k_steps=k_steps,
            dynamic_halting=False,
            return_aux=return_aux
        ) # [B, L_thought, D]
        
        fused_sequence = torch.cat([h_thought, ctx], dim=1)
        decoded = self.inner_decoder(fused_sequence)
        final_logits = self.lm_head(decoded[:, -1, :])
        
        info = {
            "aux_logits": aux_logits,
            "step_entropies": step_entropies,
            "num_thoughts": h_thought.shape[1],
            "steps_taken": torch.full((B,), float(max_k), device=input_ids.device),
            "effective_k": float(max_k)
        }
        return final_logits, info
