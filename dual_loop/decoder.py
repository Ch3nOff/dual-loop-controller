import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .memory import CognitiveWorkingMemory
from .controller import RecurrentLatentController
from .evidential import EvidentialEpistemicGate
from .open_concept import OpenConceptSynthesizer

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
        entropy_threshold: float = 1.30,
        padding_idx: Optional[int] = None,
        is_causal: bool = False,
        enable_critique: bool = True,
        use_learned_halting: bool = False,
        lambda_prior: float = 0.5,
        tau_halt: float = 0.75,
        use_ddm_halting: bool = False,
        ddm_theta_0: float = 3.0,
        ddm_gamma: float = 0.5,
        ddm_min_theta: float = 0.5,
        enable_plasticity: bool = True,
        plastic_rank: int = 32,
        plastic_lr: float = 0.15,
        use_evidential_gate: bool = True,
        use_open_concept: bool = True,
        tau_novelty: float = 0.40,
        tau_unseen: float = 0.65
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.is_causal = is_causal
        
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
            entropy_threshold=entropy_threshold,
            enable_critique=enable_critique,
            use_learned_halting=use_learned_halting,
            lambda_prior=lambda_prior,
            tau_halt=tau_halt,
            use_ddm_halting=use_ddm_halting,
            ddm_theta_0=ddm_theta_0,
            ddm_gamma=ddm_gamma,
            ddm_min_theta=ddm_min_theta,
            enable_plasticity=enable_plasticity,
            plastic_rank=plastic_rank,
            plastic_lr=plastic_lr
        )
        
        # Evidential Epistemic Gate (Subjective Logic Dirichlet decomposition)
        self.use_evidential_gate = use_evidential_gate
        self.evidential_gate = EvidentialEpistemicGate(
            d_model=d_model,
            tau_novelty=tau_novelty,
            tau_unseen=tau_unseen
        ) if use_evidential_gate else None

        # Open-Concept Prototype Synthesizer (Unprecedented / Non-Vocabulary continuous vectors)
        self.use_open_concept = use_open_concept
        self.open_concept_synthesizer = OpenConceptSynthesizer(
            d_model=d_model,
            tau_unseen=tau_unseen
        ) if use_open_concept else None
        
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
        return_aux: bool = False,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with dual-loop cognition and per-sample dynamic halting.
        """
        B, S = input_ids.shape
        if S > self.pos_emb.size(1):
            raise ValueError(
                f"Sequence length S={S} exceeds maximum supported positional embedding length {self.pos_emb.size(1)}."
            )

        x_emb = self.embedding(input_ids) + self.pos_emb[:, :S, :]
        ctx = self.context_encoder(x_emb) # [B, S, D]
        
        # Extract query token representation with sequence length bounds guard (TENSOR-04)
        pos = query_token_pos if query_token_pos >= 0 else S + query_token_pos
        pos = max(0, min(S - 1, pos))
        query_rep = ctx[:, pos, :] # [B, D]
        
        # 1. Compress context into Cognitive Working Memory (SRAM cache) with padding mask (ARCH-05)
        if padding_mask is None and self.padding_idx is not None:
            padding_mask = (input_ids == self.padding_idx)
        cwm_memory = self.cwm(ctx, key_padding_mask=padding_mask) # [B, M, D]
        
        # Clear outer loop mutable instance state at forward entry (ARCH-02 / NEW-02)
        self.outer_loop.reset_state()

        # 1b. Evidential Epistemic Self-Recognition (Dirichlet vacuity decomposition)
        evidential_telem = {}
        vacuity_u = None
        if getattr(self, "evidential_gate", None) is not None:
            evidential_telem = self.evidential_gate(h=query_rep)
            vacuity_u = evidential_telem.get("vacuity_u")

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
            H_anchor = H.detach() if not self.training else H.clone()
            
            aux_logits_list = []
            step_entropies_list = []
            lambdas_list = []
            error_norms_list = []
            h_prev_primary = None
            H_prev1 = None
            H_prev2 = None
            
            for k in range(1, max_k + 1):
                H_prev = H.clone()

                # 1. Execute unified single recurrent step of Outer Loop (NEW-01)
                # Unifies LatentCritiqueRefinementUnit, learned anchor gate, and LearnedHaltingGate
                H, err_norm, lam_k = self.outer_loop.step_deliberation(
                    H=H,
                    H_anchor=H_anchor,
                    memory=cwm_memory,
                    H_prev=H_prev,
                    h_prev_primary=h_prev_primary,
                    u_epistemic=vacuity_u,
                    step_idx=k,
                    max_steps=max_k,
                    H_prev2=H_prev2
                )
                H_prev2 = H_prev1
                H_prev1 = H_prev

                if err_norm is not None:
                    error_norms_list.append(err_norm)
                if lam_k is not None:
                    lambdas_list.append(lam_k)
                    h_prev_primary = H[:, 0, :].detach()
                
                # 2. Fuse soft prefix and decode through Inner Loop
                fused = torch.cat([H, ctx], dim=1)
                tot_len = fused.size(1)
                mask = nn.Transformer.generate_square_subsequent_mask(tot_len, device=input_ids.device) if self.is_causal else None
                dec = self.inner_decoder(fused, mask=mask)
                logits_k = self.lm_head(dec[:, -1, :]) # [B, VocabSize]
                
                # 3. Compute predictive entropy per sample from actual decoder logits
                ent_k = self.outer_loop.halting_unit.calculate_entropy(logits_k) # [B]
                step_entropies_list.append(ent_k)
                
                if self.outer_loop.audit_probe is not None:
                    aux_logits_list.append(self.outer_loop.audit_probe(H[:, 0, :]))
                
                # 4. Per-sample halting: check halting criteria (learned halting gate, DDM halting, or predictive entropy)
                if self.outer_loop.learned_halting_gate is not None and lam_k is not None and not self.training:
                    halt_decision = self.outer_loop.learned_halting_gate.should_halt_inference(lam_k)
                    newly_halted = active_mask & halt_decision
                elif getattr(self.outer_loop, 'ddm_halting', None) is not None and not self.training:
                    halt_decision, ev_ddm, _ = self.outer_loop.ddm_halting.should_halt(logits_k, k)
                    newly_halted = active_mask & halt_decision
                else:
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
                
            self.outer_loop.last_lambdas = lambdas_list
            self.outer_loop.last_error_norms = error_norms_list

            concept_telem_a = {}
            if getattr(self, "open_concept_synthesizer", None) is not None and vacuity_u is not None:
                last_err_a = (H[:, 0, :] - query_rep)
                _, concept_telem_a = self.open_concept_synthesizer(
                    h_anchor=query_rep,
                    discrepancy=last_err_a,
                    vacuity_u=vacuity_u
                )

            info = {
                "aux_logits": aux_logits_list,
                "step_entropies": step_entropies_list,
                "num_thoughts": H.shape[1],
                "steps_taken": steps_taken,
                "effective_k": steps_taken.mean().item(),
                "error_norms": error_norms_list,
                "halting_lambdas": lambdas_list,
                "ddm_evidences": [e.detach().cpu() for e in self.outer_loop.last_ddm_evidences] if getattr(self.outer_loop, 'last_ddm_evidences', None) else [],
                "epistemic_vacuity": [float(v) for v in vacuity_u.reshape(-1).detach().cpu().tolist()] if vacuity_u is not None else [],
                "plastic_trace_norm": float(self.outer_loop.plastic_unit.last_m_fast.norm().item()) if (getattr(self.outer_loop, "plastic_unit", None) is not None and self.outer_loop.plastic_unit.last_m_fast is not None) else 0.0,
                "synthesized_concepts": concept_telem_a.get("synthesized_count", 0)
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
            return_aux=return_aux,
            u_epistemic=vacuity_u
        ) # [B, L_thought, D]

        concept_telem_b = {}
        if getattr(self, "open_concept_synthesizer", None) is not None and vacuity_u is not None:
            last_err_b = (h_thought[:, 0, :] - query_rep)
            proto_c, concept_telem_b = self.open_concept_synthesizer(
                h_anchor=query_rep,
                discrepancy=last_err_b,
                vacuity_u=vacuity_u
            )
            if proto_c is not None and proto_c.any():
                h_thought = torch.cat([proto_c, h_thought], dim=1)
        
        fused_sequence = torch.cat([h_thought, ctx], dim=1)
        tot_len = fused_sequence.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(tot_len, device=input_ids.device) if self.is_causal else None
        decoded = self.inner_decoder(fused_sequence, mask=mask)
        final_logits = self.lm_head(decoded[:, -1, :])
        
        info = {
            "aux_logits": aux_logits,
            "step_entropies": step_entropies,
            "num_thoughts": h_thought.shape[1],
            "steps_taken": torch.full((B,), float(max_k), dtype=torch.float, device=input_ids.device),
            "effective_k": float(max_k),
            "error_norms": self.outer_loop.last_error_norms,
            "halting_lambdas": self.outer_loop.last_lambdas,
            "epistemic_vacuity": [float(v) for v in vacuity_u.reshape(-1).detach().cpu().tolist()] if vacuity_u is not None else [],
            "plastic_trace_norm": float(self.outer_loop.plastic_unit.last_m_fast.norm().item()) if (getattr(self.outer_loop, "plastic_unit", None) is not None and self.outer_loop.plastic_unit.last_m_fast is not None) else 0.0,
            "synthesized_concepts": concept_telem_b.get("synthesized_count", 0)
        }
        return final_logits, info
