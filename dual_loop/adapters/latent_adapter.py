import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union, List
from ..controller import RecurrentLatentController
from ..memory import CognitiveWorkingMemory, EpisodicMemoryBuffer
from ..verification import (
    HypothesisVerificationGate, UncertaintySurpriseGate, ContrastiveEvidenceAccumulator,
    DirectionalSafetyProjection, AdaptiveSurpriseThreshold, CognitiveConflictMonitor, AntiHyperSkepticismFilter
)
from ..evidential import EvidentialEpistemicGate
from ..open_concept import OpenConceptSynthesizer

class LatentDeliberationAdapter(nn.Module):
    """
    Plug-and-Play Latent Deliberation Adapter for Pretrained LLMs.
    
    Can be inserted into any standard Transformer backbone (e.g., Llama-3,
    Qwen-2.5, Mistral, GPT) at an intermediate layer (e.g., layer L/2).
    
    Operation:
    1. Intercepts hidden states H_l in the native pretrained hidden dimension D.
    2. Takes the last non-padding token (query/instruction representation).
    3. Runs K steps of recurrent latent deliberation using weight-tied
       continuous attention, without emitting discrete tokens.
    4. Injects deliberated thoughts back into the residual stream or as a soft prefix.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        num_thought_tokens: int = 4,
        max_ponder_steps: int = 3,
        capacity_factor: float = 0.5,
        num_cwm_slots: int = 16,
        adapter_mode: str = "residual", # "residual" or "prefix"
        gate_alpha_init: float = 0.05,
        vocab_size: Optional[int] = None,
        enable_critique: bool = True,
        use_learned_halting: bool = False,
        use_hypothesis_verification: bool = True,
        lambda_prior: float = 0.5,
        tau_halt: float = 0.75,
        use_surprise_gate: bool = True,
        tau_surprise: float = 0.01,
        surprise_temperature: float = 0.05,
        use_contrastive_evidence: bool = True,
        tau_contrast: float = 0.1,
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
        self.d_model = d_model
        self.num_thought_tokens = num_thought_tokens
        self.adapter_mode = adapter_mode
        self.max_ponder_steps = max_ponder_steps
        self.enable_critique = enable_critique
        self.use_learned_halting = use_learned_halting
        self.use_hypothesis_verification = use_hypothesis_verification
        self.use_surprise_gate = use_surprise_gate
        self.use_contrastive_evidence = use_contrastive_evidence
        self.use_evidential_gate = use_evidential_gate
        self.use_open_concept = use_open_concept
        object.__setattr__(self, "_lm_head_ref", None)
        
        # Memory compressor
        self.cwm = CognitiveWorkingMemory(d_model=d_model, num_slots=num_cwm_slots, n_heads=n_heads)
        
        # System 2 recurrent controller operating natively in d_model
        self.controller = RecurrentLatentController(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * 2,
            num_thought_tokens=num_thought_tokens,
            max_ponder_steps=max_ponder_steps,
            capacity_factor=capacity_factor,
            vocab_size=vocab_size,
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
        
        # Evidential Epistemic Self-Recognition Gate (Dirichlet uncertainty decomposition)
        if use_evidential_gate:
            self.evidential_gate = EvidentialEpistemicGate(
                d_model=d_model,
                tau_novelty=tau_novelty,
                tau_unseen=tau_unseen
            )
        else:
            self.evidential_gate = None

        # Open-Concept Continuous Manifold Synthesizer (Unprecedented concept expansion)
        if use_open_concept:
            self.open_concept_synthesizer = OpenConceptSynthesizer(
                d_model=d_model,
                tau_unseen=tau_unseen
            )
        else:
            self.open_concept_synthesizer = None

        # Counterfactual Hypothesis-Testing & Conservative Verification Gate (Anti-Overthinking)
        if use_hypothesis_verification:
            self.hypothesis_gate = HypothesisVerificationGate(d_model=d_model)
        else:
            self.hypothesis_gate = None

        # Task-Aware Uncertainty-Gated Bypass (Surprise Gate based on JSD)
        if use_surprise_gate:
            self.surprise_gate = UncertaintySurpriseGate(
                d_model=d_model,
                tau_surprise=tau_surprise,
                temperature=surprise_temperature,
                vocab_size=vocab_size
            )
        else:
            self.surprise_gate = None

        # Contrastive Distractor Suppression in Latent Space
        if use_contrastive_evidence:
            self.contrastive_accumulator = ContrastiveEvidenceAccumulator(
                d_model=d_model,
                n_heads=n_heads,
                tau_contrast=tau_contrast
            )
        else:
            self.contrastive_accumulator = None
        
        if adapter_mode == "residual":
            self.residual_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, d_model)
            )
            # Small scale initialization to stabilize residual injection
            nn.init.normal_(self.residual_proj[-1].weight, std=0.01)
            nn.init.zeros_(self.residual_proj[-1].bias)
            # ReZero learnable gate initialized to gate_alpha_init (stabilized residual injection)
            self.gate_alpha = nn.Parameter(torch.tensor([float(gate_alpha_init)]))

            # Directional Safety Projection: prevents deliberation from flipping
            # base model's top-1 prediction (fixes BBH Logical Deduction regression)
            self.directional_safety = DirectionalSafetyProjection()
            
            # Adaptive Surprise Threshold: per-sample tau based on base model entropy
            # (fixes BBH Date Understanding degraded sample)
            self.adaptive_tau = AdaptiveSurpriseThreshold(
                tau_base=0.005, gamma=0.05, H_ref=3.0
            )

        # Persistent Episodic Meta-Cognitive Memory Bank (Hippocampal Consolidation)
        self.episodic_memory = EpisodicMemoryBuffer(d_model=d_model)
        self.continual_mode: bool = False
        
        # Unified Cognitive Architecture Components (ACC & Rational Filter)
        self.conflict_monitor = CognitiveConflictMonitor()
        self.anti_skepticism_filter = AntiHyperSkepticismFilter()

    def set_continual_mode(self, enabled: bool = True, decay: float = 0.90):
        """Enables continual learning across trials/runs without amnesia."""
        self.continual_mode = enabled
        self.controller.set_continual_mode(enabled=enabled, decay=decay)

    def reset_state(self, force: bool = False):
        """Resets controller mutable state (or soft decays if continual_mode=True)."""
        self.controller.reset_state(force=force)

    def set_lm_head(self, lm_head: nn.Module):
        """Binds output projection for surprise gate AND directional safety projection."""
        object.__setattr__(self, "_lm_head_ref", lm_head)
        if self.surprise_gate is not None:
            self.surprise_gate.set_lm_head(lm_head)

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_steps: Optional[int] = None,
        query_idx: Union[int, torch.Tensor, List[int]] = -1,
        dynamic_halting: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
        candidate_embeds: Optional[Any] = None,
        critique_vector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            hidden_states: [B, SeqLen, D] Hidden activations from intermediate layer l.
            k_steps: Number of latent deliberation steps (0 bypasses pondering completely).
            query_idx: Position of the query/instruction token (int, or per-sample [B] Tensor/List).
            dynamic_halting: Whether to halt pondering early upon entropy/latent convergence.
            key_padding_mask: Optional [B, SeqLen] boolean mask (True for padded positions).
            candidate_embeds: Optional multiple-choice candidate options embeddings for contrastive suppression.
            critique_vector: Optional [B, D] counterfactual critique vector for autonomous self-correction.
        Returns:
            enhanced_states: [B, SeqLen', D] Modified hidden states for layer l+1.
            telemetry: Diagnostic information.
        """
        steps = self.max_ponder_steps if k_steps is None else k_steps
        
        # Zero pondering steps: exact identity bypass (System 1 mode)
        if steps == 0:
            self.controller.reset_state()
            telemetry = {
                "num_thoughts": 0,
                "ponder_steps": 0,
                "effective_k": 0.0,
                "step_entropies": [],
                "error_norms": [],
                "halting_lambdas": [],
                "acceptance_beta": [],
                "evidence_gain": [],
                "surprise_gate": [],
                "surprise_jsd": [],
                "contrastive_scores": [],
                "ddm_evidences": [],
                "adapter_mode": self.adapter_mode,
                "bypassed": True
            }
            return hidden_states, telemetry

        B, S, D = hidden_states.shape
        
        # 1. Extract query anchor (support int, per-sample Tensor [B], or List[int]) - ARCH-03
        if isinstance(query_idx, int):
            idx_int = query_idx if query_idx >= 0 else S + query_idx
            idx_int = max(0, min(S - 1, idx_int))
            query_rep = hidden_states[:, idx_int, :] # [B, D]
            is_scalar_idx = True
        else:
            if isinstance(query_idx, list):
                idx_tensor = torch.tensor(query_idx, device=hidden_states.device, dtype=torch.long)
            elif isinstance(query_idx, torch.Tensor):
                idx_tensor = query_idx.to(device=hidden_states.device, dtype=torch.long)
            else:
                raise TypeError(f"query_idx must be int, torch.Tensor, or List[int], got {type(query_idx)}")
            
            # Normalize negative indices & clamp
            idx_tensor = torch.where(idx_tensor < 0, S + idx_tensor, idx_tensor)
            idx_tensor = torch.clamp(idx_tensor, 0, S - 1)
            batch_idx = torch.arange(B, device=hidden_states.device)
            query_rep = hidden_states[batch_idx, idx_tensor, :] # [B, D]
            is_scalar_idx = False
        
        # 2. Compress context into working memory (pass key_padding_mask - ARCH-05)
        memory = self.cwm(hidden_states, key_padding_mask=key_padding_mask) # [B, M, D]
        
        # 2b. Evidential Epistemic Self-Recognition (Dirichlet vacuity of evidence)
        evidential_telem = {}
        vacuity_u = None
        if self.evidential_gate is not None:
            evidential_telem = self.evidential_gate(h=query_rep)
            vacuity_u = evidential_telem.get("vacuity_u")

        # 2c. Self-Correction / Critique Falsification: nudge fast weights away from ambiguous prior attractor
        critique_telem = {}
        if critique_vector is not None and self.controller.plastic_unit is not None:
            # Check if this query is already settled in memory to prevent hyper-skepticism
            settled_ep = self.episodic_memory.recall_settled(query_rep, sim_threshold=0.85) if hasattr(self, "episodic_memory") else None
            is_settled = (settled_ep is not None)
            pass1_m = settled_ep.get("margin", 1.0) if settled_ep else 0.0
            
            vac_val = float(vacuity_u.mean().item()) if isinstance(vacuity_u, torch.Tensor) else 0.5
            should_critique = self.anti_skepticism_filter.should_apply_critique(
                is_settled=is_settled,
                pass1_margin=pass1_m,
                vacuity_u=vac_val
            )
            
            if should_critique:
                init_t = self.controller.initialize_thoughts(query_rep)
                critique_telem = self.controller.plastic_unit.apply_critique_falsification(
                    thoughts=init_t,
                    critique_vector=critique_vector,
                    u_epistemic=vacuity_u
                )
                critique_telem["critique_blocked"] = False
            else:
                critique_telem["critique_blocked"] = True

        # 3. Deliberate in latent space (passing u_epistemic for in-situ plastic fast-weight adaptation)
        h_thought, aux, entropies = self.controller(
            query_rep=query_rep,
            memory=memory,
            k_steps=steps,
            dynamic_halting=dynamic_halting,
            return_aux=True,
            u_epistemic=vacuity_u
        ) # [B, L_thought, D]
        
        # 4. Hypothesis Verification & Anti-Overthinking Gate
        if self.hypothesis_gate is not None:
            h_thought, beta_gate, v_telem = self.hypothesis_gate(
                thoughts=h_thought,
                query_rep=query_rep,
                memory=memory
            )
        else:
            beta_gate = torch.ones(B, 1, 1, device=h_thought.device)
            v_telem = {}

        # 5. Contrastive Option Distractor Suppression & Directional Delta
        contrastive_scores_list = []
        if self.adapter_mode == "residual":
            raw_delta = self.residual_proj(h_thought[:, 0, :])
            if candidate_embeds is not None and self.contrastive_accumulator is not None:
                c_delta, contrastive_scores = self.contrastive_accumulator(
                    thought=h_thought[:, 0, :],
                    candidate_embeds=candidate_embeds,
                    query_anchor=query_rep
                )
                contrastive_scores_list = contrastive_scores.detach().cpu().tolist()
                # Contrastive refinement on top of trained deliberation projection
                raw_delta = raw_delta + 0.35 * c_delta

            # 5b. Open-Concept Manifold Synthesis (for unprecedented/unseen concepts)
            concept_telem = {}
            if self.open_concept_synthesizer is not None and vacuity_u is not None:
                last_err = (h_thought[:, 0, :] - query_rep)
                proto_c, concept_telem = self.open_concept_synthesizer(
                    h_anchor=query_rep,
                    discrepancy=last_err,
                    vacuity_u=vacuity_u
                )
                if proto_c is not None and raw_delta is not None and proto_c.any():
                    raw_delta = raw_delta + 0.15 * proto_c.squeeze(1)
        else:
            raw_delta = None
            concept_telem = {}

        # 6. Task-Aware Uncertainty-Gated Bypass (Surprise Gate with Adaptive Threshold)
        surprise_gate_tensor = torch.ones(B, 1, device=hidden_states.device)
        surprise_jsd_tensor = torch.zeros(B, device=hidden_states.device)
        if self.surprise_gate is not None and self.adapter_mode == "residual" and raw_delta is not None:
            # Compute per-sample adaptive tau from base model's confidence level
            if hasattr(self, 'adaptive_tau') and self._lm_head_ref is not None:
                adaptive_tau = self.adaptive_tau.compute_adaptive_tau(query_rep, self._lm_head_ref)  # [B]
                # Override the global tau with per-sample adaptive threshold
                saved_tau = self.surprise_gate.tau_surprise
                # Use per-sample gate: sigmoid((JSD - tau_i) / temp) for each sample
                surprise_gate_tensor, surprise_jsd_tensor = self.surprise_gate.compute_surprise_gate(
                    h_base=query_rep,
                    h_delib=query_rep + raw_delta
                )
                # Re-compute gate with per-sample adaptive thresholds
                adaptive_gate = torch.sigmoid(
                    (surprise_jsd_tensor - adaptive_tau) / self.surprise_gate.temperature
                ).unsqueeze(-1)  # [B, 1]
                surprise_gate_tensor = adaptive_gate
            else:
                surprise_gate_tensor, surprise_jsd_tensor = self.surprise_gate.compute_surprise_gate(
                    h_base=query_rep,
                    h_delib=query_rep + raw_delta
                )

        # 7. Directional Safety Projection (prevent flipping base model's top-1 prediction)
        safety_telemetry = {}
        if self.adapter_mode == "residual" and raw_delta is not None:
            if hasattr(self, 'directional_safety') and self._lm_head_ref is not None:
                raw_delta, safety_telemetry = self.directional_safety(
                    delta=raw_delta,
                    h_base=query_rep,
                    lm_head=self._lm_head_ref
                )

        # 8. Metacognitive Error-Reflection Damping (Autonomous self-divergence detection)
        # Evaluates whether the deliberation process converged toward context consensus (e_K <= e_1)
        # or drifted/diverged (e_K > e_1). If diverging, damp delta proportionally.
        if len(self.controller.last_error_norms) >= 2:
            e_1 = self.controller.last_error_norms[0]  # [B]
            e_K = self.controller.last_error_norms[-1]  # [B]
            discrepancy_drift = torch.clamp(e_K - e_1, min=0.0)  # [B]
            kappa_metacog = torch.exp(-discrepancy_drift).unsqueeze(-1)  # [B, 1]
        else:
            kappa_metacog = torch.ones(B, 1, device=hidden_states.device)

        # 9. Integrate into stream
        if self.adapter_mode == "prefix":
            # Prepend thoughts as soft prefix: [B, L_thought + S, D]
            enhanced = torch.cat([h_thought, hidden_states], dim=1)
        elif self.adapter_mode == "residual":
            # Add thoughts as a ReZero-gated residual onto the query token
            # Gated jointly by beta_gate (hypothesis test), surprise_gate (JSD bypass),
            # and kappa_metacog (metacognitive error-reflection convergence)
            # Directional safety already applied above to raw_delta
            scale = torch.tanh(self.gate_alpha)
            # Epistemic Vacuity Gating Modulation (Subjective Logic Epistemic Modesty)
            if vacuity_u is not None:
                vac_t = vacuity_u.to(device=hidden_states.device, dtype=hidden_states.dtype)
                eta_epistemic = torch.clamp(1.0 - torch.clamp(vac_t - 0.50, min=0.0) / 0.50, min=0.20, max=1.0).view(B, 1)
            else:
                eta_epistemic = 1.0
            delta = scale * surprise_gate_tensor * beta_gate.reshape(B, 1) * kappa_metacog * eta_epistemic * raw_delta # [B, D]
            enhanced = hidden_states.clone()
            if is_scalar_idx:
                enhanced[:, idx_int:idx_int+1, :] = enhanced[:, idx_int:idx_int+1, :] + delta.unsqueeze(1)
            else:
                enhanced[batch_idx, idx_tensor, :] = enhanced[batch_idx, idx_tensor, :] + delta
        else:
            raise ValueError(f"Unknown adapter mode: {self.adapter_mode}")
            
        telemetry = {
            "num_thoughts": self.num_thought_tokens,
            "ponder_steps": steps,
            "effective_k": float(len(entropies)) if (dynamic_halting and entropies) else float(steps),
            "step_entropies": [e.detach().cpu() for e in entropies] if entropies else [],
            "error_norms": [e.detach().cpu() for e in self.controller.last_error_norms] if self.controller.last_error_norms else [],
            "halting_lambdas": [l.detach().cpu() for l in self.controller.last_lambdas] if self.controller.last_lambdas else [],
            "acceptance_beta": [float(b) for b in beta_gate.reshape(-1).detach().cpu().tolist()] if self.hypothesis_gate is not None else [1.0] * B,
            "evidence_gain": [float(eg.item()) for eg in v_telem["evidence_gain"].cpu()] if "evidence_gain" in v_telem else [],
            "surprise_gate": [float(g) for g in surprise_gate_tensor.reshape(-1).detach().cpu().tolist()],
            "surprise_jsd": [float(j) for j in surprise_jsd_tensor.reshape(-1).detach().cpu().tolist()],
            "contrastive_scores": contrastive_scores_list,
            "ddm_evidences": [e.detach().cpu() for e in self.controller.last_ddm_evidences] if self.controller.last_ddm_evidences else [],
            "metacog_damping": [float(m) for m in kappa_metacog.reshape(-1).detach().cpu().tolist()],
            "directional_safety": {k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v for k, v in safety_telemetry.items()} if safety_telemetry else {},
            "epistemic_vacuity": [float(v) for v in vacuity_u.reshape(-1).detach().cpu().tolist()] if vacuity_u is not None else [],
            "plastic_trace_norm": float(self.controller.plastic_unit.last_m_fast.norm().item()) if (self.controller.plastic_unit is not None and self.controller.plastic_unit.last_m_fast is not None) else 0.0,
            "synthesized_concepts": concept_telem.get("synthesized_count", 0),
            "gate_scale": float(torch.tanh(self.gate_alpha).item()) if hasattr(self, "gate_alpha") else 1.0,
            "adapter_mode": self.adapter_mode,
            "bypassed": False
        }
        return enhanced, telemetry
