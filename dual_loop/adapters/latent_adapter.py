import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union, List
from ..controller import RecurrentLatentController
from ..memory import CognitiveWorkingMemory

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
        lambda_prior: float = 0.5,
        tau_halt: float = 0.75
    ):
        super().__init__()
        self.d_model = d_model
        self.num_thought_tokens = num_thought_tokens
        self.adapter_mode = adapter_mode
        self.max_ponder_steps = max_ponder_steps
        self.enable_critique = enable_critique
        self.use_learned_halting = use_learned_halting
        
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
            tau_halt=tau_halt
        )
        
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_steps: Optional[int] = None,
        query_idx: Union[int, torch.Tensor, List[int]] = -1,
        dynamic_halting: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            hidden_states: [B, SeqLen, D] Hidden activations from intermediate layer l.
            k_steps: Number of latent deliberation steps (0 bypasses pondering completely).
            query_idx: Position of the query/instruction token (int, or per-sample [B] Tensor/List).
            dynamic_halting: Whether to halt pondering early upon entropy/latent convergence.
            key_padding_mask: Optional [B, SeqLen] boolean mask (True for padded positions).
        Returns:
            enhanced_states: [B, SeqLen', D] Modified hidden states for layer l+1.
            telemetry: Diagnostic information.
        """
        steps = self.max_ponder_steps if k_steps is None else k_steps
        
        # Zero pondering steps: exact identity bypass (System 1 mode)
        if steps == 0:
            telemetry = {
                "num_thoughts": 0,
                "ponder_steps": 0,
                "effective_k": 0.0,
                "step_entropies": [],
                "error_norms": [],
                "halting_lambdas": [],
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
        
        # 3. Deliberate in latent space
        h_thought, aux, entropies = self.controller(
            query_rep=query_rep,
            memory=memory,
            k_steps=steps,
            dynamic_halting=dynamic_halting,
            return_aux=True
        ) # [B, L_thought, D]
        
        # 4. Integrate into stream
        if self.adapter_mode == "prefix":
            # Prepend thoughts as soft prefix: [B, L_thought + S, D]
            enhanced = torch.cat([h_thought, hidden_states], dim=1)
        elif self.adapter_mode == "residual":
            # Add thoughts as a ReZero-gated residual onto the query token
            scale = torch.tanh(self.gate_alpha)
            delta = scale * self.residual_proj(h_thought[:, 0, :]) # [B, D]
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
            "gate_scale": float(torch.tanh(self.gate_alpha).item()) if hasattr(self, "gate_alpha") else 1.0,
            "adapter_mode": self.adapter_mode,
            "bypassed": False
        }
        return enhanced, telemetry
