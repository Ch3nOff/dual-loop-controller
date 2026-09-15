import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
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
        adapter_mode: str = "prefix" # "prefix" or "residual"
    ):
        super().__init__()
        self.d_model = d_model
        self.num_thought_tokens = num_thought_tokens
        self.adapter_mode = adapter_mode
        
        # Memory compressor
        self.cwm = CognitiveWorkingMemory(d_model=d_model, num_slots=num_cwm_slots, n_heads=n_heads)
        
        # System 2 recurrent controller operating natively in d_model
        self.controller = RecurrentLatentController(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * 2,
            num_thought_tokens=num_thought_tokens,
            max_ponder_steps=max_ponder_steps,
            capacity_factor=capacity_factor
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_steps: Optional[int] = None,
        query_idx: int = -1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            hidden_states: [B, SeqLen, D] Hidden activations from intermediate layer l.
            k_steps: Number of latent deliberation steps.
            query_idx: Position of the query/instruction token (default -1).
        Returns:
            enhanced_states: [B, SeqLen', D] Modified hidden states for layer l+1.
            telemetry: Diagnostic information.
        """
        B, S, D = hidden_states.shape
        
        # 1. Extract query anchor (normalize negative index)
        if query_idx < 0:
            query_idx = S + query_idx
        query_rep = hidden_states[:, query_idx, :] # [B, D]
        
        # 2. Compress context into working memory
        memory = self.cwm(hidden_states) # [B, M, D]
        
        # 3. Deliberate in latent space
        h_thought, aux, entropies = self.controller(
            query_rep=query_rep,
            memory=memory,
            k_steps=k_steps
        ) # [B, L_thought, D]
        
        # 4. Integrate into stream
        if self.adapter_mode == "prefix":
            # Prepend thoughts as soft prefix: [B, L_thought + S, D]
            enhanced = torch.cat([h_thought, hidden_states], dim=1)
        elif self.adapter_mode == "residual":
            # Add thoughts as a gating residual onto the query token
            delta = self.residual_proj(h_thought[:, 0, :]).unsqueeze(1) # [B, 1, D]
            enhanced = hidden_states.clone()
            enhanced[:, query_idx:query_idx+1, :] = enhanced[:, query_idx:query_idx+1, :] + delta
        else:
            raise ValueError(f"Unknown adapter mode: {self.adapter_mode}")
            
        telemetry = {
            "num_thoughts": self.num_thought_tokens,
            "ponder_steps": self.controller.max_ponder_steps if k_steps is None else k_steps,
            "adapter_mode": self.adapter_mode
        }
        return enhanced, telemetry
