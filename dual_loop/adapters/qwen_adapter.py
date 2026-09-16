import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from .latent_adapter import LatentDeliberationAdapter


def _find_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """
    Introspects model architecture to extract decoder layer list.
    Supports Qwen (Qwen2, Qwen2.5, Qwen3.5), LLaMA, Mistral, and generic causal LMs.
    """
    # 1. Qwen3.5 (e.g. Qwen3_5ForConditionalGeneration.model.language_model.layers)
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    # 2. Standard HuggingFace AutoModelForCausalLM (e.g. Qwen2ForCausalLM.model.layers, Llama.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # 3. Base model direct (e.g. Qwen2Model.layers)
    if hasattr(model, "layers"):
        return model.layers
    # 4. GPT-style transformer architectures
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Could not automatically locate transformer layers in the provided model. "
        "Expected `model.model.language_model.layers`, `model.model.layers`, `model.layers`, or `model.transformer.h`."
    )


class DualLoopQwenModel(nn.Module):
    """
    Dual-Loop Cognitive Controller Adapter for Qwen (Qwen-2 / Qwen-2.5 / Qwen-3.5).
    
    Architecture:
    Attaches a weight-tied recurrent `LatentDeliberationAdapter` at an intermediate
    layer (default L // 2) via a non-invasive PyTorch forward hook.
    
    Properties:
    - Pure System 1 when k_steps=0 (exact zero-overhead identity bypass).
    - Latent deliberation when k_steps >= 1 without emitting intermediate discrete tokens.
    - Preserves causal mask, RoPE relative embeddings, and KV-cache compatibility.
    - Supports dynamic early halting per sample/batch based on latent convergence.
    - Enables parameter-efficient fine-tuning (PEFT): freezes base weights, training
      only ~1-2% adapter parameters.
    """
    def __init__(
        self,
        qwen_model: nn.Module,
        layer_idx: Optional[int] = None,
        k_steps: int = 2,
        dynamic_halting: bool = False,
        num_thought_tokens: int = 4,
        max_ponder_steps: int = 4,
        num_cwm_slots: int = 16,
        adapter_mode: str = "residual",
        capacity_factor: float = 0.5,
        query_idx: int = -1,
        vocab_size: Optional[int] = None,
        **adapter_kwargs
    ):
        super().__init__()
        self.qwen = qwen_model
        
        # Locate decoder layers
        layers = _find_transformer_layers(self.qwen)
        total_layers = len(layers)
        
        # Target layer defaults to midpoint L // 2
        self.layer_idx = layer_idx if layer_idx is not None else total_layers // 2
        if not (0 <= self.layer_idx < total_layers):
            raise IndexError(f"layer_idx {self.layer_idx} out of range [0, {total_layers-1}]")
            
        target_layer = layers[self.layer_idx]
        
        # Determine hidden size
        cfg = getattr(self.qwen, "config", None)
        hidden_size = None
        if cfg is not None:
            if hasattr(cfg, "text_config"):
                hidden_size = getattr(cfg.text_config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        if hidden_size is None:
            # Fallback: check target layer dimension
            for module in target_layer.modules():
                if isinstance(module, nn.Linear):
                    hidden_size = module.in_features
                    break
        if hidden_size is None:
            raise ValueError("Could not automatically determine model hidden_size from config or layers.")
            
        self.hidden_size = hidden_size
        self.k_steps = k_steps
        self.dynamic_halting = dynamic_halting
        self.query_idx = query_idx
        self.enabled = True
        self.last_telemetry: Dict[str, Any] = {}
        
        # Instantiate plug-and-play adapter
        self.adapter = LatentDeliberationAdapter(
            d_model=self.hidden_size,
            n_heads=getattr(cfg, "num_attention_heads", 8) if cfg else 8,
            num_thought_tokens=num_thought_tokens,
            max_ponder_steps=max_ponder_steps,
            capacity_factor=capacity_factor,
            num_cwm_slots=num_cwm_slots,
            adapter_mode=adapter_mode,
            vocab_size=vocab_size,
            **adapter_kwargs
        )
        
        # Register PyTorch forward hook on intermediate layer
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, args: Any, output: Any):
        """
        Intercepts intermediate hidden states, deliberations in latent space,
        and seamlessly injects deliberated representations into the residual stream.
        """
        if not self.enabled or self.k_steps == 0:
            if self.k_steps == 0:
                self.last_telemetry = {
                    "bypassed": True,
                    "k_steps": 0,
                    "effective_k": 0.0
                }
            return output

        # Handle tensor or tuple output (common across transformers versions)
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output
        
        if not isinstance(hidden_states, torch.Tensor):
            return output

        # Execute recurrent latent deliberation
        enhanced, telemetry = self.adapter(
            hidden_states=hidden_states,
            k_steps=self.k_steps,
            query_idx=self.query_idx,
            dynamic_halting=self.dynamic_halting
        )
        self.last_telemetry = telemetry

        if is_tuple:
            return (enhanced,) + output[1:]
        return enhanced

    def set_ponder_steps(self, k: int):
        """
        Dynamically configures pondering steps.
        k=0: Exact base model bypass.
        k>=1: System 2 latent deliberation.
        """
        self.k_steps = max(0, int(k))

    def enable_adapter(self, enabled: bool = True):
        """Enables or disables adapter interception."""
        self.enabled = enabled

    def disable_adapter(self):
        """Disables adapter (reverts directly to pure base model)."""
        self.enabled = False

    def freeze_backbone(self):
        """
        Freezes base model parameters for parameter-efficient fine-tuning (PEFT).
        Only the Dual-Loop adapter parameters retain gradients.
        """
        for param in self.qwen.parameters():
            param.requires_grad = False
        for param in self.adapter.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreezes all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Returns parameter counts and trainable ratio."""
        total_p = sum(p.numel() for p in self.parameters())
        trainable_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adapter_p = sum(p.numel() for p in self.adapter.parameters())
        backbone_p = total_p - adapter_p
        
        ratio = (trainable_p / total_p * 100.0) if total_p > 0 else 0.0
        return {
            "total_parameters": total_p,
            "trainable_parameters": trainable_p,
            "adapter_parameters": adapter_p,
            "backbone_parameters": backbone_p,
            "trainable_ratio_pct": round(ratio, 3),
            "target_layer_idx": self.layer_idx,
            "hidden_size": self.hidden_size,
            "k_steps": self.k_steps,
            "adapter_mode": self.adapter.adapter_mode
        }

    def save_adapter(self, save_path: str):
        """Saves only the lightweight adapter weights."""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(self.adapter.state_dict(), save_path)

    def load_adapter(self, load_path: str, strict: bool = True):
        """Loads lightweight adapter weights."""
        state_dict = torch.load(load_path, map_location="cpu", weights_only=True)
        self.adapter.load_state_dict(state_dict, strict=strict)

    def remove_hook(self):
        """Cleanly detaches the forward hook from the underlying model."""
        if hasattr(self, "_hook_handle") and self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def forward(self, *args, **kwargs):
        """Passes inputs directly through the base model, intercepted by the hook."""
        return self.qwen(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Autoregressive generation with Dual-Loop latent deliberation enabled."""
        return self.qwen.generate(*args, **kwargs)

    def __getattr__(self, name: str):
        """Transparently delegates undefined attributes/methods to the underlying model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.qwen, name)


def attach_dual_loop_to_qwen(
    model: nn.Module,
    layer_idx: Optional[int] = None,
    k_steps: int = 2,
    **kwargs
) -> DualLoopQwenModel:
    """
    Convenience factory to attach the Dual-Loop Cognitive Controller to any Qwen model.
    """
    if isinstance(model, DualLoopQwenModel):
        model.set_ponder_steps(k_steps)
        if layer_idx is not None and model.layer_idx != layer_idx:
            raise ValueError(
                f"Model is already wrapped with DualLoopQwenModel at layer {model.layer_idx}. "
                "Call remove_hook() before attaching to a different layer."
            )
        return model
    return DualLoopQwenModel(model, layer_idx=layer_idx, k_steps=k_steps, **kwargs)
