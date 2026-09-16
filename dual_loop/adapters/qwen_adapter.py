import os
import hashlib
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
        confidence_threshold: Optional[float] = None,
        enable_critique: bool = True,
        use_learned_halting: bool = False,
        use_hypothesis_verification: bool = True,
        **adapter_kwargs
    ):
        super().__init__()
        self.qwen = qwen_model
        
        # Locate decoder layers
        layers = _find_transformer_layers(self.qwen)
        total_layers = len(layers)
        
        # Target layer defaults to midpoint L // 2 (or nearest full_attention layer for hybrid architectures)
        if layer_idx is None:
            default_idx = total_layers // 2
            cfg = getattr(self.qwen, "config", None)
            tc = getattr(cfg, "text_config", cfg)
            layer_types = getattr(tc, "layer_types", None) if tc else None
            if layer_types and isinstance(layer_types, list):
                full_attn_indices = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
                if full_attn_indices:
                    default_idx = min(full_attn_indices, key=lambda x: abs(x - (total_layers // 2)))
            self.layer_idx = default_idx
        else:
            self.layer_idx = layer_idx

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
        self.confidence_threshold = confidence_threshold
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
            enable_critique=enable_critique,
            use_learned_halting=use_learned_halting,
            use_hypothesis_verification=use_hypothesis_verification,
            **adapter_kwargs
        )
        
        # Register PyTorch forward hook on intermediate layer
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, args: Any, output: Any):
        """
        Intercepts intermediate hidden states, deliberations in latent space,
        and seamlessly injects deliberated representations into the residual stream.
        """
        self.last_telemetry = {} # ARCH-02: prevent cross-request diagnostic leakage
        if not self.enabled or self.k_steps == 0:
            if self.k_steps == 0:
                self.last_telemetry = {
                    "bypassed": True,
                    "k_steps": 0,
                    "effective_k": 0.0
                }
            else:
                self.last_telemetry = {
                    "bypassed": True,
                    "enabled": False,
                    "k_steps": self.k_steps,
                    "effective_k": 0.0
                }
            return output

        # Handle tensor or tuple output (common across transformers versions)
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output
        
        if not isinstance(hidden_states, torch.Tensor):
            return output

        # Automatically match adapter device and dtype to intercepted hidden states
        try:
            adapter_param = next(self.adapter.parameters())
            if hidden_states.dtype != adapter_param.dtype or hidden_states.device != adapter_param.device:
                self.adapter.to(device=hidden_states.device, dtype=hidden_states.dtype)
        except StopIteration:
            pass

        # ARCH-03 & ARCH-05: Resolve per-sample query anchor and key_padding_mask
        query_idx = self.query_idx
        key_padding_mask = None
        att_mask = getattr(self, "_current_attention_mask", None)
        if att_mask is not None and isinstance(att_mask, torch.Tensor) and att_mask.dim() == 2:
            # key_padding_mask for PyTorch MHA (True = ignore/pad)
            seq_len = hidden_states.size(1)
            mask_slice = att_mask[:, :seq_len]
            key_padding_mask = (mask_slice == 0)
            is_all_negative = False
            if isinstance(self.query_idx, (int, float)) and int(self.query_idx) == -1:
                is_all_negative = True
            elif isinstance(self.query_idx, torch.Tensor) and (self.query_idx == -1).all():
                is_all_negative = True
                
            if is_all_negative:
                # Find last token index where att_mask == 1 for each sequence in batch
                pos = torch.arange(seq_len, device=att_mask.device).unsqueeze(0).expand(att_mask.size(0), -1)
                valid_positions = torch.where(mask_slice == 1, pos, -1)
                query_idx = valid_positions.max(dim=-1).values
                query_idx = torch.clamp(query_idx, min=0)

        # Execute recurrent latent deliberation
        enhanced, telemetry = self.adapter(
            hidden_states=hidden_states,
            k_steps=self.k_steps,
            query_idx=query_idx,
            dynamic_halting=self.dynamic_halting,
            key_padding_mask=key_padding_mask
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

    def set_confidence_threshold(self, threshold: Optional[float] = None):
        """
        Configures adaptive confidence-gated deliberation (Dynamic Halting).
        If margin between top-1 and top-2 candidates exceeds threshold (e.g. 3.0 nats),
        System 1 is confident and bypasses deliberation (k=0) to prevent over-pondering.
        """
        self.confidence_threshold = threshold

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

    def save_adapter(self, save_path: str, format: str = "safetensors"):
        """
        Saves only the lightweight adapter weights.
        Supports format='safetensors' (recommended) or format='pt'.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        if save_path.endswith(".safetensors") or format == "safetensors":
            if not save_path.endswith(".safetensors") and not save_path.endswith(".pt"):
                save_path = save_path + ".safetensors"
            try:
                import safetensors.torch
                contiguous_dict = {k: v.contiguous() for k, v in self.adapter.state_dict().items()}
                safetensors.torch.save_file(contiguous_dict, save_path)
                return save_path
            except ImportError:
                pass
        torch.save(self.adapter.state_dict(), save_path)
        return save_path

    def load_adapter(
        self,
        load_path: str,
        strict: bool = False,
        expected_sha256: Optional[str] = None,
        revision: Optional[str] = None
    ):
        """
        Loads lightweight adapter weights from a local file, local directory,
        or directly from Hugging Face Hub (e.g. 'CH3NDev/dual-loop-qwen3.5-2b').
        
        Args:
            load_path: Path to weights file, local directory, or HF repo ID.
            strict: Whether to strictly enforce key matching in load_state_dict.
            expected_sha256: Optional SHA-256 hex string to verify file integrity (SEC-04).
            revision: Optional commit SHA/branch for Hugging Face Hub downloads (SEC-02).
        """
        file_to_load = None
        
        # 1. Local file
        if os.path.isfile(load_path):
            file_to_load = load_path
        # 2. Local directory
        elif os.path.isdir(load_path):
            candidates = [
                os.path.join(load_path, "adapter_model.safetensors"),
                os.path.join(load_path, "qwen35_2b_adapter.pt"),
                os.path.join(load_path, "adapter.pt"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    file_to_load = c
                    break
        # 3. Hugging Face Hub repository
        if file_to_load is None:
            try:
                from huggingface_hub import hf_hub_download
                for filename in ["adapter_model.safetensors", "qwen35_2b_adapter.pt", "adapter.pt"]:
                    try:
                        file_to_load = hf_hub_download(repo_id=load_path, filename=filename, revision=revision)
                        break
                    except Exception:
                        continue
            except ImportError:
                pass
                
        if file_to_load is None:
            raise FileNotFoundError(
                f"Could not load adapter from '{load_path}'. "
                "Ensure the file exists locally or is a valid Hugging Face repository ID."
            )

        # Hash integrity check (SEC-04)
        if expected_sha256 is not None:
            sha256_hasher = hashlib.sha256()
            with open(file_to_load, "rb") as f:
                while chunk := f.read(65536):
                    sha256_hasher.update(chunk)
            actual_sha256 = sha256_hasher.hexdigest().lower()
            expected_clean = expected_sha256.strip().lower()
            if actual_sha256 != expected_clean:
                raise ValueError(
                    f"SHA-256 integrity check failed for '{file_to_load}'. "
                    f"Expected: {expected_clean}, but got: {actual_sha256}."
                )

        # Load weights safely (enforce safetensors or weights_only=True)
        if file_to_load.endswith(".safetensors"):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(file_to_load)
        else:
            # Try safetensors first in case it was saved as safetensors with a .pt extension
            loaded = False
            try:
                import safetensors.torch
                state_dict = safetensors.torch.load_file(file_to_load)
                loaded = True
            except Exception:
                pass
            if not loaded:
                try:
                    state_dict = torch.load(file_to_load, map_location="cpu", weights_only=True)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to safely deserialize '{file_to_load}' with weights_only=True. "
                        "This file may contain unsafe pickled objects. In accordance with security "
                        f"standards, arbitrary code execution via weights_only=False is strictly prohibited. Error: {e}"
                    ) from e
            
        # Allow missing gate_alpha for backward compatibility with un-gated checkpoints
        if "gate_alpha" not in state_dict and hasattr(self.adapter, "gate_alpha"):
            state_dict["gate_alpha"] = self.adapter.gate_alpha.data.clone()
            
        self.adapter.load_state_dict(state_dict, strict=strict)
        try:
            sample_param = next(self.qwen.parameters())
            self.adapter.to(device=sample_param.device, dtype=sample_param.dtype)
        except StopIteration:
            pass
        return file_to_load

    def remove_hook(self):
        """Cleanly detaches the forward hook from the underlying model."""
        if hasattr(self, "_hook_handle") and self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def forward(self, *args, **kwargs):
        """Passes inputs directly through the base model, intercepted by the hook."""
        self.last_telemetry = {}
        # Extract attention_mask if present
        self._current_attention_mask = kwargs.get("attention_mask", None)
        if self._current_attention_mask is None and len(args) > 1 and isinstance(args[1], torch.Tensor):
            self._current_attention_mask = args[1]
        try:
            return self.qwen(*args, **kwargs)
        finally:
            self._current_attention_mask = None

    def generate(self, *args, **kwargs):
        """Autoregressive generation with Dual-Loop latent deliberation enabled."""
        self.last_telemetry = {}
        self._current_attention_mask = kwargs.get("attention_mask", None)
        try:
            return self.qwen.generate(*args, **kwargs)
        finally:
            self._current_attention_mask = None

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
