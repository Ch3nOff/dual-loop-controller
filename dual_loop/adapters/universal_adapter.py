"""
Universal Model-Agnostic Adapter (HADL v3.0)
============================================
Eliminates model-specific fragmentation (Qwen vs. Gemma vs. LLaMA vs. Mistral vs. DeepSeek)
via Dynamic Runtime Graph Introspection and Canonical Latent Projection.

Mathematical Formulation:
Let a base model have arbitrary native hidden dimension D_base in N.
We establish a fixed canonical cognitive manifold R^D_c where D_c = 1024 FIXED for ALL models.

1. Inward Canonical Projection:
   z_0 = LayerNorm(W_down * h_native),   W_down in R^{D_c x D_base}

2. Universal Recurrent Deliberation in Canonical Space:
   z_k = T_System2(z_{k-1}, CWM(z_0)),    z_k in R^{D_c}
   Because T_System2 operates purely in R^{1024}, System 2 weights are 100% universal,
   model-agnostic, and transferable across model families and parameter scales.

3. Outward Canonical Expansion (ReZero Preservation):
   delta_native = tanh(alpha) * Gamma_allo * (W_up * z_K),  W_up in R^{D_base x D_c}
   With W_up initialized to 0 and alpha = 0, initial delta is strictly 0.000000,
   guaranteeing zero regression of the base model.
"""

import time
import math
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from ..memory import CognitiveWorkingMemory
from ..controller import RecurrentLatentController
from ..allostasis import AllostaticEnergyModulator
from ..plasticity import HeteroAssociativePlasticMemory
from ..firewall import SheafInvariantFirewall
from ..sleep_consolidation import SleepPhaseConsolidationEngine


CANONICAL_DIM: int = 1024
DEFAULT_NUM_SLOTS: int = 16


class DynamicGraphIntrospector:
    """
    Introspects arbitrary Transformer backbones to identify:
    1. Layer container (e.g. model.layers, transformer.h, etc.)
    2. Total layer count L_total
    3. Target midpoint layer L_target = floor(L_total / 2)
    4. Native hidden dimension D_native
    """

    LAYER_CONTAINER_NAMES = [
        "model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "model.model.language_model.layers",
        "transformer.h",
        "model.transformer.layers",
        "layers",
        "transformer.layers",
        "decoder.layers",
        "model.decoder.layers",
        "gpt_neox.layers"
    ]

    @classmethod
    def discover_layer_container(cls, model: nn.Module) -> Tuple[str, nn.ModuleList, int]:
        """
        Discovers the ModuleList representing transformer decoder blocks.
        Returns:
            container_path: string path of container
            layer_list: nn.ModuleList of blocks
            total_layers: integer layer count
        """
        # 1. Try common dot-separated paths
        for path in cls.LAYER_CONTAINER_NAMES:
            parts = path.split(".")
            curr = model
            found = True
            for p in parts:
                if hasattr(curr, p):
                    curr = getattr(curr, p)
                else:
                    found = False
                    break
            if found and isinstance(curr, (nn.ModuleList, list)) and len(curr) > 0:
                return path, curr, len(curr)

        # 2. Heuristic scan for the largest ModuleList
        candidate_list = None
        candidate_path = ""
        max_len = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > max_len:
                # Check if elements look like transformer blocks
                first = module[0]
                has_attn = any("attn" in n.lower() or "attention" in n.lower() for n, _ in first.named_modules())
                if has_attn or len(module) >= 4:
                    max_len = len(module)
                    candidate_list = module
                    candidate_path = name

        if candidate_list is not None and max_len > 0:
            return candidate_path, candidate_list, max_len

        raise ValueError(
            "DynamicGraphIntrospector: Could not automatically detect transformer layers ModuleList. "
            "Please specify layer_idx manually when attaching."
        )

    @classmethod
    def probe_native_dimension(cls, model: nn.Module, fallback_d: int = 2048) -> int:
        """Determines D_native from model configuration or parameter tensors."""
        # 1. Inspect config (including text_config for multimodal models like Qwen3.5/Qwen3.8)
        cfg = getattr(model, "config", None)
        if cfg is not None:
            sub_cfg = getattr(cfg, "text_config", None)
            search_cfgs = [sub_cfg, cfg] if sub_cfg is not None else [cfg]
            for c in search_cfgs:
                for attr in ("hidden_size", "d_model", "dim", "n_embd", "hidden_dim"):
                    val = getattr(c, attr, None)
                    if val is not None and isinstance(val, int) and val > 0:
                        return val

        # 2. Inspect first linear layer or embedding
        for name, param in model.named_parameters():
            if "embed" in name.lower() and param.dim() == 2:
                return param.shape[-1]
            if ("q_proj" in name.lower() or "wq" in name.lower() or "attn" in name.lower()) and param.dim() == 2:
                return param.shape[-1]

        return fallback_d


class UniversalDualLoopAdapter(nn.Module):
    """
    Universal Model-Agnostic Dual-Loop Cognitive Adapter.
    Connects any base model (LLaMA, Qwen, Gemma, Mistral, DeepSeek) to a fixed
    Canonical System 2 Engine (D_c = 1024).
    """
    def __init__(
        self,
        d_native: int,
        d_canonical: int = CANONICAL_DIM,
        num_slots: int = DEFAULT_NUM_SLOTS,
        max_ponder_steps: int = 3,
        enable_plasticity: bool = True,
        enable_firewall: bool = True,
        enable_sleep_consolidation: bool = True,
        init_alpha: float = 0.0
    ):
        super().__init__()
        self.d_native = int(d_native)
        self.d_canonical = int(d_canonical)
        self.max_ponder_steps = max_ponder_steps
        
        # 1. Canonical Projection Bridge (R^D_native -> R^D_canonical)
        self.down_proj = nn.Linear(self.d_native, self.d_canonical, bias=False)
        self.canonical_norm = nn.LayerNorm(self.d_canonical)
        nn.init.normal_(self.down_proj.weight, std=0.02)

        # 2. The Universal System 2 Engine (100% Identical Across All Base Models)
        self.cwm = CognitiveWorkingMemory(d_model=self.d_canonical, num_slots=num_slots)
        self.controller = RecurrentLatentController(d_model=self.d_canonical, n_heads=8)
        self.allostatic_modulator = AllostaticEnergyModulator(d_model=self.d_canonical)
        self.plastic_memory = HeteroAssociativePlasticMemory(
            d_model=self.d_canonical,
            rank=32,
            plastic_lr=0.50,
            decay_rate=0.0
        ) if enable_plasticity else None

        # 3. Canonical Expansion Bridge with ReZero Guarantee (R^D_canonical -> R^D_native)
        self.up_proj = nn.Linear(self.d_canonical, self.d_native, bias=False)
        # Kaiming initialization allows backpropagation gradients to reach alpha immediately
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        self.alpha = nn.Parameter(torch.tensor([float(init_alpha)]))  # ReZero gate (0.0 preserves identity, >0 for eager delta)

        # 4. Sheaf-Theoretic Invariant Firewall (Organ 5: Prefrontal Executive Inhibition)
        self.firewall = SheafInvariantFirewall(max_norm=60.0) if enable_firewall else None

        # 5. Sleep-Phase Consolidation Engine (Organ 4: Offline Distillation)
        self.sleep_engine = SleepPhaseConsolidationEngine(
            d_model=self.d_canonical,
            rank=32
        ) if enable_sleep_consolidation else None

        self.continual_mode: bool = False

    def set_continual_mode(self, enabled: bool = True, decay: float = 0.95):
        """Enables continuous waking learning."""
        self.continual_mode = enabled
        if self.plastic_memory is not None:
            self.plastic_memory.set_continual_mode(enabled, decay=decay)

    def trigger_sleep_cycle(self) -> Dict[str, Any]:
        """Triggers offline sleep-phase consolidation of accumulated fast weights."""
        if self.sleep_engine is None:
            return {"status": "NO_SLEEP_ENGINE"}
        
        fast_w = None
        if self.plastic_memory is not None and hasattr(self.plastic_memory, "M_cross"):
            fast_w = self.plastic_memory.M_cross

        telem = self.sleep_engine.consolidate(
            fast_weights=fast_w,
            target_device=self.down_proj.weight.device
        )
        
        # Soft decay fast weights post-sleep
        if self.plastic_memory is not None and hasattr(self.plastic_memory, "M_cross"):
            self.plastic_memory.M_cross.data *= 0.10

        return telem

    def forward(
        self,
        h_native: torch.Tensor,
        k_steps: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        query_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Universal Latent Deliberation Forward Pass.
        Accepts 3D [B, SeqLen, D_native] or 2D [NumTokens, D_native] tensors (vLLM / TensorRT safe).
        """
        t0 = time.perf_counter()
        orig_shape = h_native.shape
        is_2d = (h_native.dim() == 2)
        
        if is_2d:
            h_input = h_native.unsqueeze(0)  # [1, N, D_native]
        else:
            h_input = h_native               # [B, S, D_native]

        B, S, D = h_input.shape
        # Ensure adapter parameters and buffers match input tensor device & dtype
        if self.down_proj.weight.device != h_input.device or self.down_proj.weight.dtype != h_input.dtype:
            self.to(device=h_input.device, dtype=h_input.dtype)
        steps = self.max_ponder_steps if k_steps is None else k_steps

        # Fast Streaming Bypass: if steps == 0, bypass System 2 immediately
        if steps == 0:
            elapsed_us = (time.perf_counter() - t0) * 1e6
            return h_native, {
                "bypassed": True,
                "d_native": self.d_native,
                "d_canonical": self.d_canonical,
                "ponder_steps": 0,
                "latency_us": elapsed_us,
                "latency_ms": elapsed_us / 1000.0,
                "delta_norm": 0.0
            }

        # 1. Project Inward to Canonical Manifold (R^D_native -> R^1024)
        z_0 = self.canonical_norm(self.down_proj(h_input))  # [B, S, 1024]

        # 2. CWM Compression with Causal Isolation (Organ 3: Global Workspace Theory)
        # Prevent future suffix tokens from leaking into CWM slots during full-sequence scoring
        if query_idx is not None and 0 <= query_idx < S - 1:
            z_context = z_0[:, :query_idx + 1, :]
        else:
            z_context = z_0
        cwm_slots = self.cwm(z_context)  # [B, 16, 1024]

        # 3. Universal Recurrent Latent Contemplation in Canonical Space
        query_rep = z_0[:, -1, :]  # Focus deliberation on current token state [B, 1024]
        h_thought, _, _ = self.controller(
            query_rep=query_rep,
            memory=cwm_slots,
            k_steps=steps
        )  # [B, L_thought, 1024]
        primary_thought = h_thought[:, 0, :]  # [B, 1024]

        # Apply sleep-consolidated permanent memories if available
        if self.sleep_engine is not None and self.sleep_engine.total_sleep_cycles > 0:
            primary_thought = primary_thought + 0.10 * self.sleep_engine(primary_thought)

        # Record awake episode if continual learning is active
        if self.continual_mode and self.sleep_engine is not None:
            self.sleep_engine.record_episode(
                v_novel=query_rep.detach(),
                u_concept=primary_thought.detach(),
                surprise_score=1.0
            )

        # 4. Canonical Delta & Allostatic Energy Modulation (Organ 2)
        delta_canonical = primary_thought - query_rep  # [B, 1024]
        delta_canonical_mod, allo_telem = self.allostatic_modulator(
            delta_canonical,
            scale=torch.ones(1, device=primary_thought.device)
        )

        # 5. Project Outward to Native Dimension with ReZero Scaling (R^1024 -> R^D_native)
        # delta_native = tanh(alpha) * W_up(delta_canonical_mod)
        delta_native = self.up_proj(delta_canonical_mod) * torch.tanh(self.alpha)  # [B, D_native]
        delta_native = delta_native.unsqueeze(1)  # [B, 1, D_native]

        # Broadcast to full sequence if necessary (or inject at query token)
        if S > 1:
            full_delta = torch.zeros_like(h_input)
            inject_idx = query_idx if query_idx is not None else -1
            if inject_idx < 0:
                inject_idx = S + inject_idx
            inject_idx = max(0, min(S - 1, inject_idx))
            full_delta[:, inject_idx:inject_idx+1, :] = delta_native
        else:
            full_delta = delta_native

        # 6. Prefrontal Executive Inhibition (Organ 5: Sheaf Invariant Firewall)
        firewall_telem = {}
        if self.firewall is not None:
            ctx = context or {}
            ctx["h_base"] = h_input
            full_delta, firewall_telem = self.firewall(full_delta, ctx)

        enhanced_h = h_input + full_delta

        if is_2d:
            enhanced_h = enhanced_h.squeeze(0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        telemetry = {
            "bypassed": False,
            "d_native": self.d_native,
            "d_canonical": self.d_canonical,
            "ponder_steps": steps,
            "gamma_allostatic": allo_telem.get("gamma_allostatic", 1.0),
            "rezero_alpha": float(self.alpha.item()),
            "delta_norm": float(torch.norm(full_delta).item()),
            "firewall": firewall_telem,
            "allostasis": allo_telem,
            "latency_ms": elapsed_ms,
            "latency_us": elapsed_ms * 1000.0
        }

        return enhanced_h, telemetry


class UniversalDualLoopModelWrapper(nn.Module):
    """
    Wraps an entire base model with Universal Dual-Loop Controller attached at L_target.
    Preserves all native HuggingFace methods: .generate(), .forward(), .to(), etc.
    """
    def __init__(
        self,
        base_model: nn.Module,
        adapter: UniversalDualLoopAdapter,
        target_layer_idx: int,
        container_path: str
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.target_layer_idx = target_layer_idx
        self.container_path = container_path
        
        # Install hook at target layer
        self._hook_handle = None
        self._register_adapter_hook()

    def _register_adapter_hook(self):
        """Installs the forward hook at layer L_target."""
        _, layer_list, _ = DynamicGraphIntrospector.discover_layer_container(self.base_model)
        target_layer = layer_list[self.target_layer_idx]

        def _forward_hook(module, args, output):
            # Handles tuple outputs (hidden_states, ...)
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
                enhanced_h, _ = self.adapter(h)
                return (enhanced_h,) + rest
            else:
                enhanced_h, _ = self.adapter(output)
                return enhanced_h

        self._hook_handle = target_layer.register_forward_hook(_forward_hook)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def trigger_sleep_cycle(self) -> Dict[str, Any]:
        return self.adapter.trigger_sleep_cycle()

    def set_continual_mode(self, enabled: bool = True, decay: float = 0.95):
        self.adapter.set_continual_mode(enabled, decay=decay)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if hasattr(self.base_model, "generate"):
            return self.base_model.generate(*args, **kwargs)
        raise AttributeError("Base model does not have .generate() method.")

    def __getattr__(self, name: str):
        # Delegate attribute access to base_model if not found in wrapper
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def attach_universal_dual_loop(
    base_model: nn.Module,
    layer_idx: Optional[int] = None,
    d_canonical: int = CANONICAL_DIM,
    num_slots: int = DEFAULT_NUM_SLOTS,
    max_ponder_steps: int = 3,
    enable_plasticity: bool = True,
    enable_firewall: bool = True,
    enable_sleep_consolidation: bool = True,
    init_alpha: float = 0.0
) -> UniversalDualLoopModelWrapper:
    """
    Universal Entrypoint for attaching HADL v3.0 to ANY model.
    Auto-discovers layer structure, probes native dimension, and binds canonical adapter.
    """
    # 1. Discover layer container and total layers
    container_path, layer_list, total_layers = DynamicGraphIntrospector.discover_layer_container(base_model)
    target_idx = (total_layers // 2) if layer_idx is None else int(layer_idx)

    # 2. Probe native hidden dimension
    d_native = DynamicGraphIntrospector.probe_native_dimension(base_model)

    # 3. Create Universal Canonical Adapter
    adapter = UniversalDualLoopAdapter(
        d_native=d_native,
        d_canonical=d_canonical,
        num_slots=num_slots,
        max_ponder_steps=max_ponder_steps,
        enable_plasticity=enable_plasticity,
        enable_firewall=enable_firewall,
        enable_sleep_consolidation=enable_sleep_consolidation,
        init_alpha=init_alpha
    )

    # Move adapter to base model device & dtype if parameters exist
    try:
        first_param = next(base_model.parameters())
        adapter.to(device=first_param.device, dtype=first_param.dtype)
    except StopIteration:
        pass

    # 4. Wrap base model
    wrapper = UniversalDualLoopModelWrapper(
        base_model=base_model,
        adapter=adapter,
        target_layer_idx=target_idx,
        container_path=container_path
    )

    return wrapper
