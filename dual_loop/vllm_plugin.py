"""
vLLM Plugin for HADL Dual-Loop Cognitive Controller (Car-Lift v4.5 Edition)
============================================================================
Registers the HADL adapter as a vLLM general plugin, automatically
attaching the Dual-Loop cognitive hooks when models are loaded.

This plugin is discovered by vLLM via the `vllm.general_plugins` entry point
defined in pyproject.toml. It is called in every vLLM process (including
workers) during initialization.

Environment Variables:
    HADL_VLLM_ENABLED        : "1" to enable HADL hooks (default: "0")
    HADL_VLLM_VERSION        : "v4.5" (Car-Lift Dual-Cup) or "v3.0" (Universal) (default: "v4.5")
    HADL_VLLM_K_STEPS        : Number of deliberation steps (default: "2")
    HADL_VLLM_CHECKPOINT     : Path to distilled weights checkpoint (optional)
    HADL_VLLM_POROUS_ORIFICE : Permeability factor for porous firewall (default: "0.20")
    HADL_VLLM_DEADZONE       : Hard deadzone threshold for complete bypass (default: "0.12")
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def register():
    """
    vLLM plugin registration function.

    Called automatically by vLLM during engine initialization. Installs a
    model-loading callback that attaches HADL hooks after the model is loaded.
    """
    enabled = os.environ.get("HADL_VLLM_ENABLED", "0").strip()
    if enabled != "1":
        logger.debug("HADL vLLM plugin: Disabled (HADL_VLLM_ENABLED != 1)")
        return

    version = os.environ.get("HADL_VLLM_VERSION", "v4.5").strip()
    k_steps = int(os.environ.get("HADL_VLLM_K_STEPS", "2"))
    checkpoint = os.environ.get("HADL_VLLM_CHECKPOINT", "").strip() or None

    logger.info(f"HADL vLLM plugin: Enabled (Version={version}, k_steps={k_steps})")

    # Hook into vLLM's GPUModelRunnerBase.load_model
    try:
        from vllm.worker.model_runner import GPUModelRunnerBase

        _original_load = GPUModelRunnerBase.load_model

        def _patched_load(self, *args, **kwargs):
            """Wraps model loading to attach HADL cognitive hooks post-load."""
            result = _original_load(self, *args, **kwargs)
            _attach_hadl_hooks_to_runner(self, version=version, k_steps=k_steps, checkpoint=checkpoint)
            return result

        GPUModelRunnerBase.load_model = _patched_load
        logger.info("HADL vLLM plugin: Custom model loading hook installed successfully")

    except (ImportError, AttributeError) as e:
        logger.warning(
            f"HADL vLLM plugin: Could not install model loading hook: {e}. "
            f"HADL augmentation will not be applied."
        )


def _attach_hadl_hooks_to_runner(runner, version: str = "v4.5", k_steps: int = 2, checkpoint: Optional[str] = None):
    """
    Attaches HADL adapter hooks to the model inside a vLLM model runner.
    Called after the model weights have been loaded into GPU memory.
    """
    try:
        import torch
        from dual_loop.adapters.universal_adapter import DynamicGraphIntrospector, CANONICAL_DIM

        model = getattr(runner, "model", None)
        if model is None:
            logger.warning("HADL: Model is None, skipping hook attachment")
            return

        # Discover model architecture
        container_path, layer_list, total_layers = (
            DynamicGraphIntrospector.discover_layer_container(model)
        )
        target_idx = total_layers // 2
        ghost_idx = max(target_idx + 1, total_layers - 2)
        d_native = DynamicGraphIntrospector.probe_native_dimension(model)

        first_param = next(model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = first_param.dtype if first_param is not None else torch.bfloat16

        if version == "v4.5":
            # HADL v4.5 Car-Lift Edition (Two-Piston Hydraulics + Porous Firewall + SVD Ghost Layer)
            from dual_loop.dual_cup_poly_engine import HADLv45Controller, HADLv45ModelWrapper

            controller = HADLv45Controller(d_model=d_native)
            if checkpoint and Path(checkpoint).exists():
                ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if "controller_state_dict" in ckpt:
                    controller.load_state_dict(ckpt["controller_state_dict"], strict=False)
                elif "controller" in ckpt:
                    controller.load_state_dict(ckpt["controller"], strict=False)
                logger.info(f"HADL v4.5: Restored distilled weights from {checkpoint}")

            controller.to(device=device, dtype=dtype)
            controller.eval()

            # Attach forward hook on target layer 11 (midpoint)
            target_layer = layer_list[target_idx]
            ghost_layer = layer_list[ghost_idx]
            z_cache: Dict[str, Any] = {"stream": None}

            def _mid_hook(module, args, output):
                if isinstance(output, tuple):
                    h = output[0]
                    if isinstance(h, torch.Tensor):
                        # Dispatch through porous router
                        weights, _ = controller.router(embed_tensor=h)
                        h_enh, z_stream = controller.process_layer11(h, weights)
                        z_cache["stream"] = z_stream
                        return (h_enh,) + output[1:]
                elif isinstance(output, torch.Tensor):
                    weights, _ = controller.router(embed_tensor=output)
                    h_enh, z_stream = controller.process_layer11(output, weights)
                    z_cache["stream"] = z_stream
                    return h_enh
                return output

            def _ghost_hook(module, args, output):
                if z_cache["stream"] is not None:
                    if isinstance(output, tuple):
                        h = output[0]
                        if isinstance(h, torch.Tensor):
                            weights, _ = controller.router(embed_tensor=h)
                            h_ver = controller.process_ghost(h, z_cache["stream"], weights)
                            z_cache["stream"] = None
                            return (h_ver,) + output[1:]
                    elif isinstance(output, torch.Tensor):
                        weights, _ = controller.router(embed_tensor=output)
                        h_ver = controller.process_ghost(output, z_cache["stream"], weights)
                        z_cache["stream"] = None
                        return h_ver
                return output

            target_layer.register_forward_hook(_mid_hook)
            ghost_layer.register_forward_hook(_ghost_hook)

            logger.info(
                f"HADL v4.5 Car-Lift: Attached Two-Piston Hydraulic Controller at Layer {target_idx} "
                f"and Ghost Verification at Layer {ghost_idx} (D={d_native})"
            )

        else:
            # HADL v3.0 Universal Adapter (Canonical D_c=1024)
            from dual_loop.adapters.universal_adapter import UniversalDualLoopAdapter

            adapter = UniversalDualLoopAdapter(
                d_native=d_native,
                d_canonical=CANONICAL_DIM,
                max_ponder_steps=k_steps,
                enable_plasticity=True,
                enable_firewall=True,
                enable_sleep_consolidation=False,
                init_alpha=0.0,
            )
            adapter.to(device=device, dtype=dtype)
            adapter.eval()

            target_layer = layer_list[target_idx]

            def _universal_hook(module, args, output):
                if isinstance(output, tuple):
                    h = output[0]
                    if isinstance(h, torch.Tensor):
                        enh_h, _ = adapter(h)
                        return (enh_h,) + output[1:]
                elif isinstance(output, torch.Tensor):
                    enh_h, _ = adapter(output)
                    return enh_h
                return output

            target_layer.register_forward_hook(_universal_hook)
            logger.info(
                f"HADL v3.0 Universal: Attached at Layer {target_idx}/{total_layers} "
                f"(Native {d_native} → Canonical {CANONICAL_DIM}, k={k_steps})"
            )

    except Exception as e:
        logger.warning(f"HADL: Failed to attach hooks to vLLM: {e}. Serving without HADL.")


class HADLDeliberationLogitsProcessor:
    """
    Custom LogitsProcessor for vLLM:
    Regulates output token distribution by damping repetitive circular preamble tags
    (e.g., recursive `<think>` reasoning chatter) and penalizing repeated tokens.
    """
    def __init__(self, tokenizer=None, damping_factor: float = 0.50, repetition_penalty: float = 1.0):
        self.tokenizer = tokenizer
        self.damping_factor = damping_factor
        self.repetition_penalty = repetition_penalty
        self.banned_tokens = set()
        if tokenizer is not None:
            for tag in ["<think>", "</think>"]:
                ids = tokenizer.encode(tag, add_special_tokens=False) if hasattr(tokenizer, "encode") else []
                self.banned_tokens.update(ids)

    def __call__(self, input_ids: List[int], scores: "torch.Tensor") -> "torch.Tensor":
        # Apply repetition penalty if configured
        if self.repetition_penalty > 1.0 and input_ids:
            for tid in set(input_ids):
                if tid < scores.shape[-1]:
                    if scores[tid] > 0:
                        scores[tid] = scores[tid] / self.repetition_penalty
                    else:
                        scores[tid] = scores[tid] * self.repetition_penalty

        if not self.banned_tokens:
            return scores
        # If output length is sufficiently deep, damp preamble tokens
        if len(input_ids) > 64:
            for tid in self.banned_tokens:
                if tid < scores.shape[-1]:
                    scores[tid] = scores[tid] - (self.damping_factor * 10.0)
        return scores


def create_hadl_logits_processor(tokenizer=None, damping_factor: float = 0.50, repetition_penalty: float = 1.0) -> HADLDeliberationLogitsProcessor:
    """Factory helper to instantiate a HADL deliberation logits processor for vLLM."""
    return HADLDeliberationLogitsProcessor(tokenizer=tokenizer, damping_factor=damping_factor, repetition_penalty=repetition_penalty)
