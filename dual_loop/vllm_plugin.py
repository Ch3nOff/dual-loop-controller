"""
vLLM Plugin for HADL Dual-Loop Cognitive Controller
====================================================
Registers the HADL adapter as a vLLM general plugin, automatically
attaching the Dual-Loop cognitive hooks when models are loaded.

This plugin is discovered by vLLM via the `vllm.general_plugins` entry point
defined in pyproject.toml. It is called in every vLLM process (including
workers) during initialization.

Environment Variables:
    HADL_VLLM_ENABLED   : "1" to enable HADL hooks (default: "0")
    HADL_VLLM_K_STEPS   : Number of deliberation steps (default: "2")
"""

import os
import logging

logger = logging.getLogger(__name__)


def register():
    """
    vLLM plugin registration function.

    Called automatically by vLLM during engine initialization. Installs a
    model-loading callback that attaches HADL hooks after the model is loaded.

    This uses lazy initialization: the actual hook attachment happens when
    HADL_VLLM_ENABLED=1 is set AND the model is loaded, to avoid CUDA
    initialization issues in subprocess workers.
    """
    enabled = os.environ.get("HADL_VLLM_ENABLED", "0").strip()
    if enabled != "1":
        logger.debug("HADL vLLM plugin: Disabled (HADL_VLLM_ENABLED != 1)")
        return

    k_steps = int(os.environ.get("HADL_VLLM_K_STEPS", "2"))
    logger.info(f"HADL vLLM plugin: Enabled with k_steps={k_steps}")
    logger.info(
        "HADL adapter will be attached after model loading via "
        "DynamicGraphIntrospector."
    )

    # We register a model-loaded callback via monkey-patching vLLM's
    # model loading pipeline. The hook is installed when the model's
    # forward hooks are set up, which happens after weight loading.
    #
    # This is the recommended approach for out-of-tree model modifications
    # that don't change the model architecture but add processing hooks.
    try:
        from vllm.worker.model_runner import GPUModelRunnerBase

        _original_load = GPUModelRunnerBase.load_model

        def _patched_load(self, *args, **kwargs):
            """Wraps model loading to attach HADL hooks post-load."""
            result = _original_load(self, *args, **kwargs)
            _attach_hadl_hooks_to_runner(self, k_steps)
            return result

        GPUModelRunnerBase.load_model = _patched_load
        logger.info("HADL vLLM plugin: Model loading hook installed successfully")

    except (ImportError, AttributeError) as e:
        logger.warning(
            f"HADL vLLM plugin: Could not install model loading hook: {e}. "
            f"HADL augmentation will not be applied."
        )


def _attach_hadl_hooks_to_runner(runner, k_steps: int):
    """
    Attaches HADL adapter hooks to the model inside a vLLM model runner.
    Called after the model weights have been loaded.
    """
    try:
        import torch
        from dual_loop.adapters.universal_adapter import (
            UniversalDualLoopAdapter,
            DynamicGraphIntrospector,
            CANONICAL_DIM,
        )

        model = runner.model
        if model is None:
            logger.warning("HADL: Model is None, skipping hook attachment")
            return

        # Discover model structure
        container_path, layer_list, total_layers = (
            DynamicGraphIntrospector.discover_layer_container(model)
        )
        target_idx = total_layers // 2
        d_native = DynamicGraphIntrospector.probe_native_dimension(model)

        # Create adapter (no sleep consolidation for serving)
        adapter = UniversalDualLoopAdapter(
            d_native=d_native,
            d_canonical=CANONICAL_DIM,
            max_ponder_steps=k_steps,
            enable_plasticity=True,
            enable_firewall=True,
            enable_sleep_consolidation=False,
            init_alpha=0.0,
        )

        # Move to model device and dtype
        first_param = next(model.parameters(), None)
        if first_param is not None:
            adapter.to(device=first_param.device, dtype=first_param.dtype)

        # Register hook at mid-layer
        target_layer = layer_list[target_idx]

        def _hadl_forward_hook(module, args, output):
            if isinstance(output, tuple):
                h = output[0]
                if not isinstance(h, torch.Tensor):
                    return output
                enhanced_h, _ = adapter(h)
                return (enhanced_h,) + output[1:]
            elif isinstance(output, torch.Tensor):
                enhanced_h, _ = adapter(output)
                return enhanced_h
            return output

        target_layer.register_forward_hook(_hadl_forward_hook)

        logger.info(
            f"HADL: Adapter attached at layer {target_idx}/{total_layers} "
            f"(d_native={d_native}, d_canonical={CANONICAL_DIM}, k={k_steps})"
        )

    except Exception as e:
        logger.warning(f"HADL: Failed to attach hooks: {e}. Continuing without HADL.")
