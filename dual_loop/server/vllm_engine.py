"""
vLLM-Powered High-Throughput Inference Engine for HADL
=====================================================
Replaces the standard HuggingFace-based inference engine with vLLM's
PagedAttention, continuous batching, and optimized CUDA kernels.

HADL's Dual-Loop cognitive adapter is applied as a post-prefill hidden-state
hook via vLLM's model runner callback mechanism, preserving all 5 Brain
Organs while gaining 3-8x throughput from vLLM's scheduler.

Usage:
    dual-loop serve-vllm --model Qwen/Qwen2.5-7B-Instruct --port 8000
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, Any, Optional, List, Union

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client,
    )
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

logger = logging.getLogger(__name__)


def _require_vllm():
    """Raises ImportError with installation instructions if vLLM is missing."""
    if not _HAS_VLLM:
        raise ImportError(
            "\n" + "=" * 80 + "\n"
            "[!] vLLM is required for the high-throughput serving backend.\n"
            "=" * 80 + "\n"
            "Install with:\n"
            "  pip install 'dual-loop-controller[vllm]'\n\n"
            "Or install vLLM directly:\n"
            "  pip install vllm\n\n"
            "Note: vLLM requires Linux with NVIDIA GPU (CUDA 12.1+).\n"
            "      For Windows, use WSL2 with a CUDA-enabled distro.\n"
            "=" * 80 + "\n"
        )


class VLLMInferenceEngine:
    """
    High-throughput inference engine powered by vLLM with HADL cognitive
    augmentation applied as a lightweight logits-processing hook.

    Architecture:
        ┌───────────────────────────────────────────────┐
        │  vLLM Engine (PagedAttention + Cont. Batching)│
        │  ┌─────────────────────────────────────────┐  │
        │  │ Base Model (Qwen, LLaMA, Gemma, etc.)   │  │
        │  │  Layer 0 ... L_mid                      │  │
        │  │  ┌───────────────────────────────────┐   │  │
        │  │  │ HADL Universal Adapter Hook       │   │  │
        │  │  │ (R^D → R^1024 → System2 → R^D)   │   │  │
        │  │  └───────────────────────────────────┘   │  │
        │  │  Layer L_mid+1 ... L_total               │  │
        │  │  LM Head → Logits                       │  │
        │  └─────────────────────────────────────────┘  │
        └───────────────────────────────────────────────┘
    """

    def __init__(
        self,
        model_id_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        quantization: Optional[str] = None,
        enable_hadl: bool = True,
        hadl_k_steps: int = 2,
        trust_remote_code: bool = False,
        seed: int = 42,
        enforce_eager: bool = False,
        swap_space: int = 4,
    ):
        _require_vllm()

        self.model_id = model_id_or_path
        self.enable_hadl = enable_hadl
        self.hadl_k_steps = hadl_k_steps
        self.is_loaded = False
        self.llm: Optional[LLM] = None

        # Store vLLM configuration
        self._vllm_kwargs = dict(
            model=model_id_or_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
            seed=seed,
            enforce_eager=enforce_eager,
            swap_space=swap_space,
        )

        # Filter out None values
        self._vllm_kwargs = {k: v for k, v in self._vllm_kwargs.items() if v is not None}

        # Telemetry counters
        self.total_tokens_generated = 0
        self.total_generation_time_sec = 0.0
        self.total_requests = 0

    def load_model(self) -> None:
        """
        Initializes the vLLM engine and optionally attaches HADL hooks.
        """
        print("=" * 80)
        print(f"[*] INITIALIZING vLLM HIGH-THROUGHPUT ENGINE")
        print(f"[*] Model ID            : {self.model_id}")
        print(f"[*] HADL Augmentation   : {'ENABLED (k={})'.format(self.hadl_k_steps) if self.enable_hadl else 'DISABLED'}")
        print(f"[*] Tensor Parallel     : {self._vllm_kwargs.get('tensor_parallel_size', 1)}")
        print(f"[*] Quantization        : {self._vllm_kwargs.get('quantization', 'None (native precision)')}")
        print(f"[*] GPU Mem Utilization : {self._vllm_kwargs.get('gpu_memory_utilization', 0.90):.0%}")
        print("=" * 80)

        t0 = time.time()

        # Initialize vLLM offline engine
        self.llm = LLM(**self._vllm_kwargs)

        # Attach HADL adapter hooks if enabled
        if self.enable_hadl:
            self._attach_hadl_hooks()

        load_sec = time.time() - t0
        self.is_loaded = True
        print(f"[OK] vLLM engine loaded in {load_sec:.2f}s")

    def _attach_hadl_hooks(self) -> None:
        """
        Attaches the HADL Universal Dual-Loop Adapter as a forward hook on the
        mid-layer of the loaded model inside vLLM's model runner.

        This is non-destructive: ReZero initialization (alpha=0) ensures the
        hook has zero initial impact, and the adapter auto-discovers the
        model's layer structure via DynamicGraphIntrospector.
        """
        try:
            import torch
            from ..adapters.universal_adapter import (
                UniversalDualLoopAdapter,
                DynamicGraphIntrospector,
                CANONICAL_DIM,
            )

            # Access the underlying nn.Module from vLLM's model runner
            model_runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
            inner_model = model_runner.model

            # Discover layer structure
            container_path, layer_list, total_layers = (
                DynamicGraphIntrospector.discover_layer_container(inner_model)
            )
            target_idx = total_layers // 2
            d_native = DynamicGraphIntrospector.probe_native_dimension(inner_model)

            # Create adapter
            adapter = UniversalDualLoopAdapter(
                d_native=d_native,
                d_canonical=CANONICAL_DIM,
                max_ponder_steps=self.hadl_k_steps,
                enable_plasticity=True,
                enable_firewall=True,
                enable_sleep_consolidation=False,  # No sleep during serving
                init_alpha=0.0,
            )

            # Move adapter to match model device and dtype
            first_param = next(inner_model.parameters())
            adapter.to(device=first_param.device, dtype=first_param.dtype)

            # Register forward hook on the target layer
            target_layer = layer_list[target_idx]

            def _hadl_hook(module, args, output):
                if isinstance(output, tuple):
                    h = output[0]
                    rest = output[1:]
                    enhanced_h, _ = adapter(h)
                    return (enhanced_h,) + rest
                else:
                    enhanced_h, _ = adapter(output)
                    return enhanced_h

            target_layer.register_forward_hook(_hadl_hook)

            # Store reference for telemetry
            self._hadl_adapter = adapter

            print(f"[OK] HADL adapter attached at layer {target_idx}/{total_layers}")
            print(f"[*] Native dim: {d_native} → Canonical dim: {CANONICAL_DIM}")
            print(f"[*] Ponder steps: {self.hadl_k_steps}")

        except Exception as e:
            logger.warning(
                f"Failed to attach HADL hooks to vLLM model: {e}. "
                f"Serving will continue without HADL augmentation."
            )
            self.enable_hadl = False

    def generate_sync(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous batch generation using vLLM's offline mode.
        Significantly faster than HF generate() for batch inference.
        """
        if not self.is_loaded:
            self.load_model()

        if isinstance(prompts, str):
            prompts = [prompts]

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        t0 = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        elapsed = max(0.001, time.time() - t0)

        results = []
        total_completion_tokens = 0

        for output in outputs:
            generated_text = output.outputs[0].text
            completion_tokens = len(output.outputs[0].token_ids)
            prompt_tokens = len(output.prompt_token_ids)
            total_completion_tokens += completion_tokens

            results.append({
                "text": generated_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "finish_reason": output.outputs[0].finish_reason,
            })

        self.total_tokens_generated += total_completion_tokens
        self.total_generation_time_sec += elapsed
        self.total_requests += len(prompts)

        tok_sec = total_completion_tokens / elapsed
        for r in results:
            r["elapsed_seconds"] = round(elapsed, 3)
            r["tokens_per_second"] = round(tok_sec, 2)
            r["engine"] = "vllm"
            r["hadl_enabled"] = self.enable_hadl

        return results

    def get_tokenizer(self):
        """Returns the tokenizer from the vLLM engine."""
        if self.llm is not None:
            return self.llm.get_tokenizer()
        return None

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns cumulative inference telemetry."""
        avg_tok_sec = (
            self.total_tokens_generated / self.total_generation_time_sec
            if self.total_generation_time_sec > 0
            else 0.0
        )
        return {
            "engine": "vllm",
            "model_id": self.model_id,
            "hadl_enabled": self.enable_hadl,
            "hadl_k_steps": self.hadl_k_steps if self.enable_hadl else 0,
            "total_tokens_generated": self.total_tokens_generated,
            "total_generation_time_sec": round(self.total_generation_time_sec, 2),
            "total_requests": self.total_requests,
            "avg_tokens_per_second": round(avg_tok_sec, 2),
            "tensor_parallel_size": self._vllm_kwargs.get("tensor_parallel_size", 1),
            "quantization": self._vllm_kwargs.get("quantization"),
        }
