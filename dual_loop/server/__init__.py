"""
Dual-Loop Serving Package (vLLM Powered)
========================================
Exclusively powered by vLLM for high-throughput, production-grade serving.
Supports continuous batching, PagedAttention, and custom HADL cognitive
augmentation (Car-Lift v4.5 Two-Piston Hydraulics + SVD Ghost Layer).
"""

from __future__ import annotations

from .vllm_app import start_vllm_server
from .vllm_engine import VLLMInferenceEngine
from .vllm_offline import vllm_batch_generate, vllm_compare_base_vs_hadl


def start_server(*args, **kwargs):
    """
    Main entrypoint for starting the Dual-Loop OpenAI-compatible server.
    Exclusively delegates to start_vllm_server for high-throughput serving.
    """
    return start_vllm_server(*args, **kwargs)


__all__ = [
    "start_server",
    "start_vllm_server",
    "VLLMInferenceEngine",
    "vllm_batch_generate",
    "vllm_compare_base_vs_hadl",
]
