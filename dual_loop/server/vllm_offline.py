"""
vLLM Offline Batch Inference with HADL Augmentation
====================================================
Provides a simple API for high-throughput batch inference using vLLM's
offline mode with HADL cognitive hooks.

Useful for:
  - Batch evaluation and benchmarking
  - Dataset processing and annotation
  - Research experiments comparing base vs HADL-augmented models

Example:
    from dual_loop.server.vllm_offline import vllm_batch_generate

    results = vllm_batch_generate(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompts=["What is quantum entanglement?", "Explain P vs NP."],
        enable_hadl=True,
    )
    for r in results:
        print(r["text"])
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Union


def vllm_batch_generate(
    model: str,
    prompts: Union[str, List[str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: Optional[List[str]] = None,
    enable_hadl: bool = True,
    hadl_k_steps: int = 2,
    tensor_parallel_size: int = 1,
    quantization: Optional[str] = None,
    gpu_memory_utilization: float = 0.90,
    trust_remote_code: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    High-throughput batch generation using vLLM with optional HADL augmentation.

    Args:
        model: HuggingFace model ID or local path.
        prompts: Single prompt string or list of prompts.
        max_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        stop: List of stop strings.
        enable_hadl: Whether to attach HADL cognitive hooks.
        hadl_k_steps: Number of latent deliberation steps.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        quantization: Quantization method (e.g., "awq", "gptq").
        gpu_memory_utilization: Fraction of GPU memory to use.
        trust_remote_code: Trust remote code in model files.
        **kwargs: Additional arguments passed to VLLMInferenceEngine.

    Returns:
        List of result dicts with keys: text, prompt_tokens, completion_tokens,
        total_tokens, finish_reason, elapsed_seconds, tokens_per_second.
    """
    from .vllm_engine import VLLMInferenceEngine

    engine = VLLMInferenceEngine(
        model_id_or_path=model,
        tensor_parallel_size=tensor_parallel_size,
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_hadl=enable_hadl,
        hadl_k_steps=hadl_k_steps,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    engine.load_model()

    return engine.generate_sync(
        prompts=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )


def vllm_compare_base_vs_hadl(
    model: str,
    prompts: Union[str, List[str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    hadl_k_steps: int = 2,
    **kwargs,
) -> Dict[str, Any]:
    """
    A/B comparison helper: generates from the same model with and without
    HADL augmentation for evaluation.

    Returns:
        Dict with keys 'base_results', 'hadl_results', and 'comparison'
        containing side-by-side throughput and output analysis.
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    # Generate without HADL
    base_results = vllm_batch_generate(
        model=model,
        prompts=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_hadl=False,
        **kwargs,
    )

    # Generate with HADL
    hadl_results = vllm_batch_generate(
        model=model,
        prompts=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_hadl=True,
        hadl_k_steps=hadl_k_steps,
        **kwargs,
    )

    # Compute comparison stats
    base_avg_tps = sum(r["tokens_per_second"] for r in base_results) / len(base_results)
    hadl_avg_tps = sum(r["tokens_per_second"] for r in hadl_results) / len(hadl_results)

    return {
        "base_results": base_results,
        "hadl_results": hadl_results,
        "comparison": {
            "num_prompts": len(prompts),
            "base_avg_tokens_per_second": round(base_avg_tps, 2),
            "hadl_avg_tokens_per_second": round(hadl_avg_tps, 2),
            "throughput_ratio": round(hadl_avg_tps / base_avg_tps, 3) if base_avg_tps > 0 else None,
            "hadl_k_steps": hadl_k_steps,
        },
    }
