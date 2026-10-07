"""
vLLM-Powered OpenAI-Compatible REST API Server for HADL
=======================================================
Drop-in replacement for app.py that uses vLLM's built-in OpenAI-compatible
server with HADL hooks pre-attached.

This launches vLLM's native OpenAI server (which handles continuous batching,
streaming, and the full /v1/chat/completions spec) but with the HADL
Dual-Loop adapter hooked into the model layers.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def _check_vllm_available():
    """Verify vLLM is installed and importable."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def start_vllm_server(
    model_id_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
    host: str = "127.0.0.1",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.90,
    quantization: Optional[str] = None,
    enable_hadl: bool = True,
    hadl_k_steps: int = 2,
    trust_remote_code: bool = False,
    api_key: Optional[str] = None,
    enforce_eager: bool = False,
    max_num_seqs: int = 256,
    extra_args: Optional[List[str]] = None,
) -> None:
    """
    Launches vLLM's native OpenAI-compatible API server with HADL hooks.

    This is the recommended production serving path for maximum throughput.
    vLLM handles:
    - Continuous batching with PagedAttention
    - OpenAI API compatibility (/v1/chat/completions, /v1/completions)
    - Streaming via Server-Sent Events
    - Token-level scheduling and preemption

    HADL hooks are injected via the vLLM plugin system (dual_loop_vllm_plugin).
    """
    # SEC-02: Mandate authentication if binding to a non-loopback / public interface
    is_loopback = host in ("127.0.0.1", "localhost", "::1")
    resolved_api_key = api_key or os.environ.get("VLLM_API_KEY", "").strip() or None
    if not is_loopback and not resolved_api_key:
        raise ValueError(
            f"Binding to non-loopback host '{host}' requires an API key for authentication. "
            "Please provide --api-key or set the VLLM_API_KEY environment variable."
        )

    if not _check_vllm_available():
        print(
            "\n" + "=" * 80 + "\n"
            "[ERROR] vLLM is not installed.\n"
            "=" * 80 + "\n"
            "To use the high-throughput vLLM backend, install it:\n"
            "  pip install 'dual-loop-controller[vllm]'\n\n"
            "Or install vLLM directly:\n"
            "  pip install vllm\n\n"
            "Requirements:\n"
            "  - Linux (or WSL2 on Windows)\n"
            "  - NVIDIA GPU with CUDA 12.1+\n"
            "  - Python 3.9-3.12\n"
            "=" * 80 + "\n"
        )
        sys.exit(1)

    # Set environment variables that the HADL vLLM plugin reads
    if enable_hadl:
        os.environ["HADL_VLLM_ENABLED"] = "1"
        os.environ["HADL_VLLM_K_STEPS"] = str(hadl_k_steps)
    else:
        os.environ["HADL_VLLM_ENABLED"] = "0"

    if resolved_api_key:
        os.environ["VLLM_API_KEY"] = resolved_api_key

    print("=" * 80)
    print(f"[*] LAUNCHING vLLM + HADL HIGH-THROUGHPUT SERVER")
    print(f"[*] Model              : {model_id_or_path}")
    print(f"[*] HADL Augmentation  : {'ENABLED (k={})'.format(hadl_k_steps) if enable_hadl else 'DISABLED'}")
    print(f"[*] Tensor Parallel    : {tensor_parallel_size}")
    print(f"[*] Quantization       : {quantization or 'None (native precision)'}")
    print(f"[*] Max Concurrent Seq : {max_num_seqs}")
    print(f"[*] Endpoint           : http://{host}:{port}/v1")
    print("=" * 80)

    # Build the vLLM CLI command and run it through the vLLM entrypoint
    serve_args = [
        "--model", model_id_or_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--dtype", dtype,
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-seqs", str(max_num_seqs),
        "--swap-space", "4",
    ]

    if max_model_len is not None:
        serve_args.extend(["--max-model-len", str(max_model_len)])
    if quantization is not None:
        serve_args.extend(["--quantization", quantization])
    if trust_remote_code:
        serve_args.append("--trust-remote-code")
    if enforce_eager:
        serve_args.append("--enforce-eager")
    if api_key:
        serve_args.extend(["--api-key", api_key])
    if extra_args:
        serve_args.extend(extra_args)

    # Launch via vLLM's serve entrypoint
    try:
        from vllm.scripts import main as vllm_main

        # Prepend 'serve' to the args for vLLM's CLI parser
        sys.argv = ["vllm", "serve"] + serve_args
        vllm_main()
    except ImportError:
        # Fallback: use subprocess to call `vllm serve`
        import subprocess
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + serve_args
        print(f"[*] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
