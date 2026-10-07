"""
Dual-Loop Custom Inference Server Package
=========================================
OpenAI-compatible inference server with dynamic VRAM auto-tuning,
enabling models to run without OOM on 8GB/12GB consumer GPUs.
"""

try:
    from .vram_tuner import VRAMAutoTuner, AllocationPlan, HardwareProfile, ModelSpec
    from .engine import DualLoopInferenceEngine
    from .app import create_app
except ImportError:
    VRAMAutoTuner = None  # type: ignore
    AllocationPlan = None  # type: ignore
    HardwareProfile = None  # type: ignore
    ModelSpec = None  # type: ignore
    DualLoopInferenceEngine = None  # type: ignore
    create_app = None  # type: ignore

def start_server(
    model_id_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
    host: str = "127.0.0.1",
    port: int = 8000,
    headroom_gib: float | str = "auto",
    forced_regime: str | None = None,
    k_steps: int = 2,
    trust_remote_code: bool = False,
    api_key: str | None = None,
    log_level: str = "info"
) -> None:
    # SEC-02: Mandate authentication if binding to a non-loopback / public interface
    import os
    is_loopback = host in ("127.0.0.1", "localhost", "::1")
    resolved_api_key = api_key or os.environ.get("DUAL_LOOP_API_KEY", "").strip() or None
    if not is_loopback and not resolved_api_key:
        raise ValueError(
            f"Binding to non-loopback host '{host}' requires an API key for authentication. "
            "Please provide --api-key or set the DUAL_LOOP_API_KEY environment variable."
        )

    if DualLoopInferenceEngine is None or create_app is None:
        raise ImportError(
            "PyTorch and transformers are required to run the standard DualLoopInferenceEngine server.\n"
            "Install with: pip install 'dual-loop-controller[serve]'"
        )

    import uvicorn

    engine = DualLoopInferenceEngine(
        model_id_or_path=model_id_or_path,
        headroom_gib=headroom_gib,
        forced_regime=forced_regime,
        k_steps=k_steps,
        trust_remote_code=trust_remote_code
    )
    # Eagerly load model and print telemetry
    engine.load_model()

    app = create_app(engine, api_key=api_key)

    print("\n" + "=" * 80)
    print(f"[*] DUAL-LOOP INFERENCE SERVER IS LIVE!")
    print(f"[*] Base URL            : http://{host}:{port}/v1")
    print(f"[*] Web Dashboard       : http://localhost:{port}/")
    print(f"[*] OpenAI API Model ID : {engine.model_id}")
    print(f"[*] Ready for Hermes Agent, Cursor, LM Studio, Ollama & LangChain!")
    print("=" * 80 + "\n")

    uvicorn.run(app, host=host, port=port, log_level=log_level)

__all__ = [
    "VRAMAutoTuner",
    "AllocationPlan",
    "HardwareProfile",
    "ModelSpec",
    "DualLoopInferenceEngine",
    "create_app",
    "start_server",
]

# Conditionally export vLLM components when available
try:
    from .vllm_engine import VLLMInferenceEngine
    from .vllm_app import start_vllm_server
    from .vllm_offline import vllm_batch_generate, vllm_compare_base_vs_hadl
    __all__.extend([
        "VLLMInferenceEngine",
        "start_vllm_server",
        "vllm_batch_generate",
        "vllm_compare_base_vs_hadl",
    ])
except ImportError:
    pass
