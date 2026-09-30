"""
Dual-Loop Custom Inference Server Package
=========================================
OpenAI-compatible inference server with dynamic VRAM auto-tuning,
enabling models to run without OOM on 8GB/12GB consumer GPUs.
"""

from .vram_tuner import VRAMAutoTuner, AllocationPlan, HardwareProfile, ModelSpec
from .engine import DualLoopInferenceEngine
from .app import create_app

def start_server(
    model_id_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
    host: str = "0.0.0.0",
    port: int = 8000,
    headroom_gib: float | str = "auto",
    forced_regime: str | None = None,
    k_steps: int = 2,
    trust_remote_code: bool = False,
    log_level: str = "info"
) -> None:
    """Initializes the DualLoopInferenceEngine and serves the OpenAI-compatible FastAPI server via uvicorn."""
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

    app = create_app(engine)

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
    "start_server"
]
