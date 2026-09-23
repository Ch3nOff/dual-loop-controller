"""
HADL Cognitive Runtime Package.

Provides universal model auto-detection, OpenAI-compatible REST server,
WebSocket telemetry broadcasting, and an interactive cognitive cockpit.
"""

from .detector import (
    auto_attach_hadl,
    detect_architecture_family,
    determine_optimal_bottleneck,
    determine_optimal_hook,
    AutoDetectionResult,
    create_mock_detected_model,
)

from .server import (
    app,
    start_server,
    initialize_engine,
    engine_state,
)

__all__ = [
    "auto_attach_hadl",
    "detect_architecture_family",
    "determine_optimal_bottleneck",
    "determine_optimal_hook",
    "AutoDetectionResult",
    "create_mock_detected_model",
    "app",
    "start_server",
    "initialize_engine",
    "engine_state",
]
