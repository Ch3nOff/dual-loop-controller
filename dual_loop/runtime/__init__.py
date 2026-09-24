"""
HADL Cognitive Model Utilities.

Provides universal model auto-detection and Hugging Face Hub packaging.
"""

from .detector import (
    auto_attach_hadl,
    detect_architecture_family,
    determine_optimal_bottleneck,
    determine_optimal_hook,
    AutoDetectionResult,
    create_mock_detected_model,
)

from .hf_publisher import (
    package_hf_bundle,
    publish_to_huggingface,
)

__all__ = [
    "auto_attach_hadl",
    "detect_architecture_family",
    "determine_optimal_bottleneck",
    "determine_optimal_hook",
    "AutoDetectionResult",
    "create_mock_detected_model",
    "package_hf_bundle",
    "publish_to_huggingface",
]

