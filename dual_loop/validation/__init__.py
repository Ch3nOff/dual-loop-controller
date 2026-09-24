"""
HADL Dual-Loop Validation Package
=================================
Automated integrity and mathematical consistency validation suite for benchmark logs.
"""

from .benchmark_validator import (
    BenchmarkValidator,
    ValidationResult,
    ValidationError,
    validate_benchmark_file,
    validate_benchmark_directory
)

__all__ = [
    "BenchmarkValidator",
    "ValidationResult",
    "ValidationError",
    "validate_benchmark_file",
    "validate_benchmark_directory"
]
