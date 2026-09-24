import unittest
import os
import tempfile
import json
from dual_loop.validation.benchmark_validator import (
    BenchmarkValidator,
    ValidationError,
    validate_benchmark_file,
    validate_benchmark_directory
)


class TestBenchmarkValidator(unittest.TestCase):
    def setUp(self):
        self.validator = BenchmarkValidator(min_official_file_size=500)
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _write_json(self, name: str, data: dict) -> str:
        path = os.path.join(self.tmp_dir.name, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def test_valid_benchmark_with_sample_logs(self):
        # Create a valid benchmark file
        data = {
            "benchmark": "test_suite",
            "elapsed_seconds": 10.0,
            "total_samples": 2,
            "system_latencies": {
                "base": {"latency_ms": 15.0},
                "dual_loop": {"latency_ms": 25.0}
            },
            "samples_log": [
                {"id": 0, "accuracy": 1.0, "latency_ms": 25.0},
                {"id": 1, "accuracy": 1.0, "latency_ms": 24.5}
            ],
            "padding": "x" * 600  # Ensure size >= 500 bytes
        }
        path = self._write_json("valid_bench.json", data)
        res = self.validator.validate_file(path)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.errors), 0)

    def test_arithmetic_contradiction_latency_exceeds_total(self):
        # elapsed_seconds = 0.5s (500ms), but condition claims 1280ms
        data = {
            "benchmark": "impossible_timing",
            "elapsed_seconds": 0.5,
            "total_samples": 1,
            "conditions": {
                "slow_pass": {"latency_ms": 1280.0}
            },
            "samples_log": [{"id": 0, "status": "ok"}],
            "padding": "x" * 600
        }
        path = self._write_json("arithmetic_contradiction.json", data)
        res = self.validator.validate_file(path)
        self.assertFalse(res.is_valid)
        error_categories = [e.category for e in res.errors]
        self.assertIn("ARITHMETIC_CONTRADICTION", error_categories)

    def test_quantity_inconsistency_divergent_neurons(self):
        # Same key declared with differing integer counts
        data = {
            "benchmark": "neuron_conflict",
            "elapsed_seconds": 100.0,
            "active_neurons": 187402,
            "subsystem": {
                "active_neurons": 185344
            },
            "samples_log": [{"id": 0}],
            "padding": "x" * 600
        }
        path = self._write_json("neuron_conflict.json", data)
        res = self.validator.validate_file(path)
        self.assertFalse(res.is_valid)
        error_categories = [e.category for e in res.errors]
        self.assertIn("QUANTITY_INCONSISTENCY", error_categories)

    def test_missing_samples_log_on_small_file(self):
        # Small stub lacking samples_log
        data = {
            "benchmark": "stub",
            "accuracy": 0.95
        }
        path = self._write_json("tiny_stub.json", data)
        res = self.validator.validate_file(path)
        self.assertFalse(res.is_valid)
        error_categories = [e.category for e in res.errors]
        self.assertIn("MISSING_SAMPLES_LOG", error_categories)

    def test_metric_bound_violation(self):
        # Negative latency or AUROC > 100
        data = {
            "benchmark": "bound_check",
            "elapsed_seconds": 10.0,
            "latency_ms": -5.0,
            "samples_log": [{"id": 0}],
            "padding": "x" * 600
        }
        path = self._write_json("bound_violation.json", data)
        res = self.validator.validate_file(path)
        self.assertFalse(res.is_valid)
        error_categories = [e.category for e in res.errors]
        self.assertIn("BOUND_VIOLATION", error_categories)


if __name__ == "__main__":
    unittest.main()
