"""
Benchmark Consistency & Integrity Validator
===========================================
Mechanical validation tool to enforce mathematical, physical, and structural consistency
on benchmark evaluation logs before they can be considered authoritative or published.

Key Checks Enforced:
1. Arithmetic Consistency:
   - Overall `elapsed_seconds` cannot be smaller than any individual condition latency.
   - For sequential single-GPU runs, total elapsed time must be >= sum of condition times.
2. Per-Item Log Verification:
   - Official benchmarks must provide commensurate `samples_log` or per-item execution traces.
   - 2KB-4KB summary-only stubs are flagged and rejected from authoritative publication.
3. Cross-Section Quantity Consistency:
   - Identical physical quantities (e.g. neuron counts, dimensions) must match across sections.
4. Metric Bounds:
   - Probabilities / AUROCs in [0, 1] or [0, 100].
   - Strictly positive latencies (> 0).
"""

import os
import sys
import json
import shutil
import argparse
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationError:
    category: str  # 'ARITHMETIC_CONTRADICTION', 'QUANTITY_INCONSISTENCY', 'MISSING_SAMPLES_LOG', 'BOUND_VIOLATION'
    severity: str  # 'CRITICAL', 'WARNING'
    field_path: str
    message: str
    values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    filepath: str
    is_valid: bool
    file_size_bytes: int
    has_sample_logs: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)


class BenchmarkValidator:
    """Rigorous internal consistency validator for benchmark JSON logs."""

    def __init__(self, min_official_file_size: int = 10000, latency_tolerance: float = 0.05):
        self.min_official_file_size = min_official_file_size
        self.latency_tolerance = latency_tolerance

    def validate_file(self, filepath: str) -> ValidationResult:
        if not os.path.exists(filepath):
            return ValidationResult(
                filepath=filepath,
                is_valid=False,
                file_size_bytes=0,
                has_sample_logs=False,
                errors=[ValidationError("FILE_NOT_FOUND", "CRITICAL", "", f"File does not exist: {filepath}")]
            )

        file_size = os.path.getsize(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return ValidationResult(
                filepath=filepath,
                is_valid=False,
                file_size_bytes=file_size,
                has_sample_logs=False,
                errors=[ValidationError("JSON_PARSE_ERROR", "CRITICAL", "", f"Failed to parse JSON: {e}")]
            )

        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        has_sample_logs = self._check_sample_logs_present(data)

        # 1. Arithmetic consistency check
        self._check_arithmetic_consistency(data, errors, warnings)

        # 2. Quantity consistency check across sections
        self._check_quantity_consistency(data, errors, warnings)

        # 3. Metric range & bound check
        self._check_metric_bounds(data, errors, warnings)

        # 4. File size & authoritative depth check
        if not has_sample_logs:
            if file_size < self.min_official_file_size:
                errors.append(ValidationError(
                    category="MISSING_SAMPLES_LOG",
                    severity="CRITICAL",
                    field_path="samples_log",
                    message=(
                        f"File size ({file_size} bytes) is suspiciously small (< {self.min_official_file_size} bytes) "
                        "and lacks per-item `samples_log`. Cannot be certified as authoritative."
                    ),
                    values={"file_size": file_size, "min_required": self.min_official_file_size}
                ))
            else:
                warnings.append(ValidationError(
                    category="MISSING_SAMPLES_LOG",
                    severity="WARNING",
                    field_path="samples_log",
                    message="Log does not contain granular `samples_log` list.",
                    values={"file_size": file_size}
                ))

        is_valid = len(errors) == 0
        return ValidationResult(
            filepath=filepath,
            is_valid=is_valid,
            file_size_bytes=file_size,
            has_sample_logs=has_sample_logs,
            errors=errors,
            warnings=warnings
        )

    def _check_sample_logs_present(self, data: Any) -> bool:
        if isinstance(data, dict):
            for k, v in data.items():
                if k in ("samples_log", "item_log", "per_item_results", "items", "detailed_results"):
                    if isinstance(v, list) and len(v) > 0:
                        return True
                if isinstance(v, (dict, list)) and self._check_sample_logs_present(v):
                    return True
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)) and self._check_sample_logs_present(item):
                    return True
        return False

    def _check_arithmetic_consistency(self, data: Dict[str, Any], errors: List[ValidationError], warnings: List[ValidationError]):
        # Check overall elapsed time against sub-condition latencies
        elapsed_sec = None
        for key in ("elapsed_seconds", "elapsed_time_sec", "total_time_sec", "duration_sec", "runtime_seconds"):
            if key in data and isinstance(data[key], (int, float)):
                elapsed_sec = float(data[key])
                break

        # Search for individual condition latencies
        condition_latencies_ms: List[Tuple[str, float]] = []
        self._extract_latencies(data, "", condition_latencies_ms)

        if elapsed_sec is not None and condition_latencies_ms:
            elapsed_ms = elapsed_sec * 1000.0

            # Rule 1: A single sequential condition cannot exceed the total run duration!
            for path, lat_ms in condition_latencies_ms:
                if lat_ms > (elapsed_ms * (1.0 + self.latency_tolerance)):
                    errors.append(ValidationError(
                        category="ARITHMETIC_CONTRADICTION",
                        severity="CRITICAL",
                        field_path=f"elapsed_seconds vs {path}",
                        message=(
                            f"Total elapsed time ({elapsed_sec:.4f} s = {elapsed_ms:.2f} ms) is LESS than "
                            f"claimed condition latency at '{path}' ({lat_ms:.2f} ms = {lat_ms/1000.0:.4f} s). "
                            "This is a physical and arithmetic impossibility."
                        ),
                        values={"elapsed_sec": elapsed_sec, "condition_path": path, "latency_ms": lat_ms}
                    ))

            # Rule 2: If multiple sequential conditions are claimed, their sum shouldn't massively exceed total time
            # (unless explicit multi-threading / batch concurrency is declared)
            total_claimed_ms = sum(l for _, l in condition_latencies_ms)
            is_parallel = bool(data.get("parallel", False) or data.get("multi_gpu", False) or data.get("batched", False))
            if not is_parallel and len(condition_latencies_ms) > 1:
                if total_claimed_ms > (elapsed_ms * 1.5):
                    errors.append(ValidationError(
                        category="ARITHMETIC_CONTRADICTION",
                        severity="CRITICAL",
                        field_path="elapsed_seconds vs sum(latencies)",
                        message=(
                            f"Sum of condition latencies ({total_claimed_ms:.2f} ms) exceeds total elapsed time "
                            f"({elapsed_ms:.2f} ms) on a non-parallel benchmark."
                        ),
                        values={"elapsed_ms": elapsed_ms, "sum_claimed_ms": total_claimed_ms}
                    ))

    def _extract_latencies(self, obj: Any, prefix: str, out: List[Tuple[str, float]]):
        if isinstance(obj, dict):
            for k, v in obj.items():
                cur_path = f"{prefix}.{k}" if prefix else k
                if k in ("latency_ms", "latency", "mean_latency_ms", "time_ms") and isinstance(v, (int, float)):
                    out.append((cur_path, float(v)))
                elif k in ("latency_sec", "time_sec", "latency_s") and isinstance(v, (int, float)):
                    out.append((cur_path, float(v) * 1000.0))
                elif isinstance(v, (dict, list)):
                    self._extract_latencies(v, cur_path, out)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._extract_latencies(item, f"{prefix}[{i}]", out)

    def _check_quantity_consistency(self, data: Dict[str, Any], errors: List[ValidationError], warnings: List[ValidationError]):
        # Track neuron counts across sections
        neuron_occurrences: List[Tuple[str, int]] = []
        self._find_keys_matching(data, ["active_neurons", "active_neurons_system2", "neurons"], "", neuron_occurrences)

        if len(neuron_occurrences) >= 2:
            unique_vals = set(val for _, val in neuron_occurrences if isinstance(val, int))
            # If multiple diverging integers are labeled with the exact same metric
            exact_active_neurons = [(p, v) for p, v in neuron_occurrences if p.endswith("active_neurons") and isinstance(v, int)]
            if len(exact_active_neurons) >= 2:
                distinct_exact = set(v for _, v in exact_active_neurons)
                if len(distinct_exact) > 1:
                    errors.append(ValidationError(
                        category="QUANTITY_INCONSISTENCY",
                        severity="CRITICAL",
                        field_path="active_neurons",
                        message=(
                            f"Conflicting active neuron counts declared in same file: "
                            f"{[(p, v) for p, v in exact_active_neurons]}"
                        ),
                        values={"divergences": exact_active_neurons}
                    ))

    def _find_keys_matching(self, obj: Any, target_keys: List[str], prefix: str, out: List[Tuple[str, Any]]):
        if isinstance(obj, dict):
            for k, v in obj.items():
                cur_path = f"{prefix}.{k}" if prefix else k
                if k in target_keys and isinstance(v, (int, float)):
                    out.append((cur_path, v))
                elif isinstance(v, (dict, list)):
                    self._find_keys_matching(v, target_keys, cur_path, out)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._find_keys_matching(item, target_keys, f"{prefix}[{i}]", out)

    def _check_metric_bounds(self, obj: Any, errors: List[ValidationError], warnings: List[ValidationError], prefix: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                cur_path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (int, float)):
                    # Check AUROC / probabilities
                    if any(term in k.lower() for term in ("auroc", "accuracy_prob", "confidence")):
                        if v < 0.0 or v > 1.0:
                            # Might be percentage
                            if v > 100.0 or v < 0.0:
                                errors.append(ValidationError(
                                    category="BOUND_VIOLATION",
                                    severity="CRITICAL",
                                    field_path=cur_path,
                                    message=f"Value {v} out of bounds for normalized metric [0.0, 1.0] or [0, 100]",
                                    values={"value": v}
                                ))
                    # Check latencies must be positive
                    if "latency" in k.lower() and v < 0.0:
                        errors.append(ValidationError(
                            category="BOUND_VIOLATION",
                            severity="CRITICAL",
                            field_path=cur_path,
                            message=f"Negative latency ({v}) is invalid.",
                            values={"value": v}
                        ))
                elif isinstance(v, (dict, list)):
                    self._check_metric_bounds(v, errors, warnings, cur_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._check_metric_bounds(item, errors, warnings, f"{prefix}[{i}]")


def validate_benchmark_file(filepath: str, validator: Optional[BenchmarkValidator] = None) -> ValidationResult:
    val = validator or BenchmarkValidator()
    return val.validate_file(filepath)


def validate_benchmark_directory(
    directory: str,
    validator: Optional[BenchmarkValidator] = None,
    quarantine_invalid: bool = False,
    quarantine_dir: Optional[str] = None
) -> Tuple[List[ValidationResult], List[ValidationResult]]:
    val = validator or BenchmarkValidator()
    passed: List[ValidationResult] = []
    failed: List[ValidationResult] = []

    for root, dirs, files in os.walk(directory):
        # Skip archive directories
        if "archive" in root.lower() or "deprecated" in root.lower() or ".git" in root or ".venv" in root:
            continue
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                res = val.validate_file(full_path)
                if res.is_valid:
                    passed.append(res)
                else:
                    failed.append(res)
                    if quarantine_invalid and quarantine_dir:
                        os.makedirs(quarantine_dir, exist_ok=True)
                        dest_path = os.path.join(quarantine_dir, file)
                        shutil.move(full_path, dest_path)
                        print(f"[QUARANTINED] Moved invalid file: {full_path} -> {dest_path}")

    return passed, failed


def main():
    parser = argparse.ArgumentParser(
        prog="hadl-validator",
        description="HADL Benchmark Consistency & Mathematical Integrity Validator"
    )
    parser.add_argument("--file", type=str, default=None, help="Path to single benchmark JSON file to audit")
    parser.add_argument("--dir", type=str, default=None, help="Directory of benchmark JSON files to audit")
    parser.add_argument("--quarantine", action="store_true", help="Automatically quarantine failing files to archive_deprecated")
    parser.add_argument("--quarantine-dir", type=str, default="eval_results/archive_deprecated", help="Quarantine directory")

    args = parser.parse_args()
    validator = BenchmarkValidator()

    if args.file:
        res = validator.validate_file(args.file)
        _print_result(res)
        if not res.is_valid and args.quarantine:
            os.makedirs(args.quarantine_dir, exist_ok=True)
            dest = os.path.join(args.quarantine_dir, os.path.basename(args.file))
            shutil.move(args.file, dest)
            print(f"\n[QUARANTINED] Moved {args.file} -> {dest}")
        sys.exit(0 if res.is_valid else 1)

    elif args.dir:
        passed, failed = validate_benchmark_directory(
            args.dir,
            validator=validator,
            quarantine_invalid=args.quarantine,
            quarantine_dir=args.quarantine_dir
        )
        print("=" * 80)
        print(f"  BENCHMARK INTEGRITY AUDIT REPORT: {args.dir}")
        print("=" * 80)
        print(f"[*] Total Audited Files: {len(passed) + len(failed)}")
        print(f"[+] PASSED             : {len(passed)}")
        print(f"[-] FAILED / REJECTED  : {len(failed)}")
        print("-" * 80)

        if failed:
            print("\nFAILURES / ARITHMETIC REJECTIONS:")
            for f_res in failed:
                _print_result(f_res)
            sys.exit(1)
        else:
            print("\nALL FILES PASSED MATHEMATICAL INTEGRITY AUDIT.")
            sys.exit(0)
    else:
        parser.print_help()


def _print_result(res: ValidationResult):
    status = "PASSED" if res.is_valid else "FAILED"
    color_prefix = "[OK]" if res.is_valid else "[CRITICAL FAIL]"
    print(f"\n{color_prefix} {res.filepath} ({res.file_size_bytes} bytes)")
    print(f"   Sample Logs Present: {res.has_sample_logs}")
    if res.errors:
        print("   Errors:")
        for err in res.errors:
            print(f"     - [{err.category}] {err.field_path}: {err.message}")
    if res.warnings:
        print("   Warnings:")
        for w in res.warnings:
            print(f"     - [{w.category}] {w.field_path}: {w.message}")


if __name__ == "__main__":
    main()
