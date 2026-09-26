#!/usr/bin/env python3
"""Pre-flight submission validator and packaging utility for Gemma 4 Developer Agent Competition.

Validates an agent submission directory against the official Google ADK and swegemma constraints:
  1. Exactly one root config (agent.yaml or root_agent.yaml).
  2. Total unpacked size < 3 GiB (3,221,225,472 bytes).
  3. Strictly permitted file extensions only (.yaml, .yml, .md, .txt, .py, .json, .safetensors).
  4. Single Base Model rule across all agent/sub-agent/tool configs.
  5. Valid !include paths (no path traversal '..' or absolute paths).
  6. Automatic packaging into submission.zip ready for Kaggle upload.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import yaml

# Official competition limits from swegemma / adk-submission
MAX_TOTAL_SIZE_BYTES = 3 * 1024 * 1024 * 1024  # 3 GiB
MAX_FILE_COUNT = 10_000
MAX_YAML_FILES = 1_000
MAX_YAML_SIZE_BYTES = 50 * 1024 * 1024  # 50 MiB
MAX_INSTRUCTION_CHARS = 1_000_000

ALLOWED_EXTENSIONS = {
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".py",
    ".json",
    ".safetensors",
}

ROOT_CONFIG_CANDIDATES = [
    "agent.yaml",
    "agent.yml",
    "root_agent.yaml",
    "root_agent.yml",
]

MODEL_PREFIXES = ["openai/", "google/", "hosted_vllm/", "custom/"]


def normalize_model_name(name: str) -> str:
    """Strips provider prefixes as specified in swegemma/adk-submission."""
    for prefix in MODEL_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


class IncludeConstructor:
    """Safe !include YAML constructor for local validation."""
    def __init__(self, root_dir: Path, current_dir: Path | None = None):
        self.root_dir = root_dir.resolve()
        self.current_dir = (current_dir or root_dir).resolve()
        self.included_files: list[Path] = []

    def construct_include(self, loader: yaml.SafeLoader, node: yaml.Node) -> Any:
        rel_path = loader.construct_scalar(node)
        if not isinstance(rel_path, str):
            raise ValueError(f"Invalid !include scalar: {rel_path}")

        # Resolve relative to the current file's directory
        target = (self.current_dir / rel_path).resolve()

        # Strict security check: must stay within submission root
        try:
            target.relative_to(self.root_dir)
        except ValueError:
            raise ValueError(f"Illegal path traversal outside submission root: {rel_path} -> {target}")

        if not target.exists():
            raise FileNotFoundError(f"!include file does not exist: {target} (from {rel_path})")

        self.included_files.append(target)
        ext = target.suffix.lower()
        if ext in (".md", ".txt"):
            return target.read_text(encoding="utf-8")
        elif ext in (".yaml", ".yml"):
            sub_loader = make_safe_loader(self.root_dir, target.parent)
            with open(target, "r", encoding="utf-8") as f:
                return yaml.load(f, sub_loader)
        elif ext == ".json":
            with open(target, "r", encoding="utf-8") as f:
                return json.load(f)
        return str(target)


def make_safe_loader(root_dir: Path, current_dir: Path) -> type[yaml.SafeLoader]:
    class Loader(yaml.SafeLoader):
        pass

    inc = IncludeConstructor(root_dir, current_dir)
    Loader.add_constructor("!include", inc.construct_include)
    return Loader


def validate_submission_dir(submission_dir: Path) -> dict[str, Any]:
    submission_dir = submission_dir.resolve()
    print(f"[*] Validating submission directory: {submission_dir}")

    if not submission_dir.exists() or not submission_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {submission_dir}")

    # 1. Root Config Discovery
    root_configs = [f for f in ROOT_CONFIG_CANDIDATES if (submission_dir / f).exists()]
    if not root_configs:
        raise ValueError(
            f"MissingRootConfigError: None of {ROOT_CONFIG_CANDIDATES} found in root of {submission_dir}"
        )
    if len(root_configs) > 1:
        raise ValueError(
            f"MultipleRootConfigsError: Found multiple root configs: {root_configs}. Exactly one required."
        )
    root_config_path = submission_dir / root_configs[0]
    print(f"  [+] Found root configuration: {root_configs[0]}")

    # 2. File extensions and size check
    total_size = 0
    file_count = 0
    yaml_count = 0
    illegal_files: list[str] = []

    for root, dirs, files in os.walk(submission_dir):
        # Ignore any accidental .git or __pycache__
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(submission_dir)
            ext = file_path.suffix.lower()

            if ext not in ALLOWED_EXTENSIONS:
                illegal_files.append(str(rel_path))

            size = file_path.stat().st_size
            total_size += size
            file_count += 1
            if ext in (".yaml", ".yml"):
                yaml_count += 1
                if size > MAX_YAML_SIZE_BYTES:
                    raise ValueError(f"YAML file exceeds 50 MiB limit: {rel_path} ({size} bytes)")

    print(f"  [+] Total files: {file_count} (YAML files: {yaml_count})")
    print(f"  [+] Total unpacked size: {total_size / (1024*1024):.2f} MiB ({total_size} bytes)")

    if total_size > MAX_TOTAL_SIZE_BYTES:
        raise ValueError(f"Total size {total_size} bytes exceeds 3 GiB limit ({MAX_TOTAL_SIZE_BYTES})")
    if file_count > MAX_FILE_COUNT:
        raise ValueError(f"File count {file_count} exceeds limit of {MAX_FILE_COUNT}")
    if yaml_count > MAX_YAML_FILES:
        raise ValueError(f"YAML count {yaml_count} exceeds limit of {MAX_YAML_FILES}")

    if illegal_files:
        raise ValueError(
            f"Disallowed file extensions found ({len(illegal_files)} files):\n"
            + "\n".join(f"  - {f}" for f in illegal_files[:10])
            + ("\n  ..." if len(illegal_files) > 10 else "")
            + f"\nAllowed extensions: {sorted(list(ALLOWED_EXTENSIONS))}"
        )

    # 3. Model Registry & Single-Model Check
    declared_models: set[str] = set()

    for root, dirs, files in os.walk(submission_dir):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
        for file in files:
            if file.endswith((".yaml", ".yml")):
                yaml_path = Path(root) / file
                try:
                    loader_cls = make_safe_loader(submission_dir, yaml_path.parent)
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        data = yaml.load(f, loader_cls)
                except Exception as e:
                    raise ValueError(f"Failed to parse YAML file {yaml_path}: {e}")

                if isinstance(data, dict):
                    if "model" in data and isinstance(data["model"], str):
                        declared_models.add(normalize_model_name(data["model"]))

    print(f"  [+] Declared base models found: {declared_models}")
    if len(declared_models) == 0:
        raise ValueError("No model declaration found in any agent configuration!")
    if len(declared_models) > 1:
        raise ValueError(
            f"SingleBaseModelRule Violation: Found {len(declared_models)} different models: {declared_models}. "
            "All agents must use at most ONE declared base model."
        )

    base_model = list(declared_models)[0]
    print(f"  [+] Single base model verified: '{base_model}'")

    # 4. Deep Adapter Verification
    adapters_dir = submission_dir / "adapters"
    discovered_adapters = set()
    if adapters_dir.exists() and adapters_dir.is_dir():
        for d in adapters_dir.iterdir():
            if d.is_dir():
                discovered_adapters.add(d.name)
                cfg = d / "adapter_config.json"
                weights = d / "adapter_model.safetensors"
                if not cfg.exists():
                    raise FileNotFoundError(f"Adapter '{d.name}' is missing adapter_config.json!")
                if not weights.exists():
                    raise FileNotFoundError(f"Adapter '{d.name}' is missing adapter_model.safetensors!")
    print(f"  [+] Discovered valid adapters: {discovered_adapters}")

    # Check that all referenced adapters and agent_tool paths exist
    referenced_adapters: set[str] = set()
    for root, dirs, files in os.walk(submission_dir):
        for file in files:
            if file.endswith((".yaml", ".yml")):
                yaml_path = Path(root) / file
                with open(yaml_path, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped.startswith("adapter:"):
                            a_name = stripped.split("adapter:")[1].strip()
                            if a_name and a_name.lower() != "none" and a_name != "null":
                                referenced_adapters.add(a_name)
                        if "config_path:" in stripped:
                            c_path = stripped.split("config_path:")[1].strip()
                            resolved_tool = (yaml_path.parent / c_path).resolve()
                            if not resolved_tool.exists():
                                resolved_tool = (submission_dir / c_path).resolve()
                            if not resolved_tool.exists():
                                raise FileNotFoundError(
                                    f"agent_tool config_path '{c_path}' in {yaml_path.name} does not exist!"
                                )

    print(f"  [+] Referenced adapters across all YAMLs: {referenced_adapters}")
    missing_adapters = referenced_adapters - discovered_adapters
    if missing_adapters:
        raise ValueError(
            f"Missing required adapters: {missing_adapters}. "
            f"Referenced in configs, but not found in {adapters_dir}"
        )

    print("[SUCCESS] Submission directory passed all validation checks!")
    return {
        "status": "valid",
        "root_config": root_configs[0],
        "base_model": base_model,
        "total_files": file_count,
        "total_size_bytes": total_size,
    }


def package_submission(submission_dir: Path, output_zip: Path) -> Path:
    submission_dir = submission_dir.resolve()
    output_zip = output_zip.resolve()

    print(f"[*] Packaging {submission_dir} into {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(submission_dir):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(submission_dir).as_posix()
                z.write(file_path, arcname)

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Packaged submission.zip successfully ({zip_size_mb:.2f} MiB)!")
    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and package Gemma 4 Developer Agent submission.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("gemma4_hadl_agent"),
        help="Path to submission directory (default: gemma4_hadl_agent)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.zip"),
        help="Path to output zip file (default: submission.zip)",
    )
    parser.add_argument(
        "--package",
        action="store_true",
        default=True,
        help="Package into submission.zip if validation passes",
    )
    args = parser.parse_args()

    try:
        validate_submission_dir(args.dir)
        if args.package:
            package_submission(args.dir, args.output)
    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
