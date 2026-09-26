#!/usr/bin/env python3
"""Sandbox environment setup and dependency installation for public SWE-bench repositories.

This script runs inside the evaluation sandbox container prior to test execution.
It discovers dependencies from pyproject.toml, setup.cfg, and requirements files,
resolves compatible pre-baked wheels from /wheels, configures git exclude rules,
installs the workspace package in editable mode, and configures pytest discovery.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

WHEELS_DIR = Path("/wheels")


def normalize_pkg_name(name: str) -> str:
    """Normalizes package name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def is_prebaked_env() -> bool:
    """Checks if the sandbox environment already has pre-baked wheels installed."""
    return os.environ.get("SWE_PREBAKED", "").lower() in ("1", "true", "yes")


def find_site_packages_dir() -> Path | None:
    """Finds the primary site-packages directory for the current Python interpreter."""
    import site
    packages = site.getsitepackages()
    for p in packages:
        candidate = Path(p)
        if candidate.exists() and os.access(candidate, os.W_OK):
            return candidate
    user_site = site.getusersitepackages()
    if user_site:
        candidate = Path(user_site)
        if candidate.exists() and os.access(candidate, os.W_OK):
            return candidate
    for p in sys.path:
        candidate = Path(p)
        if candidate.name == "site-packages" and candidate.exists() and os.access(candidate, os.W_OK):
            return candidate
    return None


def parse_wheel_filename(filename: str) -> dict[str, Any] | None:
    """Parses a wheel filename according to PEP 427."""
    if not filename.endswith(".whl"):
        return None
    stem = filename[:-4]
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    distribution = parts[0]
    version = parts[1]
    pyver = parts[-3]
    abi = parts[-2]
    plat = parts[-1]
    return {
        "distribution": distribution,
        "normalized": normalize_pkg_name(distribution),
        "version": version,
        "pyver": pyver,
        "abi": abi,
        "plat": plat,
        "filename": filename,
    }


def parse_semver_for_sort(version_str: str) -> tuple[int, ...]:
    """Extracts numeric semver components for sorting wheel versions."""
    nums = [int(p) for p in re.findall(r"\d+", version_str)]
    return tuple(nums) if nums else (0,)


def deduplicate_wheels(wheel_files: list[str]) -> dict[str, str]:
    """Selects the highest version wheel per normalized distribution name."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for filename in wheel_files:
        info = parse_wheel_filename(filename)
        if info is not None:
            grouped.setdefault(info["normalized"], []).append(info)

    selected: dict[str, str] = {}
    for norm_name, candidates in grouped.items():
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                parse_semver_for_sort(c["version"]),
                1 if "py3" in c["pyver"] or "py2.py3" in c["pyver"] else 0,
                c["filename"],
            ),
            reverse=True,
        )
        selected[norm_name] = sorted_candidates[0]["filename"]
    return selected


def get_available_packages(wheels_dir: Path = WHEELS_DIR) -> dict[str, str]:
    """Returns mapping of normalized package name to wheel filename in wheels_dir."""
    if not wheels_dir.exists():
        return {}
    all_whl = [f for f in os.listdir(wheels_dir) if f.endswith(".whl")]
    return deduplicate_wheels(all_whl)


def clean_requirement_line(line: str) -> str:
    """Removes comments, markers, extras, and version constraints from a requirement line."""
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        return ""
    line = re.sub(r";.*$", "", line)
    line = re.sub(r"\[.*?\]", "", line)
    line = re.split(r"[=<>!~]", line)[0]
    return line.strip()


def collect_from_pyproject(pyproject_path: Path) -> list[str]:
    """Collects dependencies and test/optional-dependencies from pyproject.toml."""
    if not pyproject_path.exists() or tomllib is None:
        return []
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        print(f"Warning: could not parse {pyproject_path}: {e}", file=sys.stderr)
        return []

    collected: list[str] = []
    project = data.get("project", {})
    for dep in project.get("dependencies", []):
        cleaned = clean_requirement_line(dep)
        if cleaned:
            collected.append(cleaned)
    for group, deps in project.get("optional-dependencies", {}).items():
        for dep in deps:
            cleaned = clean_requirement_line(dep)
            if cleaned:
                collected.append(cleaned)
    return collected


def collect_from_requirements_files(workspace: Path) -> list[str]:
    """Collects dependency names from requirements*.txt and test-requirements*.txt."""
    collected: list[str] = []
    req_files = list(workspace.glob("**/requirements*.txt")) + list(workspace.glob("**/test-requirements*.txt"))
    for rf in req_files:
        if "docker/" in str(rf) or ".git" in str(rf):
            continue
        try:
            for line in rf.read_text(encoding="utf-8", errors="replace").splitlines():
                cleaned = clean_requirement_line(line)
                if cleaned:
                    collected.append(cleaned)
        except Exception:
            pass
    return collected


def setup_git_exclude(workspace: Path) -> None:
    """Configures git exclude rules to keep git diff clean."""
    exclude_file = workspace / ".git" / "info" / "exclude"
    exclude_file.parent.mkdir(parents=True, exist_ok=True)
    rules = "__pycache__/\n*.pyc\n.pytest_cache/\n*.egg-info/\nbuild/\ndist/\n.coverage\n"
    try:
        if exclude_file.exists():
            existing = exclude_file.read_text(encoding="utf-8", errors="replace")
            to_add = [r for r in rules.splitlines() if r not in existing]
            if to_add:
                with open(exclude_file, "a", encoding="utf-8") as f:
                    f.write("\n" + "\n".join(to_add) + "\n")
        else:
            exclude_file.write_text(rules, encoding="utf-8")
    except Exception as e:
        print(f"Notice: setup_git_exclude skipped: {e}", file=sys.stderr)


def install_editable_package(workspace: Path) -> None:
    """Installs the workspace repository in editable mode without dependencies."""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        f"--find-links={WHEELS_DIR}",
        "--no-build-isolation",
        "--no-deps",
        "-e",
        str(workspace),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            fallback = [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(workspace)]
            subprocess.run(fallback, capture_output=True, text=True, check=False)
    except Exception as e:
        print(f"Notice: install_editable_package skipped: {e}", file=sys.stderr)


def configure_workspace_pytest_ini(workspace: Path) -> None:
    """Configures pytest.ini in the workspace root for robust test discovery."""
    ini_path = workspace / "pytest.ini"
    existing = ""
    if ini_path.exists():
        try:
            existing = ini_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            existing = ""

    default_addopts = "--import-mode=importlib -p no:anyio"
    norecursedirs_val = ".* build dist venv"

    if "[pytest]" in existing:
        lines = []
        has_addopts = False
        has_norecursedirs = False
        for line in existing.splitlines():
            if line.strip().startswith("addopts"):
                has_addopts = True
                curr_opts = line.split("=", 1)[1] if "=" in line else ""
                for opt in default_addopts.split():
                    if opt not in curr_opts:
                        line = line.rstrip() + " " + opt
                lines.append(line)
            elif line.strip().startswith("norecursedirs"):
                has_norecursedirs = True
                for d in norecursedirs_val.split():
                    if d not in line:
                        line = line.rstrip() + " " + d
                lines.append(line)
            else:
                lines.append(line)
        if not has_addopts:
            lines.append(f"addopts = {default_addopts}")
        if not has_norecursedirs:
            lines.append(f"norecursedirs = {norecursedirs_val}")
        content = "\n".join(lines) + "\n"
    else:
        content = f"""[pytest]
addopts = {default_addopts}
norecursedirs = {norecursedirs_val}
python_classes = Test* *Test
python_files = test_*.py *_test.py
filterwarnings =
    ignore::DeprecationWarning
    ignore::UserWarning
""" + (("\n" + existing) if existing.strip() else "")

    try:
        ini_path.write_text(content, encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not write {ini_path}: {e}", file=sys.stderr)


def configure_workspace_conftest(workspace: Path) -> None:
    """Configures conftest.py in the workspace root to assist test discovery."""
    conftest_path = workspace / "conftest.py"
    existing = ""
    if conftest_path.exists():
        try:
            existing = conftest_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            existing = ""

    hook_header = "# Standard test discovery hook for SWE-gemma public benchmark\n"
    if hook_header.strip() not in existing:
        new_content = hook_header + (existing + "\n" if existing else "")
        try:
            conftest_path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            print(f"Warning: could not write {conftest_path}: {e}", file=sys.stderr)


def execute_fast_path(
    workspace: Path | None = None,
    repo: str = "",
    site_packages_dir: Path | None = None,
) -> None:
    """Fast-path dependency injection: injects workspace paths (< 0.05s)."""
    ws = workspace if workspace is not None else (Path("/workspace") if Path("/workspace").exists() else Path.cwd())
    ws_resolved = ws.resolve()

    sp_dir = site_packages_dir or find_site_packages_dir()
    if sp_dir is None:
        print("Warning: No site-packages directory found.", file=sys.stderr)
        return

    with contextlib.suppress(Exception):
        sp_dir.mkdir(parents=True, exist_ok=True)

    # Configure workspace_paths.pth in site-packages
    target_pth = sp_dir / "workspace_paths.pth"
    paths_to_add = [str(ws_resolved)]
    if ws_resolved.exists() and ws_resolved.is_dir():
        for sub in ws_resolved.iterdir():
            if sub.is_dir() and not sub.name.startswith("."):
                paths_to_add.append(str(sub))
    try:
        target_pth.write_text("\n".join(paths_to_add) + "\n")
    except Exception as e:
        print(f"Warning: could not write {target_pth}: {e}", file=sys.stderr)

    configure_workspace_pytest_ini(ws_resolved)
    configure_workspace_conftest(ws_resolved)


def main(
    repo: str | None = None,
    fast_path: bool | None = None,
    workspace: Path | str | None = None,
) -> None:
    args = list(sys.argv[1:])
    parsed_repo = repo
    parsed_fast_path = fast_path
    parsed_workspace = Path(workspace) if workspace else None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--fast-path", "-f"):
            parsed_fast_path = True
        elif arg == "--no-fast-path":
            parsed_fast_path = False
        elif arg in ("--workspace", "-w") and i + 1 < len(args):
            parsed_workspace = Path(args[i + 1])
            i += 1
        elif not arg.startswith("-") and parsed_repo is None:
            parsed_repo = arg
        i += 1

    if parsed_repo is None:
        parsed_repo = ""

    ws = parsed_workspace if parsed_workspace is not None else (Path("/workspace") if Path("/workspace").exists() else Path.cwd())

    setup_git_exclude(ws)
    install_editable_package(ws)

    should_fast_path = parsed_fast_path if parsed_fast_path is not None else is_prebaked_env()

    if should_fast_path:
        print(f"Executing fast-path dependency injection for repo={parsed_repo} in {ws}...")
        execute_fast_path(ws, repo=parsed_repo)
        print("Fast-path dependency injection complete.")
        return

    # Slow path: resolve available wheels and install
    available = get_available_packages()
    print(f"Available wheels: {len(available)} packages")

    all_deps: list[str] = []
    pyproject_path = ws / "pyproject.toml"
    if pyproject_path.exists():
        all_deps.extend(collect_from_pyproject(pyproject_path))
    all_deps.extend(collect_from_requirements_files(ws))

    to_install: list[Path] = []
    for dep in set(all_deps):
        norm = normalize_pkg_name(dep)
        if norm in available:
            to_install.append(WHEELS_DIR / available[norm])

    if to_install:
        print(f"Installing {len(to_install)} resolved wheels via pip...")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            f"--find-links={WHEELS_DIR}",
            "--no-build-isolation",
        ] + [str(w) for w in to_install]
        subprocess.run(cmd, check=False)

    execute_fast_path(ws, repo=parsed_repo)


if __name__ == "__main__":
    main()
