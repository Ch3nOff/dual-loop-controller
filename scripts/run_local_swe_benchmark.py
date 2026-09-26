#!/usr/bin/env python3
"""Local SWE-bench Evaluation & Benchmark Runner for Gemma 4 Developer Agent.

Wraps task inspection, sandbox initialization, and patch verification.
Can inspect tasks from competition/tasks.jsonl and test agent performance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_tasks(tasks_path: Path, repo_filter: str | None = None) -> List[Dict[str, Any]]:
    tasks = []
    if not tasks_path.exists():
        print(f"[!] Tasks file not found at {tasks_path}")
        return tasks

    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if repo_filter and repo_filter.lower() not in t.get("repo", "").lower():
                    continue
                tasks.append(t)
    return tasks


def show_task_summary(tasks: List[Dict[str, Any]]) -> None:
    print(f"\n{'='*75}")
    print(f"Loaded {len(tasks)} tasks.")
    repos: Dict[str, int] = {}
    for t in tasks:
        r = t.get("repo", "unknown")
        repos[r] = repos.get(r, 0) + 1

    for r, count in sorted(repos.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {r:25s}: {count:3d} tasks")
    print(f"{'='*75}\n")


def inspect_task(task_id: str, tasks: List[Dict[str, Any]]) -> None:
    matched = [t for t in tasks if t.get("instance_id") == task_id]
    if not matched:
        print(f"[!] Task ID '{task_id}' not found.")
        return

    task = matched[0]
    print(f"\n{'='*75}")
    print(f"Task ID: {task.get('instance_id')}")
    print(f"Repository: {task.get('repo')} (base commit: {task.get('base_commit')})")
    print(f"{'-'*75}")
    print("Problem Statement:")
    print(task.get("problem_statement", "")[:1000])
    if len(task.get("problem_statement", "")) > 1000:
        print("... [truncated]")
    print(f"{'-'*75}")
    if task.get("hints_text"):
        print("Hints:")
        print(task.get("hints_text")[:500])
        print(f"{'-'*75}")
    patch = task.get("patch", "")
    print(f"Ground Truth Patch ({len(patch.splitlines())} lines):")
    for line in patch.splitlines()[:15]:
        print(f"  {line}")
    if len(patch.splitlines()) > 15:
        print(f"  ... and {len(patch.splitlines()) - 15} more lines")
    print(f"{'='*75}\n")


def run_submission_eval(
    submission_dir: Path,
    tasks_file: Path,
    task_id: str | None = None,
    sandbox: str = "subprocess",
    max_tool_calls: int = 50,
    max_time_minutes: int = 30,
) -> None:
    """Executes official swegemma eval CLI if available or prints command."""
    print(f"[*] Preparing evaluation for submission: {submission_dir}")
    print(f"    Sandbox Mode: {sandbox} | Max Tool Calls: {max_tool_calls} | Time: {max_time_minutes}m")

    cmd = [
        "swegemma",
        "eval",
        "--tasks", str(tasks_file),
        "--submission-dir", str(submission_dir),
        "--results-dir", "results/run_latest",
        "--sandbox", sandbox,
        "--max-tool-calls", str(max_tool_calls),
        "--max-time-minutes", str(max_time_minutes),
    ]
    if task_id:
        cmd.extend(["--task-id", task_id])

    print("\n[+] To execute swegemma evaluation runner:")
    print("    " + " ".join(cmd) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 SWE Local Benchmark Runner.")
    parser.add_argument("--tasks", type=Path, default=Path("competition/tasks.jsonl"))
    parser.add_argument("--submission", type=Path, default=Path("gemma4_hadl_agent"))
    parser.add_argument("--task-id", type=str, default=None, help="Inspect or evaluate specific task ID")
    parser.add_argument("--repo", type=str, default=None, help="Filter by repository name (fastapi, rich, requests, httpx)")
    parser.add_argument("--inspect", action="store_true", help="Print task details and ground truth patch")
    parser.add_argument("--sandbox", type=str, default="subprocess", choices=["docker", "subprocess"])
    args = parser.parse_args()

    tasks = load_tasks(args.tasks, repo_filter=args.repo)
    if not tasks:
        sys.exit(1)

    if args.inspect:
        if args.task_id:
            inspect_task(args.task_id, tasks)
        else:
            show_task_summary(tasks)
            print("Pass --task-id <id> with --inspect to view full task details.")
    else:
        show_task_summary(tasks)
        run_submission_eval(
            submission_dir=args.submission,
            tasks_file=args.tasks,
            task_id=args.task_id,
            sandbox=args.sandbox,
        )


if __name__ == "__main__":
    main()
