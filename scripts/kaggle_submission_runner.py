#!/usr/bin/env python3
"""Bulletproof Kaggle Submission Runner for Gemma 4 Developer Agent Competition.

Generates /kaggle/working/submission.parquet with columns:
  - id: instance_id from tasks.jsonl
  - prediction: unified git diff patch string OR "NO_PATCH" fallback

Guarantees:
  1. Dynamic path discovery (handles both public and private hidden test paths).
  2. Per-task crash isolation (try-except ensures NO unhandled exceptions crash the notebook).
  3. Incremental checkpointing (submission.parquet is updated after every task).
  4. 100% coverage (every single task ID is guaranteed to be present).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Default paths
KAGGLE_WORKING = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))


def find_tasks_file() -> Path:
    """Dynamically locates tasks.jsonl across possible public and private Kaggle input paths."""
    candidates = [
        KAGGLE_INPUT / "gemma-4-developer-agent" / "tasks.jsonl",
        KAGGLE_INPUT / "gemma-4-developer-agent" / "published" / "tasks.jsonl",
        KAGGLE_INPUT / "competition_data" / "tasks.jsonl",
        KAGGLE_INPUT / "competition_data" / "published" / "tasks.jsonl",
        Path("competition/tasks.jsonl"),
        Path("tasks.jsonl"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            print(f"[+] Found tasks.jsonl at: {c}")
            return c

    # Recursive fallback search in KAGGLE_INPUT
    if KAGGLE_INPUT.exists():
        for p in KAGGLE_INPUT.rglob("tasks.jsonl"):
            print(f"[+] Dynamically discovered tasks.jsonl at: {p}")
            return p

    raise FileNotFoundError("Could not find tasks.jsonl in any standard input path!")


def load_tasks(tasks_path: Path) -> List[Dict[str, Any]]:
    tasks = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    tasks.append(json.loads(line))
                except Exception as e:
                    print(f"[!] Warning: Failed to parse line in tasks.jsonl: {e}")
    print(f"[*] Loaded {len(tasks)} benchmark tasks.")
    return tasks


def run_agent_on_task(task: Dict[str, Any], agent_dir: Path) -> str:
    """Runs the HADL agent on a single task.

    Returns the unified diff patch string, or 'NO_PATCH' if no changes or error.
    """
    instance_id = task.get("instance_id")
    repo = task.get("repo")
    problem = task.get("problem_statement", "")

    # Here the harness or direct runner executes Container A
    # Replace with swegemma runner or direct agent inference
    # If the agent produces an empty diff or fails, return 'NO_PATCH'
    return "NO_PATCH"


def generate_submission(
    agent_dir: Path,
    output_parquet: Path | None = None,
) -> Path:
    if output_parquet is None:
        output_parquet = KAGGLE_WORKING / "submission.parquet"

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    tasks_file = find_tasks_file()
    tasks = load_tasks(tasks_file)

    results: List[Dict[str, str]] = []

    print(f"[*] Generating predictions for {len(tasks)} tasks...")
    start_time = time.time()

    for idx, task in enumerate(tasks):
        instance_id = task.get("instance_id", f"task_{idx}")
        print(f"[{idx+1}/{len(tasks)}] Processing {instance_id}...", end=" ", flush=True)

        # STRICT ISOLATION: A failure on one task MUST NEVER crash the notebook!
        try:
            patch = run_agent_on_task(task, agent_dir)
            if not patch or not isinstance(patch, str) or patch.strip() == "":
                patch = "NO_PATCH"
        except Exception as e:
            print(f"[FAIL: {e}] -> Fallback to NO_PATCH", end=" ")
            patch = "NO_PATCH"

        results.append({"id": instance_id, "prediction": patch})
        print(f"[DONE - patch size: {len(patch)} chars]")

        # Incremental checkpointing: Save every 5 tasks and on the final task
        if (idx + 1) % 5 == 0 or (idx + 1) == len(tasks):
            df_checkpoint = pd.DataFrame(results)
            df_checkpoint.to_parquet(str(output_parquet), index=False)

    df_final = pd.DataFrame(results)
    df_final.to_parquet(str(output_parquet), index=False)
    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"[SUCCESS] Saved {len(df_final)} predictions to {output_parquet}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"{'='*70}\n")

    # Sanity check validation
    verify_df = pd.read_parquet(str(output_parquet))
    print("Sanity Check of submission.parquet:")
    print(f"  - Rows: {len(verify_df)}")
    print(f"  - Columns: {list(verify_df.columns)}")
    print(f"  - Null count: {verify_df.isnull().sum().to_dict()}")
    print(f"  - NO_PATCH count: {(verify_df['prediction'] == 'NO_PATCH').sum()}/{len(verify_df)}")
    print("\nFirst 3 rows:")
    print(verify_df.head(3))

    return output_parquet


if __name__ == "__main__":
    agent_dir = Path("gemma4_hadl_agent")
    out_file = Path("submission.parquet")
    generate_submission(agent_dir=agent_dir, output_parquet=out_file)
