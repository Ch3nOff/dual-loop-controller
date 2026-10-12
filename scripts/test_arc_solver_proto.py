import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARC_DIR = PROJECT_ROOT / "datasets" / "arc_prize_2026"

def format_grid(grid):
    return "\n".join(" ".join(str(c) for c in row) for row in grid)

def main():
    eval_ch = json.load(open(ARC_DIR / "arc-agi_evaluation_challenges.json"))
    eval_sol = json.load(open(ARC_DIR / "arc-agi_evaluation_solutions.json"))
    
    print(f"Loaded {len(eval_ch)} evaluation challenges.")
    for i, (k, task) in enumerate(list(eval_ch.items())[:5]):
        print(f"\n=== Task {i+1}: {k} ===")
        print(f"Num Train pairs: {len(task['train'])}, Num Test pairs: {len(task['test'])}")
        in_grid = task['train'][0]['input']
        out_grid = task['train'][0]['output']
        print(f"Train 0 Input: {len(in_grid)}x{len(in_grid[0])}")
        print(f"Train 0 Output: {len(out_grid)}x{len(out_grid[0])}")
        test_in = task['test'][0]['input']
        test_sol = eval_sol[k][0]
        print(f"Test 0 Input: {len(test_in)}x{len(test_in[0])}")
        print(f"Test 0 Ground Truth: {len(test_sol)}x{len(test_sol[0])}")

if __name__ == "__main__":
    main()
