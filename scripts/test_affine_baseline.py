import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARC_DIR = PROJECT_ROOT / "datasets" / "arc_prize_2026"

def check_task(task, sol_grids):
    train_pairs = task["train"]
    test_inputs = [t["input"] for t in task["test"]]
    
    # Check simple candidates
    candidates = []
    
    # Candidate 1: Identity
    candidates.append(("identity", lambda g: [row[:] for row in g]))
    # Candidate 2: H-flip
    candidates.append(("h_flip", lambda g: [row[::-1] for row in g]))
    # Candidate 3: V-flip
    candidates.append(("v_flip", lambda g: g[::-1]))
    # Candidate 4: Rot90
    candidates.append(("rot90", lambda g: [list(r) for r in zip(*g[::-1])]))
    # Candidate 5: Rot180
    candidates.append(("rot180", lambda g: [row[::-1] for row in g[::-1]]))
    # Candidate 6: Rot270
    candidates.append(("rot270", lambda g: [list(r) for r in zip(*g)][::-1]))
    # Candidate 7: Transpose
    candidates.append(("transpose", lambda g: [list(r) for r in zip(*g)]))
    
    # Test each candidate on train
    for name, fn in candidates:
        valid = True
        for pair in train_pairs:
            try:
                pred = fn(pair["input"])
                if pred != pair["output"]:
                    valid = False
                    break
            except Exception:
                valid = False
                break
        if valid:
            # Verified on all train pairs!
            # Test on test pair
            test_pred = fn(test_inputs[0])
            is_match = (test_pred == sol_grids[0])
            return name, is_match
            
    return None, False

def main():
    eval_ch = json.load(open(ARC_DIR / "arc-agi_evaluation_challenges.json"))
    eval_sol = json.load(open(ARC_DIR / "arc-agi_evaluation_solutions.json"))
    
    matched = 0
    for k, task in eval_ch.items():
        name, is_match = check_task(task, eval_sol[k])
        if is_match:
            matched += 1
            print(f"[+] Task {k} solved by {name}!")
            
    print(f"Simple affine transforms solved: {matched} / {len(eval_ch)}")

if __name__ == "__main__":
    main()
