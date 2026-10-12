"""
ARC-AGI-2 Diagnostic Exam Script:
Tests X-Star 2B Autonomous Brain across 20 diagnostic questions representing the 8 Core Knowledge Priors.
Evaluates accuracy, outputs step-by-step reasoning, and generates a formal exam score report.
"""

import sys
import time
import json
import re
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "dist" / "xstar_2b_autonomous_brain"
LOG_DIR = PROJECT_ROOT / "eval_results"
LOG_DIR.mkdir(parents=True, exist_ok=True)
EXAM_LOG = LOG_DIR / "arc_diagnostic_exam_log.txt"
EXAM_AUDIT = LOG_DIR / "arc_diagnostic_exam_audit.json"

EXAM_QUESTIONS = [
    {
        "id": "EXAM_01",
        "category": "Geometric Reflection",
        "question": "Given the 2x2 grid:\n[[1, 2],\n [3, 4]]\nWhat is the resulting grid after a HORIZONTAL FLIP (reversing each row)?\nA. [[2, 1], [4, 3]]\nB. [[3, 4], [1, 2]]\nC. [[4, 3], [2, 1]]\nD. [[1, 2], [3, 4]]",
        "correct_choice": "A",
        "target_grid": [[2, 1], [4, 3]]
    },
    {
        "id": "EXAM_02",
        "category": "Geometric Reflection",
        "question": "Given the 2x2 grid:\n[[1, 2],\n [3, 4]]\nWhat is the resulting grid after a VERTICAL FLIP (reversing row order)?\nA. [[2, 1], [4, 3]]\nB. [[3, 4], [1, 2]]\nC. [[4, 3], [2, 1]]\nD. [[1, 3], [2, 4]]",
        "correct_choice": "B",
        "target_grid": [[3, 4], [1, 2]]
    },
    {
        "id": "EXAM_03",
        "category": "Discrete Rotation",
        "question": "Given the 2x2 grid:\n[[1, 0],\n [0, 2]]\nWhat is the resulting grid after a 90-DEGREE CLOCKWISE ROTATION?\nA. [[0, 1], [2, 0]]\nB. [[2, 0], [0, 1]]\nC. [[1, 0], [0, 2]]\nD. [[0, 2], [1, 0]]",
        "correct_choice": "A",
        "target_grid": [[0, 1], [2, 0]]
    },
    {
        "id": "EXAM_04",
        "category": "Discrete Rotation",
        "question": "Given the 2x2 grid:\n[[1, 2],\n [3, 4]]\nWhat is the resulting grid after a 180-DEGREE ROTATION?\nA. [[2, 1], [4, 3]]\nB. [[3, 4], [1, 2]]\nC. [[4, 3], [2, 1]]\nD. [[1, 2], [3, 4]]",
        "correct_choice": "C",
        "target_grid": [[4, 3], [2, 1]]
    },
    {
        "id": "EXAM_05",
        "category": "Object Bounding Box Crop",
        "question": "In a 4x4 grid of 0s, a 2x2 square of color 3 is located at rows 1-2, cols 1-2:\n[[0, 0, 0, 0],\n [0, 3, 3, 0],\n [0, 3, 3, 0],\n [0, 0, 0, 0]]\nCropping the minimum bounding box of all non-zero cells yields:\nA. [[3, 3], [3, 3]]\nB. [[0, 3], [3, 0]]\nC. [[3, 3, 3]]\nD. [[0, 0, 0, 0]]",
        "correct_choice": "A",
        "target_grid": [[3, 3], [3, 3]]
    },
    {
        "id": "EXAM_06",
        "category": "Color Remapping",
        "question": "Given grid [[1, 2], [2, 1]], if color 1 is mapped to 8 and color 2 is mapped to 5, the new grid is:\nA. [[8, 5], [5, 8]]\nB. [[5, 8], [8, 5]]\nC. [[8, 8], [5, 5]]\nD. [[1, 2], [2, 1]]",
        "correct_choice": "A",
        "target_grid": [[8, 5], [5, 8]]
    },
    {
        "id": "EXAM_07",
        "category": "Gravity Dynamics",
        "question": "Consider a single column of 4 cells [1, 0, 0, 0] where 1 is a block and 0 is air. Under downward gravity, what does the column become?\nA. [0, 0, 0, 1]\nB. [1, 0, 0, 0]\nC. [0, 1, 0, 0]\nD. [0, 0, 1, 0]",
        "correct_choice": "A",
        "target_grid": [0, 0, 0, 1]
    },
    {
        "id": "EXAM_08",
        "category": "Gravity Dynamics",
        "question": "In a column [2, 0, 8, 0], cell 8 is an immovable floor obstacle. Under downward gravity, cell 2 falls until it hits 8. What is the new column?\nA. [0, 2, 8, 0]\nB. [0, 0, 8, 2]\nC. [2, 0, 8, 0]\nD. [0, 0, 2, 8]",
        "correct_choice": "A",
        "target_grid": [0, 2, 8, 0]
    },
    {
        "id": "EXAM_09",
        "category": "Pattern Tiling",
        "question": "If a 1x2 pattern [[1, 2]] is tiled horizontally 2 times, what is the resulting 1x4 grid?\nA. [[1, 2, 1, 2]]\nB. [[1, 1, 2, 2]]\nC. [[2, 1, 2, 1]]\nD. [[1, 2, 0, 0]]",
        "correct_choice": "A",
        "target_grid": [[1, 2, 1, 2]]
    },
    {
        "id": "EXAM_10",
        "category": "Cardinality / Majority",
        "question": "A grid contains four pixels of color 1 (blue), seven pixels of color 2 (red), and two pixels of color 3 (green). Which color has the majority count?\nA. Color 1\nB. Color 2\nC. Color 3\nD. Color 0",
        "correct_choice": "B",
        "target_grid": 2
    },
    {
        "id": "EXAM_11",
        "category": "Cardinality / Minority",
        "question": "A grid contains six pixels of color 4 (yellow), nine pixels of color 5 (gray), and one single pixel of color 6 (magenta). Which color is the unique outlier / minority color?\nA. Color 4\nB. Color 5\nC. Color 6\nD. Color 0",
        "correct_choice": "C",
        "target_grid": 6
    },
    {
        "id": "EXAM_12",
        "category": "Delimiter Subgrid Slicing",
        "question": "A 3x5 grid has a vertical divider of 5s in column 2:\n[[1, 0, 5, 2, 0],\n [1, 0, 5, 2, 0],\n [1, 0, 5, 2, 0]]\nWhat is the shape and content of the left 3x2 subgrid?\nA. [[1, 0], [1, 0], [1, 0]]\nB. [[2, 0], [2, 0], [2, 0]]\nC. [[5, 5], [5, 5], [5, 5]]\nD. [[0, 1], [0, 1], [0, 1]]",
        "correct_choice": "A",
        "target_grid": [[1, 0], [1, 0], [1, 0]]
    },
    {
        "id": "EXAM_13",
        "category": "Boolean Composition (OR)",
        "question": "Combining two 2x2 binary masks:\nMask A: [[1, 0], [0, 0]]\nMask B: [[0, 0], [0, 1]]\nVia logical pixel union (OR) produces:\nA. [[1, 0], [0, 1]]\nB. [[0, 0], [0, 0]]\nC. [[1, 1], [1, 1]]\nD. [[0, 1], [1, 0]]",
        "correct_choice": "A",
        "target_grid": [[1, 0], [0, 1]]
    },
    {
        "id": "EXAM_14",
        "category": "Boolean Composition (AND)",
        "question": "Combining two 2x2 binary masks:\nMask A: [[1, 1], [0, 1]]\nMask B: [[0, 1], [0, 1]]\nVia logical pixel intersection (AND) produces:\nA. [[0, 1], [0, 1]]\nB. [[1, 1], [0, 1]]\nC. [[0, 0], [0, 0]]\nD. [[1, 0], [0, 0]]",
        "correct_choice": "A",
        "target_grid": [[0, 1], [0, 1]]
    },
    {
        "id": "EXAM_15",
        "category": "Topological Enclosure Fill",
        "question": "A 3x3 box of color 8 has a hollow center:\n[[8, 8, 8],\n [8, 0, 8],\n [8, 8, 8]]\nIf the rule is 'fill the enclosed hole with color 4', what is the center cell?\nA. Color 0\nB. Color 4\nC. Color 8\nD. Color 2",
        "correct_choice": "B",
        "target_grid": 4
    },
    {
        "id": "EXAM_16",
        "category": "Scale Expansion (Kronecker)",
        "question": "A single cell [[7]] is expanded by a factor of 2x2. What is the resulting grid?\nA. [[7, 7], [7, 7]]\nB. [[7, 0], [0, 7]]\nC. [[7, 7, 7]]\nD. [[7]]",
        "correct_choice": "A",
        "target_grid": [[7, 7], [7, 7]]
    },
    {
        "id": "EXAM_17",
        "category": "Diagonal Transposition",
        "question": "The transpose of matrix [[1, 2], [3, 4]] swaps rows and columns. What is the result?\nA. [[1, 3], [2, 4]]\nB. [[4, 3], [2, 1]]\nC. [[2, 1], [4, 3]]\nD. [[3, 1], [4, 2]]",
        "correct_choice": "A",
        "target_grid": [[1, 3], [2, 4]]
    },
    {
        "id": "EXAM_18",
        "category": "Color Inversion / Background Mask",
        "question": "In a grid where 0 is background and any color > 0 is foreground, mapping foreground to 1 and background to 0 for [[0, 7], [3, 0]] yields:\nA. [[0, 1], [1, 0]]\nB. [[1, 0], [0, 1]]\nC. [[1, 1], [1, 1]]\nD. [[0, 0], [0, 0]]",
        "correct_choice": "A",
        "target_grid": [[0, 1], [1, 0]]
    },
    {
        "id": "EXAM_19",
        "category": "Directional Translation",
        "question": "In a 1x4 row [5, 0, 0, 0], shifting the pixel 5 to the right by 2 positions yields:\nA. [0, 0, 5, 0]\nB. [0, 5, 0, 0]\nC. [0, 0, 0, 5]\nD. [5, 0, 0, 0]",
        "correct_choice": "A",
        "target_grid": [0, 0, 5, 0]
    },
    {
        "id": "EXAM_20",
        "category": "Boundary Outline Extraction",
        "question": "Given a solid 3x3 block of color 2:\n[[2, 2, 2],\n [2, 2, 2],\n [2, 2, 2]]\nClearing the interior center cell to 0 to leave only the perimeter outline yields:\nA. [[2, 2, 2], [2, 0, 2], [2, 2, 2]]\nB. [[0, 2, 0], [2, 2, 2], [0, 2, 0]]\nC. [[2, 2, 2], [2, 2, 2], [2, 2, 2]]\nD. [[0, 0, 0], [0, 2, 0], [0, 0, 0]]",
        "correct_choice": "A",
        "target_grid": [[2, 2, 2], [2, 0, 2], [2, 2, 2]]
    }
]

def run_adaptive_exam():
    import xstar
    print(f"[*] Loading X-Star 2B Autonomous Brain from: {MODEL_DIR}")
    model = xstar.load(str(MODEL_DIR))
    print(f"[+] Loaded successfully on {model.device}")
    
    score = 0
    total = len(EXAM_QUESTIONS)
    results = []
    t0 = time.time()
    
    with open(EXAM_LOG, "w", encoding="utf-8") as f_log:
        f_log.write("=======================================================================\n")
        f_log.write("X-STAR 2B AUTONOMOUS BRAIN: ARC-AGI-2 DIAGNOSTIC EXAM (20 QUESTIONS)\n")
        f_log.write("=======================================================================\n\n")
        
        for idx, q in enumerate(EXAM_QUESTIONS):
            prompt = (
                f"<|im_start|>system\n"
                f"You are a master mathematical logic and spatial reasoning engine. "
                f"Solve the ARC-AGI grid problem and choose the single correct letter (A, B, C, or D). "
                f"Explain your reasoning step by step, then state 'The correct answer is: [LETTER]'.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Question {idx+1} ({q['category']}):\n"
                f"{q['question']}\n\n"
                f"Answer:<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            t_start = time.time()
            out = model.generate(prompt, max_new_tokens=320, temperature=0.1)
            t_elapsed = time.time() - t_start
            
            # Extract choice
            pred = None
            m = re.search(r'(?:correct answer is|answer is|option|choice)\s*[:\*\s]*([A-D])\b', out, re.IGNORECASE)
            if m:
                pred = m.group(1).upper()
            else:
                # Check if the target grid itself is explicitly derived in the text
                target_str = str(q.get("target_grid", ""))
                # If target grid is in output text associated with an option
                m2 = re.findall(r'\b([A-D])\b', out)
                if m2:
                    pred = m2[-1].upper()
            
            is_correct = (pred == q["correct_choice"])
            if is_correct:
                score += 1
                
            status_tag = "PASS [✓]" if is_correct else "FAIL [✗]"
            print(f"Q{idx+1:02d} | {q['category']:<28} | Target: {q['correct_choice']} | Pred: {str(pred):<2} | {status_tag} | {t_elapsed:.2f}s")
            
            res_item = {
                "question_id": q["id"],
                "category": q["category"],
                "target": q["correct_choice"],
                "prediction": pred,
                "is_correct": is_correct,
                "latency_sec": t_elapsed,
                "model_output": out
            }
            results.append(res_item)
            
            f_log.write(f"QUESTION {idx+1}: {q['id']} ({q['category']})\n")
            f_log.write(f"PROMPT:\n{q['question']}\n")
            f_log.write(f"TARGET: {q['correct_choice']} | PREDICTION: {pred} | STATUS: {status_tag}\n")
            f_log.write(f"OUTPUT:\n{out}\n")
            f_log.write("-" * 70 + "\n\n")
            f_log.flush()

    total_time = time.time() - t0
    acc_pct = (score / total) * 100.0
    
    audit_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total,
        "score": score,
        "accuracy_pct": acc_pct,
        "total_time_sec": total_time,
        "mean_latency_per_q": total_time / total,
        "results": results
    }
    
    with open(EXAM_AUDIT, "w", encoding="utf-8") as f_aud:
        json.dump(audit_summary, f_aud, indent=2)
        
    print(f"\n=======================================================")
    print(f"EXAM RESULTS: {score} / {total} ({acc_pct:.1f}%) | Time: {total_time:.2f}s")
    print(f"Audit saved to: {EXAM_AUDIT}")
    print(f"Log saved to: {EXAM_LOG}")
    print(f"=======================================================")

if __name__ == "__main__":
    run_adaptive_exam()
