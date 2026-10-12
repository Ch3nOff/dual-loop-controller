"""
ARC-AGI-2 Teaching & Epistemic Grounding Curriculum:
Provides rich conceptual tutoring to X-Star 2B Autonomous Brain covering the 8 Core Knowledge Priors of ARC:
1. Geometric Symmetries & Reflections
2. Discrete Rotations & Translations
3. Object Segmentation & Bounding Box Cropping
4. Color Remapping & Enclosed Flood Fill
5. Gravity & Physical Obstacle Collisions
6. Pattern Repetition & Tiling
7. Cardinality & Size Comparison
8. Sub-Grid Partitioning & Boolean Mask Operations
"""

import sys
import time
import json
from pathlib import Path
import torch

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
TUTORING_LOG = LOG_DIR / "arc_tutoring_session_log.txt"
TUTORING_AUDIT = LOG_DIR / "arc_tutoring_audit.json"

CURRICULUM_MODULES = [
    {
        "id": "ARC_PRIOR_01",
        "title": "Geometric Reflection & Axis Symmetry",
        "concept": "Objects reflected across horizontal, vertical, or diagonal axes preserve shapes while inverting orientation.",
        "lesson": (
            "Teacher: Let us master ARC-AGI Reflection Symmetry.\n"
            "Concept: When a grid has an axis of symmetry (e.g., center line), the output reflects elements.\n"
            "Example Input (3x3):\n"
            "1 0 0\n"
            "2 0 0\n"
            "3 0 0\n"
            "Rule: Horizontal reflection across the center column.\n"
            "Output (3x3):\n"
            "0 0 1\n"
            "0 0 2\n"
            "0 0 3\n"
            "Python formulation: `output_grid = [row[::-1] for row in input_grid]`.\n"
            "Key Principle: Always identify whether the transformation preserves dimensions and flips row or column indices."
        )
    },
    {
        "id": "ARC_PRIOR_02",
        "title": "Discrete 90-Degree Rotations",
        "concept": "Rotating a 2D matrix clockwise by 90 degrees maps cell (r, c) to (c, H - 1 - r).",
        "lesson": (
            "Teacher: Let us master 90-Degree Rotations in ARC.\n"
            "Concept: A grid of size H x W becomes W x H when rotated 90 degrees.\n"
            "Clockwise 90 deg: cell (r, c) becomes (c, H - 1 - r).\n"
            "Counter-clockwise 90 deg: cell (r, c) becomes (W - 1 - c, r).\n"
            "Example Input (2x3):\n"
            "1 2 3\n"
            "4 5 6\n"
            "Output (3x2 Clockwise):\n"
            "4 1\n"
            "5 2\n"
            "6 3\n"
            "Python formulation: `output_grid = list(zip(*input_grid[::-1]))`.\n"
            "Key Principle: Check if dimensions swap between input (H x W) and output (W x H)."
        )
    },
    {
        "id": "ARC_PRIOR_03",
        "title": "Object Extraction & Minimum Bounding Box Cropping",
        "concept": "Detect contiguous foreground clusters and crop out the minimal subgrid enclosing them.",
        "lesson": (
            "Teacher: Let us master Object Extraction & Cropping.\n"
            "Concept: The background is usually color 0 (black). The task asks to extract a unique colored object.\n"
            "Steps:\n"
            "1. Find all non-zero cells: `coords = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != 0]`.\n"
            "2. Determine bounding box: `r_min = min(r for r, c in coords)`, `r_max = max(r for r, c in coords)`, `c_min = min(c for r, c in coords)`, `c_max = max(c for r, c in coords)`.\n"
            "3. Crop: `[row[c_min : c_max + 1] for row in grid[r_min : r_max + 1]]`.\n"
            "Key Principle: When output grid is strictly smaller than input, check if it matches the bounding box of a target object."
        )
    },
    {
        "id": "ARC_PRIOR_04",
        "title": "Color Remapping & Enclosed Flood Fill",
        "concept": "Colors represent semantics. Enclosed topological boundaries can be filled with a distinct fill color.",
        "lesson": (
            "Teacher: Let us master Topological Enclosures and Flood Fill.\n"
            "Concept: When a set of colored cells forms a closed loop/boundary, the empty interior (color 0) is replaced by a target color.\n"
            "Example: A square border of color 2 (red) surrounds an interior empty cell 0. The output replaces interior 0 with color 3 (green).\n"
            "Color Palette Reference:\n"
            "0: Black (Background), 1: Blue, 2: Red, 3: Green, 4: Yellow,\n"
            "5: Gray, 6: Magenta, 7: Orange, 8: Cyan, 9: Brown.\n"
            "Key Principle: Distinguish between background pixels touching grid borders and enclosed interior pixels."
        )
    },
    {
        "id": "ARC_PRIOR_05",
        "title": "Gravity & Physical Contact Dynamics",
        "concept": "Colored elements slide downwards or in a designated direction until hitting an obstacle or boundary.",
        "lesson": (
            "Teacher: Let us master Gravity Transformations in ARC.\n"
            "Concept: Floating colored cells drop down along columns until resting on an obstacle (or bottom row).\n"
            "Example Column:\n"
            "[1, 0, 0, 2, 0] with obstacle at bottom -> drops to [0, 0, 1, 2, 0] or rests above obstacle 2.\n"
            "Python column gravity:\n"
            "`vals = [c for c in col if c != 0]`\n"
            "`new_col = [0]*(len(col) - len(vals)) + vals`\n"
            "Key Principle: Relative vertical order of falling objects is preserved, empty space moves upwards."
        )
    },
    {
        "id": "ARC_PRIOR_06",
        "title": "Pattern Repetition & Tiling",
        "concept": "A small seed motif (e.g. 3x3) is replicated across a larger canvas or multiplied by an indicator mask.",
        "lesson": (
            "Teacher: Let us master Tiling & Kronecker Scaling.\n"
            "Concept: The input is a small kernel K (e.g. 2x2) and the output is a tiled grid K x M.\n"
            "If an indicator grid has 1s and 0s, replace each 1 with the full motif K, and each 0 with empty space.\n"
            "Python formulation:\n"
            "For row in mask: for c in row: place K if c == 1 else zeros.\n"
            "Key Principle: Check if output dimensions are exact integer multiples of input dimensions (e.g., 3x -> 9x)."
        )
    },
    {
        "id": "ARC_PRIOR_07",
        "title": "Cardinality & Size-Based Selection",
        "concept": "Count pixels per object or color; select the majority, minority, or largest connected component.",
        "lesson": (
            "Teacher: Let us master Cardinality and Size Comparison in ARC.\n"
            "Concept: Multiple colored shapes exist in the input. The output only retains the largest shape, or outputs the color of the most frequent object as a 1x1 grid.\n"
            "Example: If grid has 5 blue pixels and 3 red pixels, majority color is blue (1).\n"
            "Key Principle: Always count non-zero pixels per color palette to see if the rule is `argmax(count(color))`."
        )
    },
    {
        "id": "ARC_PRIOR_08",
        "title": "Divider Lines & Multi-Grid Boolean Operations",
        "concept": "Grids separated by a delimiter color (e.g. color 5 or 4) are combined via logical operations (AND, OR, XOR).",
        "lesson": (
            "Teacher: Let us master Delimiter Lines and Subgrid Boolean Composition.\n"
            "Concept: Often a grid has a dividing line (e.g., a solid column of gray 5s) splitting it into Left and Right subgrids.\n"
            "The output grid is the size of one subgrid, formed by combining Left and Right:\n"
            "- Overlap (AND): pixel active only if active in both.\n"
            "- Union (OR): pixel active if active in either.\n"
            "- Difference (XOR): pixel active if active in exactly one.\n"
            "Key Principle: Detect continuous monochromatic dividing rows/columns to extract sub-frames."
        )
    }
]

def run_tutoring_session():
    import xstar
    model = xstar.load(str(MODEL_DIR))
    info = model.get_info()
    print(f"[+] Loaded model successfully on {model.device} ({info.get('gpu_name', 'GPU')})")
    
    transcripts = []
    total_tokens = 0
    t0 = time.time()
    
    with open(TUTORING_LOG, "w", encoding="utf-8") as f_log:
        f_log.write("=======================================================================\n")
        f_log.write("X-STAR 2B AUTONOMOUS BRAIN: ARC-AGI-2 KNOWLEDGE PRIORS TUTORING SESSION\n")
        f_log.write("=======================================================================\n\n")
        
        for idx, module in enumerate(CURRICULUM_MODULES):
            print(f"\n--- [Module {idx+1}/{len(CURRICULUM_MODULES)}] {module['title']} ---")
            prompt = (
                f"<|im_start|>system\n"
                f"You are the X-Star 2B Autonomous Brain learning the fundamental transformation priors of the ARC-AGI benchmark.\n"
                f"Listen carefully to the teacher's lesson and summarize the exact algorithmic rule and transformation invariant.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Lesson: {module['title']}\n"
                f"Concept: {module['concept']}\n\n"
                f"{module['lesson']}\n\n"
                f"Question: Explain how you identify this pattern and what invariant you track.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            res = model.generate(prompt, max_new_tokens=160, temperature=0.1)
            # Count tokens roughly
            tok_count = len(res.split()) * 1.3
            total_tokens += tok_count
            
            print(f"X-Star Brain Response:\n{res[:200]}...")
            
            record = {
                "module_id": module["id"],
                "title": module["title"],
                "lesson_summary": module["concept"],
                "model_response": res
            }
            transcripts.append(record)
            
            f_log.write(f"MODULE {idx+1}: {module['title']}\n")
            f_log.write(f"CONCEPT: {module['concept']}\n")
            f_log.write(f"MODEL RESPONSE:\n{res}\n")
            f_log.write("-" * 70 + "\n\n")
            f_log.flush()

    total_time = time.time() - t0
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modules_taught": len(CURRICULUM_MODULES),
        "total_time_sec": total_time,
        "estimated_tokens": total_tokens,
        "throughput_tok_s": total_tokens / max(total_time, 0.001),
        "modules": transcripts
    }
    with open(TUTORING_AUDIT, "w", encoding="utf-8") as f_aud:
        json.dump(audit_data, f_aud, indent=2)
        
    print(f"\n[✓] Tutoring Session Complete! Time: {total_time:.2f}s, Log saved to: {TUTORING_LOG}")

if __name__ == "__main__":
    run_tutoring_session()
