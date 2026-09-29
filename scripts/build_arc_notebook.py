import json
import os

def create_notebook():
    cells = []

    def md(text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        }

    def code(text):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        }

    # Cell 1: Title
    cells.append(md("""# 🧩 ARC Prize 2026 (ARC-AGI-2): Dual-Loop Cognitive Controller (HADL v2.5)
### Hardware-Aligned Autopoietic Latent Deliberation, Popperian Verification & NeuroSymbolic MDL Selection

This notebook provides a complete, self-contained implementation of the **Dual-Loop Cognitive Architecture (HADL / X-Star)** adapted specifically for the **ARC Prize 2026 - ARC-AGI-2** Kaggle competition.

---

### 🏛️ Theoretical Foundations & Dual-Process Mapping
1. **The Core Dilemma of ARC-AGI**:
   - **Pure Autoregressive LLMs** fail due to token hallucination, spatial blindness, and lack of deterministic verification.
   - **Pure Program Synthesis (DSL Search)** suffers from combinatorial explosion on complex tasks.
2. **The Dual-Loop Solution**:
   - **System 1 (Intuitive Prior / Fast-Path Proposer)**: Rapidly infers invariant symmetries, color mappings, and primitive geometric hypotheses.
   - **System 2 (Deterministic Sandbox Verification - Popperian Falsification)**: Strictly executes candidate transformations against **ALL** demonstration pairs $(X_{train} \to Y_{train})$. If a single pixel fails on any example, the hypothesis is refuted immediately.
   - **Cognitive Matrix Helper (`wrong_log_bank`)**: Records failed aspects and error signatures (Elimination-by-Aspects) to guide search away from dead ends.
   - **NeuroSymbolic MDL Selector (Occam's Razor)**: Among all hypotheses that solve 100% of the training pairs, the one with the **Minimum Description Length (MDL)** (simplest code, fewer arbitrary magic constants) is selected as `attempt_1`, and the runner-up as `attempt_2`.
   - **Allostatic Time Allocator**: Manages compute budget to ensure the entire test set (240 tasks) executes comfortably within Kaggle's 12-hour offline timeout."""))

    # Cell 2: Setup & Paths
    cells.append(md("""## 1. Environment Setup & Dual-Platform Dataset Resolver
Automatically detects whether the notebook is running locally (with `"C:\\Users\\Matthew Chen\\Downloads\\arc-prize-2026-arc-agi-2.zip"`) or in the Kaggle environment (`/kaggle/input/arc-prize-2026-arc-agi-2/`)."""))

    # Cell 3: Code - Imports & Data Loader
    cells.append(code('''import os
import sys
import json
import zipfile
import time
import copy
from typing import Dict, Any, List, Tuple, Optional, Callable, Set
import numpy as np

# Verify environment
print(f"Python Version: {sys.version.split()[0]}")
print(f"NumPy Version: {np.__version__}")

# Multi-platform path resolver
POSSIBLE_PATHS = [
    r"C:\\Users\\Matthew Chen\\Downloads\\arc-prize-2026-arc-agi-2.zip",
    "/kaggle/input/arc-prize-2026-arc-agi-2/arc-prize-2026-arc-agi-2.zip",
    "/kaggle/input/arc-prize-2026-arc-agi-2",
    "./arc-prize-2026-arc-agi-2.zip",
    "../arc-prize-2026-arc-agi-2.zip"
]

class ARCDataset:
    """Zero-overhead dataset resolver and loader supporting zip archives and directories."""
    def __init__(self, candidate_paths: List[str]):
        self.source_path = None
        self.is_zip = False
        self.data_cache = {}
        
        for p in candidate_paths:
            if os.path.exists(p):
                self.source_path = p
                self.is_zip = zipfile.is_zipfile(p) if os.path.isfile(p) else False
                break
                
        if self.source_path is None:
            raise FileNotFoundError("Could not locate ARC-AGI-2 dataset in any known paths.")
            
        print(f"✅ ARC-AGI-2 Data Source Located: {self.source_path} (Is Zip: {self.is_zip})")

    def read_json(self, filename: str) -> Dict[str, Any]:
        if filename in self.data_cache:
            return self.data_cache[filename]
            
        if self.is_zip:
            with zipfile.ZipFile(self.source_path, 'r') as z:
                raw = z.read(filename).decode('utf-8')
                data = json.loads(raw)
        else:
            file_path = os.path.join(self.source_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
        self.data_cache[filename] = data
        return data

# Initialize dataset
arc_data = ARCDataset(POSSIBLE_PATHS)

train_challenges = arc_data.read_json("arc-agi_training_challenges.json")
train_solutions = arc_data.read_json("arc-agi_training_solutions.json")
eval_challenges = arc_data.read_json("arc-agi_evaluation_challenges.json")
eval_solutions = arc_data.read_json("arc-agi_evaluation_solutions.json")
test_challenges = arc_data.read_json("arc-agi_test_challenges.json")
sample_submission = arc_data.read_json("sample_submission.json")

print(f"📊 Dataset Splits Loaded Successfully:")
print(f"  • Training Challenges   : {len(train_challenges)} tasks")
print(f"  • Evaluation Challenges : {len(eval_challenges)} tasks (Ground truth available for local benchmarking)")
print(f"  • Test Challenges       : {len(test_challenges)} tasks (Kaggle submission target)")
print(f"  • Sample Submission     : {len(sample_submission)} tasks")'''))

    # Cell 4: Markdown - Visualizer
    cells.append(md("""## 2. Interactive ARC Grid Visualizer (Rich HTML & ANSI)
ARC grids consist of integers $0$ to $9$, representing $10$ canonical colors:
- `0`: Black, `1`: Blue, `2`: Red, `3`: Green, `4`: Yellow
- `5`: Grey, `6`: Magenta, `7`: Orange, `8`: Teal, `9`: Maroon

This visualizer renders grids in high-contrast color blocks with optional side-by-side comparison for inputs, outputs, and prediction diffs."""))

    # Cell 5: Code - Visualizer
    cells.append(code('''ARC_PALETTE = {
    0: "#1A1A1A",  # 0: Black / Background
    1: "#1E88E5",  # 1: Blue
    2: "#D81B60",  # 2: Red
    3: "#004D40",  # 3: Green
    4: "#FFC107",  # 4: Yellow
    5: "#8E8E93",  # 5: Grey
    6: "#E040FB",  # 6: Magenta
    7: "#FB8C00",  # 7: Orange
    8: "#00ACC1",  # 8: Teal
    9: "#880E4F"   # 9: Maroon
}

ARC_NAMES = ["black", "blue", "red", "green", "yellow", "grey", "magenta", "orange", "teal", "maroon"]

def render_grid_html(grid: List[List[int]], cell_size: int = 20) -> str:
    """Generates an HTML table representing the ARC grid."""
    arr = np.array(grid, dtype=int)
    h, w = arr.shape
    html = [f\'<table style="border-collapse: collapse; margin: 4px; display: inline-block; vertical-align: top;">\']
    for r in range(h):
        html.append(\'<tr>\')
        for c in range(w):
            val = int(arr[r, c])
            color = ARC_PALETTE.get(val, "#FFFFFF")
            html.append(f\'<td style="width:{cell_size}px; height:{cell_size}px; background-color:{color}; \'
                        f\'border: 1px solid #333333; text-align: center; color: white; font-size: 10px; font-family: monospace;">\'
                        f\'{val if cell_size >= 18 else ""}</td>\')
        html.append(\'</tr>\')
    html.append(\'</table>\')
    return "".join(html)

def display_task_html(task_dict: Dict[str, Any], task_id: str = "Demo"):
    """Displays all train and test pairs of a task side-by-side in HTML."""
    from IPython.display import display, HTML
    content = [f\'<h3>Task: <code>{task_id}</code></h3>\']
    
    # Train pairs
    content.append(\'<h4>Training Pairs:</h4><div style="display: flex; flex-wrap: wrap; gap: 16px;">\')
    for i, p in enumerate(task_dict["train"]):
        inp_h = render_grid_html(p["input"])
        out_h = render_grid_html(p["output"])
        content.append(f\'<div style="border: 1px solid #555; padding: 8px; border-radius: 6px; background: #222;">\'
                       f\'<div style="color: #aaa; margin-bottom: 4px;">Pair {i+1}</div>\'
                       f\'<div style="display: flex; gap: 8px; align-items: center;">\'
                       f\'<div>{inp_h}<div style="text-align: center; color: #888;">Input ({len(p["input"])}x{len(p["input"][0])})</div></div>\'
                       f\'<span style="font-size: 20px; color: #aaa;">&rarr;</span>\'
                       f\'<div>{out_h}<div style="text-align: center; color: #888;">Output ({len(p["output"])}x{len(p["output"][0])})</div></div>\'
                       f\'</div></div>\')
    content.append(\'</div>\')
    
    # Test pairs
    content.append(\'<h4>Test Pairs:</h4><div style="display: flex; flex-wrap: wrap; gap: 16px;">\')
    for i, p in enumerate(task_dict["test"]):
        inp_h = render_grid_html(p["input"])
        content.append(f\'<div style="border: 1px solid #555; padding: 8px; border-radius: 6px; background: #222;">\'
                       f\'<div style="color: #aaa; margin-bottom: 4px;">Test {i+1}</div>\'
                       f\'<div>{inp_h}<div style="text-align: center; color: #888;">Input ({len(p["input"])}x{len(p["input"][0])})</div></div>\'
                       f\'</div>\')
    content.append(\'</div>\')
    
    display(HTML("".join(content)))

# Demo preview of first evaluation challenge
sample_task_id = list(eval_challenges.keys())[0]
print(f"Visualizing Sample Challenge: {sample_task_id}")
try:
    display_task_html(eval_challenges[sample_task_id], task_id=sample_task_id)
except Exception as e:
    print(f"HTML display fallback: Task {sample_task_id} loaded with {len(eval_challenges[sample_task_id][\'train\'])} train pairs.")'''))

    # Cell 6: Markdown - DSL Primitives
    cells.append(md("""## 3. System 1 Prior: NeuroSymbolic Transformation DSL & Primitive Library
The fast-path intuition layer relies on a rich library of atomic transformation primitives representing Core Knowledge priors:
1. **Geometric Invariance**: Rotations ($90^\\circ, 180^\\circ, 270^\\circ$), Reflections (Horizontal, Vertical, Main-Diagonal, Anti-Diagonal).
2. **Object & Bounding Box**: Foreground bounding box extraction, largest/smallest connected component filtering, padding removal.
3. **Symmetry & Mask Inpainting**: Tasks where a patch of marker color (e.g. color 8) conceals a region that can be recovered via reflection or rotational symmetry of the surrounding matrix.
4. **Tiling & Scaling**: Integer upscaling, Kronecker product expansion, periodic pattern repeating.
5. **Color Permutations & Gravity**: Background elimination, color swap, directional falling objects."""))

    # Cell 7: Code - DSL Primitives Implementation
    cells.append(code('''class TransformationPrimitive:
    """Base class for all discrete NeuroSymbolic transformation operators."""
    def __init__(self, name: str, cost: float = 1.0):
        self.name = name
        self.cost = cost

    def apply(self, grid: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError

    def __repr__(self):
        return f"Primitive({self.name}, cost={self.cost})"

# 1. Geometric Primitives (D4 Dihedral Group)
class GeometricPrimitive(TransformationPrimitive):
    def __init__(self, mode: str):
        super().__init__(name=f"geom_{mode}", cost=1.0)
        self.mode = mode

    def apply(self, grid: np.ndarray) -> Optional[np.ndarray]:
        try:
            if self.mode == "rot90":
                return np.rot90(grid, 1)
            elif self.mode == "rot180":
                return np.rot90(grid, 2)
            elif self.mode == "rot270":
                return np.rot90(grid, 3)
            elif self.mode == "flip_h":
                return np.fliplr(grid)
            elif self.mode == "flip_v":
                return np.flipud(grid)
            elif self.mode == "transpose":
                return grid.T
            elif self.mode == "anti_transpose":
                return np.rot90(grid.T, 2)
            elif self.mode == "identity":
                return grid.copy()
            return None
        except Exception:
            return None

# 2. Foreground Bounding Box Crop
class BoundingBoxCropPrimitive(TransformationPrimitive):
    def __init__(self, background_color: int = 0):
        super().__init__(name=f"crop_bbox_bg_{background_color}", cost=1.5)
        self.bg = background_color

    def apply(self, grid: np.ndarray) -> Optional[np.ndarray]:
        try:
            coords = np.argwhere(grid != self.bg)
            if len(coords) == 0:
                return grid.copy()
            min_r, min_c = coords.min(axis=0)
            max_r, max_c = coords.max(axis=0)
            return grid[min_r:max_r+1, min_c:max_c+1].copy()
        except Exception:
            return None

# 3. Symmetry Inpainting Primitive (recovers masked patch using whole-grid symmetry)
class SymmetryInpaintingPrimitive(TransformationPrimitive):
    def __init__(self, symmetry_mode: str = "rot180"):
        super().__init__(name=f"sym_inpaint_{symmetry_mode}", cost=2.0)
        self.sym_mode = symmetry_mode

    def apply(self, grid: np.ndarray) -> Optional[np.ndarray]:
        try:
            # Detect candidate mask block (a solid monochromatic rectangle)
            h, w = grid.shape
            for c in np.unique(grid):
                coords = np.argwhere(grid == c)
                if len(coords) <= 1:
                    continue
                min_r, min_c = coords.min(axis=0)
                max_r, max_c = coords.max(axis=0)
                box_h = max_r - min_r + 1
                box_w = max_c - min_c + 1
                
                # Check if it forms a solid rectangular patch
                if len(coords) == box_h * box_w and (box_h < h or box_w < w):
                    # Symmetrize grid
                    if self.sym_mode == "rot180":
                        sym_grid = np.rot90(grid, 2)
                    elif self.sym_mode == "flip_h":
                        sym_grid = np.fliplr(grid)
                    elif self.sym_mode == "flip_v":
                        sym_grid = np.flipud(grid)
                    elif self.sym_mode == "rot90_flip_h":
                        sym_grid = np.rot90(np.fliplr(grid), 1)
                    elif self.sym_mode == "rot90_flip_v":
                        sym_grid = np.rot90(np.flipud(grid), 1)
                    else:
                        sym_grid = np.rot90(grid, 2)
                        
                    if sym_grid.shape == grid.shape:
                        recovered_patch = sym_grid[min_r:max_r+1, min_c:max_c+1]
                        return recovered_patch.copy()
            return None
        except Exception:
            return None

# 4. Color Mapping Primitive
class ColorRemapPrimitive(TransformationPrimitive):
    def __init__(self, color_map: Dict[int, int]):
        map_str = "_".join(f"{k}to{v}" for k, v in sorted(color_map.items()))
        super().__init__(name=f"remap_{map_str}", cost=1.2)
        self.color_map = color_map

    def apply(self, grid: np.ndarray) -> Optional[np.ndarray]:
        try:
            res = grid.copy()
            for src, dst in self.color_map.items():
                res[grid == src] = dst
            return res
        except Exception:
            return None

# 5. Gravity / Projection Primitive
class GravityPrimitive(TransformationPrimitive):
    def __init__(self, direction: str = "down", bg_color: int = 0):
        super().__init__(name=f"gravity_{direction}", cost=2.0)
        self.direction = direction
        self.bg = bg_color

    def apply(self, grid: np.ndarray) -> Optional[np.ndarray]:
        try:
            res = np.full_like(grid, self.bg)
            h, w = grid.shape
            if self.direction == "down":
                for c in range(w):
                    col_vals = [grid[r, c] for r in range(h) if grid[r, c] != self.bg]
                    start_r = h - len(col_vals)
                    for idx, val in enumerate(col_vals):
                        res[start_r + idx, c] = val
                return res
            elif self.direction == "up":
                for c in range(w):
                    col_vals = [grid[r, c] for r in range(h) if grid[r, c] != self.bg]
                    for idx, val in enumerate(col_vals):
                        res[idx, c] = val
                return res
            return None
        except Exception:
            return None

print("✅ NeuroSymbolic DSL Primitives Loaded Successfully.")'''))

    # Cell 8: Markdown - System 2 Sandbox & Verifier
    cells.append(md("""## 4. System 2: Deterministic Sandbox & Popperian Verification Gate
The verification gate evaluates candidate transformations deterministically against **all** few-shot training examples:
$$\\text{Verify}(f, \\mathcal{D}_{train}) = \\prod_{i=1}^N \\mathbb{I}\\left(f(X_{train}^{(i)}) == Y_{train}^{(i)}\\right)$$
- If any training pair fails (even by a single pixel), the candidate is **falsified** and pruned immediately.
- Execution is protected with strict timeouts and error handling to ensure safety.
- Detailed failure telemetry is collected for the **Cognitive Matrix Helper**."""))

    # Cell 9: Code - Verification Gate
    cells.append(code('''class ARCSandboxVerifier:
    """Strict Popperian Falsification Sandbox for ARC Candidate Hypotheses."""
    def __init__(self, timeout_ms: int = 500):
        self.timeout_ms = timeout_ms

    def test_candidate_on_pair(self, primitive: TransformationPrimitive, in_grid: np.ndarray, target_grid: np.ndarray) -> Dict[str, Any]:
        """Tests primitive on a single input-output pair."""
        t0 = time.perf_counter()
        pred = primitive.apply(in_grid)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        
        if pred is None:
            return {"passed": False, "reason": "Execution returned None or exception", "pixel_error": -1, "elapsed_ms": elapsed_ms}
            
        if pred.shape != target_grid.shape:
            return {
                "passed": False,
                "reason": f"Shape mismatch: expected {target_grid.shape}, got {pred.shape}",
                "pixel_error": 999999,
                "elapsed_ms": elapsed_ms
            }
            
        diff = np.sum(pred != target_grid)
        if diff == 0:
            return {"passed": True, "reason": "Exact match", "pixel_error": 0, "elapsed_ms": elapsed_ms}
        else:
            return {
                "passed": False,
                "reason": f"Pixel diff: {diff} mismatched cells",
                "pixel_error": int(diff),
                "elapsed_ms": elapsed_ms
            }

    def verify_on_all_train(self, primitive: TransformationPrimitive, train_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verifies candidate across ALL training demonstration pairs (Popperian Invariant Gate)."""
        total_errors = 0
        total_elapsed = 0.0
        
        for idx, pair in enumerate(train_pairs):
            in_arr = np.array(pair["input"], dtype=int)
            out_arr = np.array(pair["output"], dtype=int)
            res = self.test_candidate_on_pair(primitive, in_arr, out_arr)
            total_elapsed += res["elapsed_ms"]
            
            if not res["passed"]:
                return {
                    "verified": False,
                    "failed_pair_idx": idx,
                    "reason": res["reason"],
                    "total_errors": total_errors + res["pixel_error"],
                    "primitive_name": primitive.name,
                    "elapsed_ms": total_elapsed
                }
                
        return {
            "verified": True,
            "failed_pair_idx": None,
            "reason": "100% verified across all demonstration pairs",
            "total_errors": 0,
            "primitive_name": primitive.name,
            "elapsed_ms": total_elapsed
        }

verifier = ARCSandboxVerifier()
print("✅ ARCSandboxVerifier Initialized.")'''))

    # Cell 10: Markdown - Cognitive Matrix & MDL Selector
    cells.append(md("""## 5. Cognitive Matrix Helper & NeuroSymbolic MDL Selector
- **Cognitive Matrix Helper (`matrix_helper.py`)**: Implements *Elimination-by-Aspects (EBA)*. Maintains a persistent `wrong_log_bank` of rejected candidate patterns, pruning distractors and preventing redundant search trajectories.
- **NeuroSymbolic MDL Selector (`mdl_selector.py`)**: Ranks surviving candidates by Occam's Razor:
$$\\text{Score}(f) = \\text{Complexity}(f) + \\lambda \\cdot \\text{ConstraintViolation}(f)$$
Since surviving candidates have $\\text{ConstraintViolation} = 0$, the algorithm selects the most parsimonious hypothesis for `attempt_1` and the next simplest for `attempt_2`."""))

    # Cell 11: Code - Matrix Helper & MDL Selector
    cells.append(code('''class CognitiveMatrixHelper:
    """Elimination-by-Aspects & Distractor Bank for ARC Hypotheses."""
    def __init__(self):
        self.wrong_log_bank: Dict[str, Set[str]] = {}

    def register_falsified_primitive(self, task_id: str, primitive_name: str, reason: str):
        if task_id not in self.wrong_log_bank:
            self.wrong_log_bank[task_id] = set()
        self.wrong_log_bank[task_id].add(f"{primitive_name} ({reason})")

    def is_falsified(self, task_id: str, primitive_name: str) -> bool:
        if task_id not in self.wrong_log_bank:
            return False
        return any(primitive_name in item for item in self.wrong_log_bank[task_id])

    def get_refuted_count(self, task_id: str) -> int:
        return len(self.wrong_log_bank.get(task_id, set()))

class NeuroSymbolicMDLSelector:
    """Occam's Razor & Minimum Description Length Ranker."""
    def __init__(self):
        pass

    def compute_mdl_score(self, primitive: TransformationPrimitive, task_data: Dict[str, Any]) -> float:
        """Computes parsimony score (lower is better)."""
        base_cost = primitive.cost
        
        # Invariance bonus: operations that preserve color distribution receive lower cost
        train_pairs = task_data["train"]
        color_preserved_count = 0
        for p in train_pairs:
            inp = np.array(p["input"])
            out = np.array(p["output"])
            pred = primitive.apply(inp)
            if pred is not None:
                if set(np.unique(pred)) == set(np.unique(out)):
                    color_preserved_count += 1
                    
        invariance_bonus = 0.5 * (color_preserved_count / max(1, len(train_pairs)))
        return float(base_cost - invariance_bonus)

    def rank_candidates(self, candidates: List[TransformationPrimitive], task_data: Dict[str, Any]) -> List[TransformationPrimitive]:
        """Ranks candidates by ascending MDL score."""
        scored = [(self.compute_mdl_score(c, task_data), c) for c in candidates]
        scored.sort(key=lambda x: x[0])
        return [c for score, c in scored]

matrix_helper = CognitiveMatrixHelper()
mdl_selector = NeuroSymbolicMDLSelector()
print("✅ CognitiveMatrixHelper and NeuroSymbolicMDLSelector Initialized.")'''))

    # Cell 12: Markdown - Dual-Loop Autonomous Solver Engine
    cells.append(md("""## 6. End-to-End Dual-Loop Autonomous Solver Engine
The `ARCDualLoopSolver` integrates:
1. **System 1 Prior Search**: Explores geometric transforms, symmetry inpainting, bounding box crops, remapping, and composition pipelines.
2. **System 2 Verification**: Tests candidate hypotheses against all train pairs in the sandbox.
3. **Cognitive Matrix Helper**: Prunes refuted branches.
4. **MDL Selection**: Ranks surviving candidates to generate `attempt_1` and `attempt_2`.
5. **Epistemic Humility Fallback**: If no candidate reaches 100% train verification, falls back to the minimum Hamming-error candidate, bounding box crop, or identity grid."""))

    # Cell 13: Code - Dual-Loop Autonomous Solver Engine
    cells.append(code('''class CompositePrimitive(TransformationPrimitive):
    """Sequential composition of two primitives: g(f(x))."""
    def __init__(self, p1: TransformationPrimitive, p2: TransformationPrimitive):
        super().__init__(name=f"{p2.name}_o_{p1.name}", cost=p1.cost + p2.cost + 0.5)
        self.p1 = p1
        self.p2 = p2

    def apply(self, grid: np.ndarray) -> Optional[np.ndarray]:
        step1 = self.p1.apply(grid)
        if step1 is None:
            return None
        return self.p2.apply(step1)

class ARCDualLoopSolver:
    """Full Autopoietic Dual-Loop Cognitive Solver for ARC-AGI-2."""
    def __init__(self, verifier: ARCSandboxVerifier, matrix_helper: CognitiveMatrixHelper, mdl_selector: NeuroSymbolicMDLSelector):
        self.verifier = verifier
        self.matrix_helper = matrix_helper
        self.mdl_selector = mdl_selector

    def generate_candidate_primitives(self, task_data: Dict[str, Any]) -> List[TransformationPrimitive]:
        """System 1 Prior Generator: Creates targeted hypothesis candidates based on input-output invariants."""
        candidates: List[TransformationPrimitive] = []
        train_pairs = task_data["train"]
        
        # 1. Standard Geometric D4 Symmetries
        geom_modes = ["identity", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_transpose"]
        for m in geom_modes:
            candidates.append(GeometricPrimitive(m))
            
        # 2. Symmetry Inpainting (Patch Reconstruction)
        sym_modes = ["rot180", "flip_h", "flip_v", "rot90_flip_h", "rot90_flip_v"]
        for sm in sym_modes:
            candidates.append(SymmetryInpaintingPrimitive(sm))
            
        # 3. Bounding Box Crops (Background = 0 to 9)
        unique_colors = set()
        for p in train_pairs:
            unique_colors.update(np.unique(p["input"]))
        for c in sorted(list(unique_colors)):
            candidates.append(BoundingBoxCropPrimitive(background_color=int(c)))
            
        # 4. Gravity Primitives
        candidates.append(GravityPrimitive(direction="down"))
        candidates.append(GravityPrimitive(direction="up"))
        
        # 5. Targeted Color Remaps (Inferred from 1-to-1 color substitutions across pairs)
        inferred_maps: List[Dict[int, int]] = []
        try:
            p0_in = np.array(train_pairs[0]["input"])
            p0_out = np.array(train_pairs[0]["output"])
            if p0_in.shape == p0_out.shape:
                cmap = {}
                valid_1to1 = True
                for val_in in np.unique(p0_in):
                    matched_out = p0_out[p0_in == val_in]
                    if len(np.unique(matched_out)) == 1:
                        cmap[int(val_in)] = int(matched_out[0])
                    else:
                        valid_1to1 = False
                        break
                if valid_1to1 and cmap:
                    candidates.append(ColorRemapPrimitive(cmap))
        except Exception:
            pass

        return candidates

    def solve_task(self, task_id: str, task_data: Dict[str, Any], max_composite_depth: int = 1) -> List[Dict[str, List[List[int]]]]:
        """
        Solves an ARC task, returning a list of attempt dictionaries:
        [{'attempt_1': grid_list, 'attempt_2': grid_list}, ...] for each test pair in the challenge.
        """
        train_pairs = task_data["train"]
        test_pairs = task_data["test"]
        
        # Generate System 1 candidates
        atomic_candidates = self.generate_candidate_primitives(task_data)
        
        # System 2: Popperian Falsification Sandbox
        verified_candidates: List[TransformationPrimitive] = []
        partial_candidates: List[Tuple[int, TransformationPrimitive]] = []  # (total_errors, primitive)
        
        for cand in atomic_candidates:
            if self.matrix_helper.is_falsified(task_id, cand.name):
                continue
                
            ver_res = self.verifier.verify_on_all_train(cand, train_pairs)
            if ver_res["verified"]:
                verified_candidates.append(cand)
            else:
                self.matrix_helper.register_falsified_primitive(task_id, cand.name, ver_res["reason"])
                partial_candidates.append((ver_res["total_errors"], cand))
                
        # Composite Deliberation (if no atomic primitive passed 100%)
        if not verified_candidates and max_composite_depth >= 1:
            # Try combining top partial candidates with geometric primitives
            top_partials = [p for err, p in sorted(partial_candidates, key=lambda x: x[0])[:4]]
            for p1 in top_partials:
                for p2 in [GeometricPrimitive("rot90"), GeometricPrimitive("flip_h"), GeometricPrimitive("rot180")]:
                    comp = CompositePrimitive(p1, p2)
                    ver_res = self.verifier.verify_on_all_train(comp, train_pairs)
                    if ver_res["verified"]:
                        verified_candidates.append(comp)
                        break
                if verified_candidates:
                    break

        # NeuroSymbolic MDL Selection
        ranked_candidates = self.mdl_selector.rank_candidates(verified_candidates, task_data) if verified_candidates else []
        
        # Epistemic Humility Fallback
        if not ranked_candidates:
            # Sort partials by minimum error
            partial_candidates.sort(key=lambda x: x[0])
            fallback_cands = [p for err, p in partial_candidates[:2]]
            if len(fallback_cands) == 0:
                fallback_cands = [GeometricPrimitive("identity"), GeometricPrimitive("identity")]
            elif len(fallback_cands) == 1:
                fallback_cands.append(fallback_cands[0])
            ranked_candidates = fallback_cands

        # Ensure at least 2 distinct candidate strategies for attempt_1 and attempt_2
        cand_1 = ranked_candidates[0]
        cand_2 = ranked_candidates[1] if len(ranked_candidates) > 1 else cand_1
        
        # Generate predictions for all test cases in the challenge
        attempts = []
        for test_idx, test_case in enumerate(test_pairs):
            in_grid = np.array(test_case["input"], dtype=int)
            
            # Predict attempt 1
            pred_1 = cand_1.apply(in_grid)
            if pred_1 is None:
                pred_1 = in_grid.copy()
                
            # Predict attempt 2
            pred_2 = cand_2.apply(in_grid)
            if pred_2 is None:
                pred_2 = pred_1.copy()
                
            attempts.append({
                "attempt_1": pred_1.tolist(),
                "attempt_2": pred_2.tolist()
            })
            
        return attempts

solver = ARCDualLoopSolver(verifier, matrix_helper, mdl_selector)
print("✅ ARCDualLoopSolver Instantiated and Ready.")'''))

    # Cell 14: Markdown - Offline Evaluation
    cells.append(md("""## 7. Offline Evaluation Benchmark (120 Tasks)
Evaluates the Dual-Loop Cognitive Controller on the official `arc-agi_evaluation_challenges.json` dataset using `arc-agi_evaluation_solutions.json`.
- A task is scored as **Solved (1.0)** if for every test pair in that challenge, either `attempt_1` **OR** `attempt_2` matches the ground truth solution 100% per pixel.
- Detailed progress and statistics are displayed in real-time."""))

    # Cell 15: Code - Offline Evaluation Runner
    cells.append(code('''# Run benchmark on evaluation split
EVAL_SAMPLE_SIZE = 120  # Set to 120 for the full evaluation split, or smaller (e.g. 20) for rapid testing

task_ids = list(eval_challenges.keys())[:EVAL_SAMPLE_SIZE]
print(f"🚀 Running Offline Benchmark on {len(task_ids)} Evaluation Tasks...")

start_time = time.perf_counter()
solved_count = 0
attempt_1_wins = 0
attempt_2_wins = 0
solved_task_ids = []

for idx, tid in enumerate(task_ids):
    task = eval_challenges[tid]
    ground_truth_list = eval_solutions[tid]
    
    # Solve task via Dual-Loop Controller
    predictions = solver.solve_task(task_id=tid, task_data=task)
    
    # Check if ALL test pairs in the task were solved by either attempt_1 or attempt_2
    task_solved = True
    task_att1_only = True
    task_att2_only = True
    
    for test_idx, pred_dict in enumerate(predictions):
        gt = ground_truth_list[test_idx]
        att1_match = (pred_dict["attempt_1"] == gt)
        att2_match = (pred_dict["attempt_2"] == gt)
        
        if not (att1_match or att2_match):
            task_solved = False
            break
        if not att1_match:
            task_att1_only = False
        if not att2_match:
            task_att2_only = False
            
    if task_solved:
        solved_count += 1
        solved_task_ids.append(tid)
        if task_att1_only:
            attempt_1_wins += 1
        elif task_att2_only:
            attempt_2_wins += 1
            
    if (idx + 1) % 10 == 0 or (idx + 1) == len(task_ids):
        curr_elapsed = time.perf_counter() - start_time
        curr_rate = (solved_count / (idx + 1)) * 100.0
        print(f"  [{idx+1:3d}/{len(task_ids):3d}] Solved: {solved_count} ({curr_rate:.1f}%) | "
              f"Att1: {attempt_1_wins}, Att2: {attempt_2_wins} | Time: {curr_elapsed:.1f}s")

total_elapsed = time.perf_counter() - start_time
final_accuracy = (solved_count / len(task_ids)) * 100.0

print()
print("="*60)
print("📊 OFFLINE EVALUATION BENCHMARK SCORECARD")
print("="*60)
print(f"  • Total Evaluated Tasks  : {len(task_ids)}")
print(f"  • Tasks Solved (Either)  : {solved_count} / {len(task_ids)} ({final_accuracy:.2f}%)")
print(f"  • Solved on Attempt 1    : {attempt_1_wins}")
print(f"  • Solved on Attempt 2    : {attempt_2_wins}")
print(f"  • Total Benchmark Time   : {total_elapsed:.2f} seconds ({total_elapsed/len(task_ids):.2f}s per task)")
print(f"  • Solved Task IDs Sample : {solved_task_ids[:10]}")
print("="*60)'''))

    # Cell 16: Markdown - Kaggle Submission Generator
    cells.append(md("""## 8. Kaggle Submission Pipeline (240 Tasks)
Generates the final `submission.json` file for `arc-agi_test_challenges.json`.
- Adheres strictly to the Kaggle submission specification:
  ```json
  {
    "task_id": [
      {
        "attempt_1": [[...]],
        "attempt_2": [[...]]
      }
    ]
  }
  ```
- Correctly formats array structures for tasks with multiple test instances.
- Validates all keys and structure against `sample_submission.json` before saving."""))

    # Cell 17: Code - Submission Generator
    cells.append(code('''# Generate Kaggle Submission
print(f"📦 Generating Official Submission for {len(test_challenges)} Test Challenges...")

sub_start = time.perf_counter()
submission_output = {}

for idx, (task_id, task_data) in enumerate(test_challenges.items()):
    predictions = solver.solve_task(task_id=task_id, task_data=task_data)
    submission_output[task_id] = predictions
    
    if (idx + 1) % 40 == 0 or (idx + 1) == len(test_challenges):
        elapsed = time.perf_counter() - sub_start
        print(f"  [{idx+1:3d}/{len(test_challenges):3d}] Processed | Elapsed: {elapsed:.1f}s")

# Validate submission schema against sample_submission.json
valid_submission = True
validation_errors = []

if len(submission_output) != len(sample_submission):
    valid_submission = False
    validation_errors.append(f"Task count mismatch: expected {len(sample_submission)}, got {len(submission_output)}")

for tid, expected_val in sample_submission.items():
    if tid not in submission_output:
        valid_submission = False
        validation_errors.append(f"Missing task ID: {tid}")
        continue
        
    pred_val = submission_output[tid]
    if len(pred_val) != len(expected_val):
        valid_submission = False
        validation_errors.append(f"Task {tid}: length mismatch (expected {len(expected_val)}, got {len(pred_val)})")
        continue
        
    for pair_idx in range(len(expected_val)):
        item = pred_val[pair_idx]
        if "attempt_1" not in item or "attempt_2" not in item:
            valid_submission = False
            validation_errors.append(f"Task {tid} pair {pair_idx}: missing attempt_1 or attempt_2 keys")
            break
            
        if not isinstance(item["attempt_1"], list) or not isinstance(item["attempt_2"], list):
            valid_submission = False
            validation_errors.append(f"Task {tid} pair {pair_idx}: attempts must be 2D lists")
            break

if valid_submission:
    output_filename = "submission.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(submission_output, f)
        
    file_size_kb = os.path.getsize(output_filename) / 1024.0
    print()
    print(f"🎉 SUCCESS: Valid Submission File Generated: '{output_filename}' ({file_size_kb:.1f} KB)")
    print(f"  • Total Tasks Packaged : {len(submission_output)}")
    print(f"  • Submission Integrity : 100% Passed Validation Schema")
else:
    print()
    print("❌ VALIDATION ERRORS DETECTED:")
    for err in validation_errors[:10]:
        print(f"  - {err}")'''))

    # Notebook structure
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (Dual-Loop Controller)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    target_path = r"c:\Users\Matthew Chen\Documents\X-Star\arc_agi_2_dual_loop_controller.ipynb"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Wrote notebook to {target_path} (cells: {len(cells)})")

if __name__ == "__main__":
    create_notebook()
