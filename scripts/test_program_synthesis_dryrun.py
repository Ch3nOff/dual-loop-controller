import json
import re
import ast
import traceback
from pathlib import Path
import xstar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "dist" / "xstar_2b_autonomous_brain"
CHALLENGES_FILE = PROJECT_ROOT / "datasets" / "arc_prize_2026" / "arc-agi_evaluation_challenges.json"

print(f"[*] Loading model from {MODEL_DIR}")
model = xstar.load(str(MODEL_DIR))

challenges = json.load(open(CHALLENGES_FILE, "r", encoding="utf-8"))
task_id = list(challenges.keys())[1]  # 135a2760
task = challenges[task_id]

prompt = (
    "<|im_start|>system\n"
    "You are a master Python algorithmic solver for the ARC-AGI prize. "
    "Analyze the demonstrations. Determine the spatial transformation rule. "
    "Write a concise Python function `def transform(grid: list[list[int]]) -> list[list[int]]` "
    "that takes the input 2D list and returns the output 2D list. "
    "Enclose your Python function strictly in ```python\n...\n```.<|im_end|>\n"
    f"<|im_start|>user\n"
    f"Task: {task_id}\n\n"
)
for i, p in enumerate(task["train"]):
    prompt += f"Demo {i+1} Input ({len(p['input'])}x{len(p['input'][0])}):\n{p['input']}\n"
    prompt += f"Demo {i+1} Output ({len(p['output'])}x{len(p['output'][0])}):\n{p['output']}\n\n"

prompt += "Write the Python function `def transform(grid)` to solve this transformation.<|im_end|>\n<|im_start|>assistant\n"

print("[*] Generating Python solver with max_new_tokens=600...")
out = model.generate(prompt, max_new_tokens=600, temperature=0.1, strip_thinking=False)
print("=== MODEL OUTPUT (FIRST 1200 CHARS) ===")
print(out[:1200])
print("\n=== EXTRACTING PYTHON CODE ===")
code_blocks = re.findall(r"```python\s*([\s\S]*?)```", out)
if code_blocks:
    code = code_blocks[-1]
    print("Found code block:\n", code)

    # SEC-07: Static AST verification before execution
    def validate_code_ast(code_str: str) -> tuple[bool, str]:
        if "__" in code_str:
            return False, "Dunder attributes or special methods forbidden"
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        safe_modules = {"numpy", "np", "copy", "math", "re", "collections", "itertools", "functools", "typing"}
        forbidden_calls = {"eval", "exec", "compile", "open", "input", "breakpoint", "help", "quit", "exit"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in safe_modules:
                        return False, f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in safe_modules:
                    return False, f"Forbidden import from: {node.module}"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                    return False, f"Forbidden call: {node.func.id}"
            elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                return False, f"Private attribute access forbidden: {node.attr}"
            elif isinstance(node, ast.Name) and node.id.startswith("_"):
                return False, f"Private identifier forbidden: {node.id}"
        return True, ""

    valid, err_msg = validate_code_ast(code)
    if not valid:
        print(f"[-] Execution rejected by AST sandbox validator: {err_msg}")
    else:
        # Test execution in sandbox
        try:
            def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
                safe_modules = {"numpy", "np", "copy", "math", "re", "collections", "itertools", "functools", "typing"}
                if name.split(".")[0] not in safe_modules:
                    raise ImportError(f"Import of {name} is prohibited in sandbox")
                return __import__(name, globals, locals, fromlist, level)

            safe_builtins = {
                "__import__": _safe_import,
                "abs": abs, "min": min, "max": max, "sum": sum, "all": all, "any": any,
                "len": len, "range": range, "bool": bool, "int": int, "float": float,
                "str": str, "dict": dict, "list": list, "set": set, "tuple": tuple,
                "enumerate": enumerate, "zip": zip, "reversed": reversed, "sorted": sorted,
                "map": map, "filter": filter, "round": round, "pow": pow, "divmod": divmod,
                "isinstance": isinstance, "issubclass": issubclass,
                "AssertionError": AssertionError, "ValueError": ValueError,
                "TypeError": TypeError, "ZeroDivisionError": ZeroDivisionError,
                "IndexError": IndexError, "KeyError": KeyError, "AttributeError": AttributeError,
                "True": True, "False": False, "None": None,
            }
            import numpy as np
            import copy
            env = {
                "__builtins__": safe_builtins,
                "np": np,
                "numpy": np,
                "copy": copy,
            }
            exec(code, env)
            if "transform" in env:
                fn = env["transform"]
                t0_in = task["train"][0]["input"]
                res = fn(t0_in)
                print(f"[+] Execution SUCCESS! Output shape: {len(res)}x{len(res[0])}")
                print(f"    Matches Demo 1 Output? {res == task['train'][0]['output']}")
            else:
                print("[-] 'transform' function not defined in scope.")
        except Exception as e:
            print(f"[-] Execution error: {e}")
            traceback.print_exc()
else:
    print("[-] No python code block found.")
