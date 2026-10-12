import json
from pathlib import Path

def inspect(path):
    print(f"=== {path.name} ===")
    nb = json.loads(path.read_text(encoding="utf-8"))
    print(f"Total cells: {len(nb['cells'])}")
    for i, c in enumerate(nb['cells']):
        src = c.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        print(f"--- Cell {i} ({c.get('cell_type')}) ---")
        print(src[:400])
        print("...\n")

inspect(Path(r"C:\Users\Matthew Chen\Downloads\gemma-4-agent-budget-fit-single-agent.ipynb"))
inspect(Path(r"C:\Users\Matthew Chen\Downloads\gemma-4-measure-before-you-tune.ipynb"))
