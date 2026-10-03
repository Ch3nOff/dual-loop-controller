import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

nb_path = Path(r"C:\Users\Matthew Chen\Downloads\pathfinder-gemma-4-agent-eda-baseline.ipynb")
if not nb_path.exists():
    print(f"File not found: {nb_path}")
    sys.exit(1)

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell.get('cell_type')
    src = ''.join(cell.get('source', []))
    first_line = src.split('\n')[0] if src else ''
    print(f"Cell {i:2d} [{ctype:4s}]: {first_line[:80]} (length: {len(src)})")
