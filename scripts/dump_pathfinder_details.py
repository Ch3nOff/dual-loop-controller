import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

nb_path = Path(r"C:\Users\Matthew Chen\Downloads\pathfinder-gemma-4-agent-eda-baseline.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

def dump_cells(cell_indices, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for idx in cell_indices:
            cell = nb['cells'][idx]
            ctype = cell.get('cell_type')
            src = ''.join(cell.get('source', []))
            f.write(f"\n{'='*80}\nCELL {idx} [{ctype}]\n{'='*80}\n")
            f.write(src)
            f.write("\n")

dump_cells([1, 2, 3, 5, 10, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30, 34], "scripts/pathfinder_extracted.txt")
print("Dumped pathfinder details to scripts/pathfinder_extracted.txt")
