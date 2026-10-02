import json

with open('getting_started_gemma_4_hadl_dualloop.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scripts/getting_started_cells.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        ctype = cell.get('cell_type')
        src = ''.join(cell.get('source', []))
        out.write(f"\n{'='*60}\n")
        out.write(f"CELL {i} ({ctype})\n")
        out.write(f"{'='*60}\n")
        out.write(src)
        out.write("\n")
print(f"Dumped {len(nb['cells'])} cells.")
