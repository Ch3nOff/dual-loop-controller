import json

with open('getting_started_gemma_4_hadl_dualloop.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'gemma4_xalpha_agent' in src or 'submission.zip' in src or 'adapter' in src:
        print(f"Cell {i} ({cell.get('cell_type')}):")
        print(src[:300])
        print("...")
