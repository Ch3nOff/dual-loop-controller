import json, re, sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

target_file = Path.home() / "Downloads" / "gemma-eda-baseline-for-a-start-lb-top-1.ipynb"
if not target_file.exists():
    print(f"File not found: {target_file}")
    sys.exit(0)

with open(target_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'package(BUNDLE' in src or 'validate_bundle' in src:
        print(f"Cell {i}:")
        print(src)
