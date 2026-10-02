import json, re, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('C:/Users/Matthew Chen/Downloads/gemma-eda-baseline-for-a-start-lb-top-1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'package(BUNDLE' in src or 'validate_bundle' in src:
        print(f"Cell {i}:")
        print(src)
