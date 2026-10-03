import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Load Top 1
with open('scripts/competitor_dumps/top1_cell_38.py', 'r', encoding='utf-8') as f:
    top1_code = f.read()

# Load Pathfinder extracted
with open('scripts/pathfinder_extracted.txt', 'r', encoding='utf-8') as f:
    pathfinder_all = f.read()

# Find Cell 29 in Pathfinder
c29_idx = pathfinder_all.find("CELL 29")
c30_idx = pathfinder_all.find("CELL 30")
c29_text = pathfinder_all[c29_idx:c30_idx]

print("Top 1 prompt extract:")
for line in top1_code.splitlines()[22:50]:
    print("  ", line)

print("\nPathfinder prompt extract:")
for line in c29_text.splitlines()[8:45]:
    print("  ", line)
