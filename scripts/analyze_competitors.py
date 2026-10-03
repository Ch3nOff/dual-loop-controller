import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def dump_notebook(path):
    print(f"==================================================")
    print(f"FILE: {path}")
    print(f"==================================================")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return

    print(f"Total cells: {len(nb.get('cells', []))}")
    for i, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        source = "".join(cell.get('source', []))
        if cell_type == 'code':
            print(f"\n--- CODE CELL {i} ---")
            lines = source.splitlines()
            # print up to 35 lines
            for line in lines[:35]:
                print(line)
            if len(lines) > 35:
                print(f"... ({len(lines) - 35} more lines) ...")

if __name__ == '__main__':
    from pathlib import Path
    dl_dir = Path.home() / "Downloads"
    f_top1 = dl_dir / "gemma-eda-baseline-for-a-start-lb-top-1.ipynb"
    f_bc = dl_dir / "black-cat-swe-agent-pack-instinct.ipynb"

    with open('scripts/competitor_analysis.txt', 'w', encoding='utf-8') as out:
        sys.stdout = out
        if f_top1.exists():
            dump_notebook(str(f_top1))
        if f_bc.exists():
            dump_notebook(str(f_bc))
