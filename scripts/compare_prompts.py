import sys
sys.stdout.reconfigure(encoding='utf-8')

def view(title, path):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    with open(path, 'r', encoding='utf-8') as f:
        print(f.read())

view("TOP 1 SYSTEM PROMPT", "scripts/competitor_dumps/top1_cell_38.py")
view("BLACK CAT SYSTEM PROMPT", "scripts/competitor_dumps/black_cat_anchor/prompts/system.md")
view("XALPHA SYSTEM PROMPT", "gemma4_xalpha_agent/prompts/system.md")
