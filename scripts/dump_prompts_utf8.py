import sys

def main():
    with open('scripts/prompt_diffs_utf8.txt', 'w', encoding='utf-8') as out:
        def view(title, path):
            out.write(f"\n{'='*70}\n{title}\n{'='*70}\n")
            with open(path, 'r', encoding='utf-8') as f:
                out.write(f.read() + "\n")

        view("TOP 1 SYSTEM PROMPT", "scripts/competitor_dumps/top1_cell_38.py")
        view("BLACK CAT SYSTEM PROMPT", "scripts/competitor_dumps/black_cat_anchor/prompts/system.md")
        view("XALPHA SYSTEM PROMPT", "gemma4_xalpha_agent/prompts/system.md")
        view("BLACK CAT ANALYZER", "scripts/competitor_dumps/black_cat_anchor/prompts/analyzer.md")
        view("XALPHA ANALYZER", "gemma4_xalpha_agent/prompts/analyzer.md")

if __name__ == '__main__':
    main()
