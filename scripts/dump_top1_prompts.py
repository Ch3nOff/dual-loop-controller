import sys
import ast
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

dump_file = Path('scripts/competitor_dumps/top1_cell_38.py')
if not dump_file.exists():
    print(f"Notice: Competitor dump file not found: {dump_file}")
    sys.exit(0)

with open(dump_file, 'r', encoding='utf-8') as f:
    source_code = f.read()

# SEC-06: Parse AST statically to extract prompt strings without arbitrary code execution via exec()
tree = ast.parse(source_code)

def extract_fstring_template(func_node):
    for stmt in func_node.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.JoinedStr):
            parts = []
            for part in stmt.value.values:
                if isinstance(part, ast.Constant):
                    parts.append(str(part.value))
                elif isinstance(part, ast.FormattedValue):
                    if isinstance(part.value, ast.Name):
                        parts.append(f"[{part.value.id}]")
                    else:
                        parts.append("[dynamic_content]")
            return "".join(parts)
        elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant):
            return str(stmt.value.value)
    return ""

sys_prompt = ""
ana_prompt = ""

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        if node.name == 'build_system_prompt':
            sys_prompt = extract_fstring_template(node)
        elif node.name == 'build_analyzer_prompt':
            ana_prompt = extract_fstring_template(node)

print("=== TOP 1 SYSTEM PROMPT (STATIC AST EXTRACTION) ===")
print(sys_prompt.strip())
print("\n=== TOP 1 ANALYZER PROMPT (STATIC AST EXTRACTION) ===")
print(ana_prompt.strip())
