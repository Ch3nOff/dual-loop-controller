import json
from pathlib import Path

agent_dir = Path("gemma4_xalpha_agent")
agent_files = {}

for p in sorted(agent_dir.rglob("*")):
    if p.is_file():
        rel = p.relative_to(agent_dir).as_posix()
        agent_files[rel] = p.read_text(encoding="utf-8")

print(f"Loaded {len(agent_files)} files from {agent_dir}")

with open("getting_started_gemma_4_hadl_dualloop.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update cell 4 code
cell_4_code = f"""import base64
import os
import shutil
from pathlib import Path

# Path resolution with automatic Kaggle / local environment detection
if Path('/kaggle/input/competitions/gemma-4-developer-agent').exists():
    DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
elif Path('competition').exists():
    DATA_DIR = Path('competition')
else:
    DATA_DIR = Path('.')

WORKING_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
AGENT_DIR = WORKING_DIR / 'gemma4_xalpha_agent'

# Clean previous agent hierarchy
if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR, ignore_errors=True)
AGENT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical 6-file clean bundle (Zero toxic LoRA, Dual-Loop code_analyzer subagent, Bayesian Defect Atlas)
AGENT_FILES = {json.dumps(agent_files, indent=2)}
BINARY_FILES = {{}}

# Materialize the full ADK Dual-Loop enterprise hierarchy
for rel_path, content in sorted(AGENT_FILES.items()):
    target_file = AGENT_DIR / rel_path
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(content, encoding='utf-8')

print(f'[+] Materialized Agent X-Alpha Dual-Loop hierarchy in: {{AGENT_DIR}}')
print(f'    Text files written: {{len(AGENT_FILES)}}')
for f in sorted(AGENT_FILES.keys()):
    print(f'     - {{f}}')

# Load tasks from dataset
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
try:
    from swegemma.models import load_tasks
    tasks = load_tasks(TASKS_PATH)
except (ImportError, Exception):
    import json
    from types import SimpleNamespace
    tasks = []
    if TASKS_PATH.exists():
        with open(TASKS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    tasks.append(SimpleNamespace(
                        instance_id=d.get('instance_id'),
                        repo=d.get('repo'),
                        base_commit=d.get('base_commit'),
                        problem_statement=d.get('problem_statement')
                    ))

print(f'Data directory: {{DATA_DIR}}')
print(f'Loaded {{len(tasks)}} benchmark tasks from {{TASKS_PATH.name}}')
for t in tasks[:5]:
    print(f'  - {{t.instance_id}} ({{t.repo}} @ {{t.base_commit[:8]}})')
"""

nb['cells'][4]['source'] = [line + '\n' for line in cell_4_code.splitlines()]

with open("getting_started_gemma_4_hadl_dualloop.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[SUCCESS] Updated getting_started_gemma_4_hadl_dualloop.ipynb successfully!")
