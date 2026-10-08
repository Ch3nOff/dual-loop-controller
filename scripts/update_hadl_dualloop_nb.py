import json
from pathlib import Path

nb_path = Path("getting_started_gemma_4_hadl_dualloop.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

files = {
    "agent.yaml": Path("gemma4_xalpha_agent/agent.yaml").read_text(encoding="utf-8"),
    "configs/sampling.yaml": Path("gemma4_xalpha_agent/configs/sampling.yaml").read_text(encoding="utf-8"),
    "eval_config.yaml": Path("gemma4_xalpha_agent/eval_config.yaml").read_text(encoding="utf-8"),
    "prompts/analyzer.md": Path("gemma4_xalpha_agent/prompts/analyzer.md").read_text(encoding="utf-8"),
    "prompts/system.md": Path("gemma4_xalpha_agent/prompts/system.md").read_text(encoding="utf-8"),
    "sub_agents/code_analyzer.yaml": Path("gemma4_xalpha_agent/sub_agents/code_analyzer.yaml").read_text(encoding="utf-8"),
}

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("AGENT_FILES =" in line for line in cell["source"]):
        new_source = []
        new_source.append("# Clean and create agent directory\n")
        new_source.append("if AGENT_DIR.exists():\n")
        new_source.append("    shutil.rmtree(AGENT_DIR, ignore_errors=True)\n")
        new_source.append("AGENT_DIR.mkdir(parents=True, exist_ok=True)\n\n")
        new_source.append("# Canonical 6-file Dual-Loop Upper Router Championship Bundle\n")
        new_source.append("AGENT_FILES = {\n")
        for k, v in sorted(files.items()):
            escaped = v.replace("\\", "\\\\")
            new_source.append(f'  "{k}": """{escaped}""",\n')
        new_source.append("}\n\n")
        new_source.append("# Write all agent files\n")
        new_source.append("for rel_path, content in sorted(AGENT_FILES.items()):\n")
        new_source.append("    target_file = AGENT_DIR / rel_path\n")
        new_source.append("    target_file.parent.mkdir(parents=True, exist_ok=True)\n")
        new_source.append("    target_file.write_text(content.strip() + '\\n', encoding='utf-8')\n\n")
        new_source.append("print(f'[+] Materialized Agent X-Alpha Dual-Loop hierarchy in: {AGENT_DIR}')\n")
        new_source.append("print(f'    Total files written: {len(AGENT_FILES)}')\n")
        new_source.append("for f in sorted(AGENT_FILES.keys()):\n")
        new_source.append("    print(f'     - {f}')\n\n")
        new_source.append("# Safe tasks loading (handles hidden dataset variations without crashing)\n")
        new_source.append("TASKS_PATH = DATA_DIR / 'tasks.jsonl'\n")
        new_source.append("tasks = []\n")
        new_source.append("if TASKS_PATH.exists():\n")
        new_source.append("    try:\n")
        new_source.append("        from swegemma.models import load_tasks\n")
        new_source.append("        tasks = load_tasks(TASKS_PATH)\n")
        new_source.append("    except Exception:\n")
        new_source.append("        with open(TASKS_PATH, 'r', encoding='utf-8') as f:\n")
        new_source.append("            for line in f:\n")
        new_source.append("                if line.strip():\n")
        new_source.append("                    d = json.loads(line)\n")
        new_source.append("                    tasks.append(type('Task', (), d))\n")
        new_source.append("print(f'[+] Loaded {len(tasks)} benchmark tasks from {TASKS_PATH.name}')\n")
        cell["source"] = new_source
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully!")
