import json
from pathlib import Path

# Load agent files
agent_dir = Path("gemma4_xalpha_agent")
agent_files = {}
for p in sorted(agent_dir.rglob("*")):
    if p.is_file():
        rel = p.relative_to(agent_dir).as_posix()
        agent_files[rel] = p.read_text(encoding="utf-8")

print(f"Loaded {len(agent_files)} agent files:")
for k in agent_files:
    print(f"  - {k} ({len(agent_files[k])} chars)")
