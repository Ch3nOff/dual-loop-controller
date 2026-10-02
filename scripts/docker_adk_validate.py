#!/usr/bin/env python3
"""Run full ADK submission validation inside the official swebench-sandbox Docker container."""

import subprocess
import sys
from pathlib import Path

CONTAINER_SCRIPT = r'''
import os
import sys
import re
import json
import zipfile
import subprocess
from pathlib import Path

print("=== INSIDE DOCKER (swebench-sandbox:latest) ===")
print("Python version:", sys.version)

zip_path = Path("/submission/submission.zip")
extract_dir = Path("/workspace/agent_test")
extract_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(extract_dir)

files = sorted(p for p in extract_dir.rglob("*") if p.is_file())
print(f"[+] Extracted {len(files)} files into /workspace/agent_test:")
for f in files:
    rel = f.relative_to(extract_dir).as_posix()
    print(f"    - {rel} ({f.stat().st_size:,} bytes)")

# 1. Check root agent.yaml
agent_yaml = extract_dir / "agent.yaml"
assert agent_yaml.exists(), "agent.yaml does not exist!"
agent_text = agent_yaml.read_text(encoding="utf-8")
assert "model: gemma-4-31b-it-qat-w4a16-ct" in agent_text
assert "sub_agents/code_analyzer.yaml" in agent_text
print("[+] agent.yaml verified: clean model, valid sub-agent wiring.")

# 2. Check no toxic adapters
assert "adapter:" not in agent_text, "Illegal adapter in agent.yaml!"
assert not (extract_dir / "adapters").exists(), "adapters directory must not exist!"
print("[+] Verified: ZERO unvalidated adapters in agent bundle.")

# 3. Check code_analyzer subagent
analyzer_yaml = extract_dir / "sub_agents" / "code_analyzer.yaml"
assert analyzer_yaml.exists(), "sub_agents/code_analyzer.yaml does not exist!"
analyzer_text = analyzer_yaml.read_text(encoding="utf-8")
assert "model: gemma-4-31b-it-qat-w4a16-ct" in analyzer_text
assert "tools:" in analyzer_text
print("[+] sub_agents/code_analyzer.yaml verified.")

# 4. Check sampling config
sampling_yaml = extract_dir / "configs" / "sampling.yaml"
assert sampling_yaml.exists(), "configs/sampling.yaml does not exist!"
sampling_text = sampling_yaml.read_text(encoding="utf-8")
assert "include_thoughts: false" in sampling_text, "include_thoughts must be false!"
print("[+] configs/sampling.yaml verified (include_thoughts: false).")

# 5. Check eval_config.yaml
eval_yaml = extract_dir / "eval_config.yaml"
assert eval_yaml.exists(), "eval_config.yaml does not exist!"
eval_text = eval_yaml.read_text(encoding="utf-8")
assert "max_turns: 100" in eval_text
assert "timeout_seconds: 180" in eval_text
print("[+] eval_config.yaml verified (100 turns, 180s timeout).")

# 6. Check prompts and template safety
system_md = (extract_dir / "prompts" / "system.md").read_text(encoding="utf-8")
analyzer_md = (extract_dir / "prompts" / "analyzer.md").read_text(encoding="utf-8")

# Strict brace check
for match in re.finditer(r'(?<![\$\{\\]){+[^{}]*}+', system_md):
    var = match.group().strip('{}').strip().removesuffix('?')
    assert False, f"Illegal brace in system.md: {match.group()}"

for match in re.finditer(r'(?<![\$\{\\]){+[^{}]*}+', analyzer_md):
    var = match.group().strip('{}').strip().removesuffix('?')
    assert var == 'problem_description', f"Illegal brace in analyzer.md: {match.group()}"

print("[+] Prompts verified: 0 illegal curly braces in system.md, valid {problem_description?} in analyzer.md.")

# 7. Check python compilation of all candidate code patterns
print("[+] Testing Python compiler inside Docker sandbox...")
p1 = subprocess.run(["python3", "-m", "py_compile", str(extract_dir / "prompts" / "system.md")], capture_output=True)
# Expected non-zero because system.md is markdown, not python

# Test python syntax check on valid code
code_test = "import os, sys\ndef add(a: int, b: int) -> int:\n    return a + b\n"
(extract_dir / "test_comp.py").write_text(code_test, encoding="utf-8")
res_comp = subprocess.run(["python3", "-m", "py_compile", str(extract_dir / "test_comp.py")], capture_output=True)
assert res_comp.returncode == 0, "Python compilation failed inside sandbox!"
print("[+] python3 -m py_compile works inside Docker sandbox.")

# Test pytest execution inside Docker sandbox
test_code = "from test_comp import add\ndef test_add(): assert add(2, 3) == 5\n"
(extract_dir / "test_add.py").write_text(test_code, encoding="utf-8")
res_pytest = subprocess.run([
    "python3", "-m", "pytest", str(extract_dir / "test_add.py"),
    "-q", "-x", "-p", "no:anyio", "-o", "timeout=0"
], capture_output=True, text=True, cwd=str(extract_dir))
assert res_pytest.returncode == 0, f"Pytest failed: {res_pytest.stdout} {res_pytest.stderr}"
print("[+] python3 -m pytest with all agent flags works inside Docker sandbox: 1 passed.")

print("=== DOCKER VERIFICATION 100% SUCCESSFUL ===")
'''

# Write container script to a temporary runner
Path("scripts/docker_test_runner.py").write_text(CONTAINER_SCRIPT, encoding="utf-8")

cmd = [
    "docker", "run", "--rm",
    "-v", f"{Path.cwd()}:/submission:ro",
    "-v", f"{Path.cwd()}/scripts/docker_test_runner.py:/runner.py:ro",
    "swebench-sandbox:latest",
    "python3", "/runner.py"
]

print("Executing ADK Docker container test...")
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    sys.exit(res.returncode)
