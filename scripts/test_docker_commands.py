#!/usr/bin/env python3
"""Test every command declared in Agent X-Alpha system prompt inside Docker container."""

import subprocess
import sys

def run_in_docker(cmd_str: str) -> subprocess.CompletedProcess:
    full_cmd = [
        "docker", "run", "--rm",
        "swebench-sandbox:latest",
        "/bin/bash", "-c", cmd_str
    ]
    return subprocess.run(full_cmd, capture_output=True, text=True)

commands = [
    # 1. Basic tool availability
    ("git --version", "git version"),
    ("grep --version", "grep (GNU grep)"),
    ("find --version", "find (GNU findutils)"),
    ("sed --version", "sed (GNU sed)"),
    ("awk -W version", "mawk"),
    ("python3 --version", "Python 3.13"),
    ("pytest --version", "pytest"),
    
    # 2. Search commands
    ("cd /workspace && git init -q && touch test.py && git add test.py && git grep -n -F 'test' -- '*.py' || echo 'EMPTY_MATCH'", "EMPTY_MATCH"),
    ("cd /workspace && echo 'hello world' > foo.py && grep -rn 'hello' --include=*.py .", "foo.py:1:hello world"),
    ("cd /workspace && echo -e 'line1\\nline2\\nline3' > bar.py && sed -n '2,3p' bar.py", "line2\nline3"),
    
    # 3. Compilation check
    ("cd /workspace && echo 'x = 1 + 2' > valid.py && python3 -m py_compile valid.py", ""),
    
    # 4. Pytest flags: -q -x -p no:anyio -o timeout=0
    ("cd /workspace && echo 'def test_pass(): assert 1 == 1' > test_sample.py && python3 -m pytest test_sample.py -q -x -p no:anyio -o timeout=0 > /tmp/t.log 2>&1; tail -n 25 /tmp/t.log", "1 passed"),
    
    # 5. Git status & diff pipes
    ("cd /workspace && git init -q && git status --short", ""),
    ("cd /workspace && git init -q && git diff | head -150", ""),
]

print("=" * 70)
print("TESTING AGENT COMMANDS IN SWEBENCH-SANDBOX DOCKER CONTAINER")
print("=" * 70)

all_passed = True
for cmd, expected in commands:
    res = run_in_docker(cmd)
    if res.returncode == 0 and (expected in res.stdout or expected in res.stderr or expected == ""):
        print(f" [PASS] Command: {cmd[:55]}...")
    else:
        print(f" [FAIL] Command: {cmd}")
        print(f"   Exit code: {res.returncode}")
        print(f"   Stdout: {res.stdout.strip()}")
        print(f"   Stderr: {res.stderr.strip()}")
        all_passed = False

if all_passed:
    print("\n[SUCCESS] ALL COMMANDS VERIFIED AND WORKING FLAWLESSLY IN DOCKER!")
else:
    print("\n[FAIL] SOME COMMANDS FAILED IN DOCKER!")
    sys.exit(1)
