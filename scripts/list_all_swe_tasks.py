import json

with open("competition/tasks.jsonl", "r", encoding="utf-8") as f:
    tasks = [json.loads(line) for line in f if line.strip()]

repos = {}
for t in tasks:
    r = t.get("repo", "unknown")
    if r not in repos:
        repos[r] = []
    repos[r].append(t)

selected = []
# Pick 8 from fastapi
selected.extend(repos.get("fastapi/fastapi", [])[:8])
# Pick 6 from rich
selected.extend(repos.get("Textualize/rich", [])[:6])
# Pick 5 from requests
selected.extend(repos.get("psf/requests", [])[:5])
# Pick 1 from httpx
selected.extend(repos.get("encode/httpx", [])[:1])

print(f"Total Selected SWE Tasks: {len(selected)}")
for i, t in enumerate(selected, 1):
    first_line = t["problem_statement"].strip().splitlines()[0]
    # Clean emojis
    summary = first_line[:75]
    patch_lines = len(t.get("patch", "").splitlines())
    test_lines = len(t.get("test_patch", "").splitlines())
    print(f"{i:02d}. [{t['instance_id']:<15}] ({t['repo']:<18}): {summary} (Patch: {patch_lines}L, Tests: {test_lines}L)")
