import json

with open("competition/tasks.jsonl", "r", encoding="utf-8") as f:
    tasks = [json.loads(line) for line in f if line.strip()]

print(f"Total tasks: {len(tasks)}")
repo_counts = {}
for t in tasks:
    r = t.get("repo", "unknown")
    repo_counts[r] = repo_counts.get(r, 0) + 1

for r, c in repo_counts.items():
    print(f"  {r}: {c} tasks")

# Show sample 5 tasks
print("\nSample Tasks:")
for i, t in enumerate(tasks[:5]):
    print(f"{i+1}. [{t['instance_id']}] ({t['repo']}): {t['problem_statement'].splitlines()[0][:80]}")
