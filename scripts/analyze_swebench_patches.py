import json, re, sys
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

tasks = [json.loads(line) for line in open('competition/tasks.jsonl', 'r', encoding='utf-8') if line.strip()]

repo_files = defaultdict(lambda: defaultdict(int))
repo_churn = defaultdict(list)
repo_test_types = defaultdict(lambda: defaultdict(int))
single_file_count = 0
docs_src_count = 0

for t in tasks:
    repo = t['repo']
    patch = t.get('patch', '')
    test_patch = t.get('test_patch', '')
    
    files = re.findall(r'--- a/(.*?)\n', patch)
    u_files = set(files)
    if len(u_files) == 1:
        single_file_count += 1
    if any(f.startswith("docs_src/") for f in u_files):
        docs_src_count += 1
        
    for f in u_files:
        repo_files[repo][f] += 1
    
    added = len(re.findall(r'^\+[^+]', patch, re.M))
    removed = len(re.findall(r'^-[^-]', patch, re.M))
    repo_churn[repo].append(added + removed)
    
    if 'def test_' in test_patch:
        repo_test_types[repo]['new_test_func'] += 1
    if 'pytest.raises' in test_patch:
        repo_test_types[repo]['raises_exception'] += 1

print(f"Total tasks: {len(tasks)}")
print(f"Single file fixes: {single_file_count}/{len(tasks)} ({single_file_count/len(tasks):.1%})")
print(f"Docs_src fixes: {docs_src_count}/{len(tasks)} ({docs_src_count/len(tasks):.1%})")

print("\n=== Top Files Per Repo ===")
for repo, fdict in repo_files.items():
    print(f"\n-- {repo} (tasks: {len(repo_churn[repo])}) --")
    sorted_f = sorted(fdict.items(), key=lambda x: -x[1])[:10]
    for f, count in sorted_f:
        print(f"   {count:2d} ({count/len(repo_churn[repo]):.1%}): {f}")

print("\n=== Test Characteristics ===")
for repo, tdict in repo_test_types.items():
    n = len(repo_churn[repo])
    print(f"{repo}: new test func: {tdict['new_test_func']}/{n} ({tdict['new_test_func']/n:.1%}), raises: {tdict['raises_exception']}/{n} ({tdict['raises_exception']/n:.1%})")
