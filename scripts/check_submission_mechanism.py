import json, sys
sys.stdout.reconfigure(encoding='utf-8')

def check(path):
    print(f"=== {path} ===")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        if any(w in src for w in ['submission.zip', 'vllm', 'Evaluator', 'evaluate_task']):
            lines = [l for l in src.splitlines() if any(w in l for w in ['submission.zip', 'vllm', 'Evaluator', 'evaluate_task'])]
            print(f"Cell {i} ({c.get('cell_type')}):")
            for l in lines[:5]:
                print("  ", l)

check('C:/Users/Matthew Chen/Downloads/gemma-eda-baseline-for-a-start-lb-top-1.ipynb')
check('C:/Users/Matthew Chen/Downloads/black-cat-swe-agent-pack-instinct.ipynb')
