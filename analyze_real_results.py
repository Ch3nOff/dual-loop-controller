import json
import numpy as np

with open('eval_results/eval_results_1789538565.json', 'r', encoding='utf-8') as f:
    base_data = json.load(f)

with open('eval_results/eval_results_1789538521.json', 'r', encoding='utf-8') as f:
    loop_data = json.load(f)

base_samples = base_data['samples']['arc_easy']
loop_samples = loop_data['samples']['arc_easy']

print(f"Total samples evaluated: {len(base_samples)}")

base_acc = base_data['results']['arc_easy']['acc,none']
loop_acc = loop_data['results']['arc_easy']['acc,none']
base_norm = base_data['results']['arc_easy']['acc_norm,none']
loop_norm = loop_data['results']['arc_easy']['acc_norm,none']

print(f"Base Acc: {base_acc*100:.1f}%, Base Acc_Norm: {base_norm*100:.1f}%")
print(f"Loop Acc: {loop_acc*100:.1f}%, Loop Acc_Norm: {loop_norm*100:.1f}%")

# Let's inspect sample fields
s0 = base_samples[0]
print("Sample keys:", list(s0.keys()))
for k in ['target', 'filtered_resps']:
    if k in s0:
        print(f"  {k}: {s0[k]}")

# Inspect log-likelihood shifts across all 50 questions
ll_deltas = []
logprob_shifts = []
discrepancies = []

for i in range(len(base_samples)):
    b = base_samples[i]
    l = loop_samples[i]
    
    target = b['target']
    
    # lm-eval multiple choice predictions:
    # filtered_resps contains tuples: (logprob, is_greedy)
    b_resps = b.get('filtered_resps', [])
    l_resps = l.get('filtered_resps', [])
    
    b_logprobs = [r[0] for r in b_resps]
    l_logprobs = [r[0] for r in l_resps]
    
    # Average absolute logprob shift across choices
    shift = np.mean([abs(l_lp - b_lp) for b_lp, l_lp in zip(b_logprobs, l_logprobs)])
    logprob_shifts.append(shift)
    
    # Target logprob delta
    target_delta = l_logprobs[target] - b_logprobs[target]
    ll_deltas.append(target_delta)
    
    b_pred = int(np.argmax(b_logprobs))
    l_pred = int(np.argmax(l_logprobs))
    
    if b_pred != l_pred:
        discrepancies.append({
            'doc_id': b['doc_id'],
            'question': b['doc']['question'],
            'choices': b['doc']['choices']['text'],
            'target': target,
            'b_pred': b_pred,
            'l_pred': l_pred,
            'b_logprobs': [round(x, 3) for x in b_logprobs],
            'l_logprobs': [round(x, 3) for x in l_logprobs],
            'b_correct': (b_pred == target),
            'l_correct': (l_pred == target),
        })

print(f"\nDiscrepant predictions count: {len(discrepancies)}")
for d in discrepancies:
    print(f"--- Question ID: {d['doc_id']} ---")
    print(f"Question: {d['question']}")
    for idx, c in enumerate(d['choices']):
        mark = " [TARGET]" if idx == d['target'] else ""
        print(f"  Choice {idx}: {c}{mark}")
    print(f"Base: Pred={d['b_pred']} (Correct={d['b_correct']}), Logprobs={d['b_logprobs']}")
    print(f"Loop: Pred={d['l_pred']} (Correct={d['l_correct']}), Logprobs={d['l_logprobs']}")

print(f"\nLogprob Perturbation Statistics:")
print(f"Mean logprob shift per choice: {np.mean(logprob_shifts):.4f} nats")
print(f"Max logprob shift per choice: {np.max(logprob_shifts):.4f} nats")
print(f"Mean target logprob delta: {np.mean(ll_deltas):+.4f} nats")
