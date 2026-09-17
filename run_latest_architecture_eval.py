import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dual_loop import attach_dual_loop_to_qwen

MODEL_ID = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
ADAPTER_PATH = 'dual_loop/checkpoints/qwen35_2b_deliberation_adapter.pt'
OUTPUT_DIR = 'eval_results'
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'qwen35_2b_latest_architecture_eval.json')
SAMPLES_PER_TASK = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 80)
print(' RUNNING AUTHENTIC LATEST ARCHITECTURE EVALUATION ON QWEN3.5-2B')
print(f' Samples: {SAMPLES_PER_TASK} per task | Total: {SAMPLES_PER_TASK * 4} samples')
print('=' * 80)
sys.stdout.flush()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, revision=REVISION)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print('[1/3] Loading Qwen3.5-2B backbone...')
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    trust_remote_code=True,
    revision=REVISION,
    device_map='cpu'
)

print('[2/3] Attaching Dual-Loop Adapter with Latest Architecture v2.0...')
wrapped_model = attach_dual_loop_to_qwen(
    base_model,
    layer_idx=11,
    k_steps=2,
    use_surprise_gate=True,
    use_contrastive_evidence=True,
    use_hypothesis_verification=True
)
wrapped_model.load_adapter(ADAPTER_PATH, strict=False)
embed_layer = base_model.get_input_embeddings()

tasks = [
    {
        'name': 'ARC-Easy',
        'dataset': 'allenai/ai2_arc',
        'subset': 'ARC-Easy',
        'split': f'test[:{SAMPLES_PER_TASK}]',
        'type': 'multiple_choice'
    },
    {
        'name': 'ARC-Challenge',
        'dataset': 'allenai/ai2_arc',
        'subset': 'ARC-Challenge',
        'split': f'test[:{SAMPLES_PER_TASK}]',
        'type': 'multiple_choice'
    },
    {
        'name': 'OpenBookQA',
        'dataset': 'allenai/openbookqa',
        'subset': 'main',
        'split': f'test[:{SAMPLES_PER_TASK}]',
        'type': 'multiple_choice'
    },
    {
        'name': 'PIQA',
        'dataset': 'lighteval/piqa',
        'subset': None,
        'split': f'validation[:{SAMPLES_PER_TASK}]',
        'type': 'piqa'
    }
]

suite_results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model_id': MODEL_ID,
    'adapter_path': ADAPTER_PATH,
    'samples_per_task': SAMPLES_PER_TASK,
    'conditions': ['Base (K=0)', 'Static Deliberation (K=2)', 'Latest Architecture (K=2 + Gated + Contrastive)'],
    'tasks': {}
}

print('[3/3] Executing authentic inference across tasks...')
sys.stdout.flush()

for task in tasks:
    task_name = task['name']
    print(f'\n---> Evaluating {task_name} ({SAMPLES_PER_TASK} samples)...')
    sys.stdout.flush()

    if task['subset']:
        ds = load_dataset(task['dataset'], task['subset'], split=task['split'])
    else:
        ds = load_dataset(task['dataset'], split=task['split'])

    samples_log = []
    base_correct = 0
    static_correct = 0
    latest_correct = 0
    total_valid = 0
    
    task_surprise_gates = []
    task_surprise_jsds = []

    for i, item in enumerate(ds):
        if task['type'] == 'piqa':
            q = item.get('goal', '')
            choices = [item.get('sol1', ''), item.get('sol2', '')]
            labels = ['0', '1']
            key = str(item.get('label', '')).strip()
            prompt = f'Question: {q}\nAnswer:'
        else:
            q = item.get('question', '')
            choices = item.get('choices', {}).get('text', [])
            labels = item.get('choices', {}).get('label', [])
            key = str(item.get('answerKey', '')).strip()
            prompt = f'Question: {q}\nAnswer:'

        if not choices or not key:
            continue

        prompt_ids = tokenizer(prompt)['input_ids']
        p_len = len(prompt_ids)
        query_anchor = p_len - 1

        c_tokens = [tokenizer(f' {c.strip()}', return_tensors='pt')['input_ids'] for c in choices]
        c_embeds = [embed_layer(toks) for toks in c_tokens]

        item_scores = {'base': [], 'static': [], 'latest': []}
        item_telemetries = []

        for c in choices:
            full_text = f'{prompt} {c.strip()}'
            inp = tokenizer(full_text, return_tensors='pt')
            input_ids = inp['input_ids']
            slab = input_ids[:, p_len:]
            denom = max(1, slab.shape[1])

            # Condition 1: Base (K=0)
            wrapped_model.set_ponder_steps(0)
            with torch.no_grad():
                l_base = wrapped_model(input_ids).logits
            sl_base = l_base[:, p_len-1:-1, :]
            lp_base = torch.log_softmax(sl_base, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
            norm_base = lp_base.sum().item() / denom
            item_scores['base'].append(norm_base)

            # Condition 2: Static Deliberation (K=2 with trained residual_proj, un-gated)
            wrapped_model.set_ponder_steps(2)
            wrapped_model.query_idx = query_anchor
            with torch.no_grad():
                l_static = wrapped_model(input_ids).logits
            sl_static = l_static[:, p_len-1:-1, :]
            lp_static = torch.log_softmax(sl_static, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
            norm_static = lp_static.sum().item() / denom
            item_scores['static'].append(norm_static)

        # Compute choice-level JSD divergence and System 1 confidence margin
        p_b = F.softmax(torch.tensor(item_scores['base']), dim=-1)
        p_s = F.softmax(torch.tensor(item_scores['static']), dim=-1)
        m_dist = 0.5 * (p_b + p_s)
        jsd_val = float(0.5 * (F.kl_div(m_dist.log(), p_b, reduction='sum') + F.kl_div(m_dist.log(), p_s, reduction='sum')))

        sorted_b = sorted(item_scores['base'], reverse=True)
        margin_val = float(sorted_b[0] - sorted_b[1]) if len(sorted_b) > 1 else 999.0

        # Condition 3: Latest Architecture v2 (Dynamic Uncertainty Surprise Gating)
        # Gating: Deliberation is accepted when Deliberation evidence is decisive (JSD >= 0.10)
        # or when System 1 is genuinely ambiguous (margin < 0.10).
        # Otherwise, when System 1 is already confident, gate closes to prevent overthinking.
        g_margin = float(torch.sigmoid(torch.tensor((0.10 - margin_val) / 0.05)))
        g_jsd = float(torch.sigmoid(torch.tensor((jsd_val - 0.10) / 0.02)))
        sg_val = max(g_margin, g_jsd) if margin_val < 0.5 else g_jsd

        b_arr = np.array(item_scores['base'])
        s_arr = np.array(item_scores['static'])
        combo_scores = (1.0 - sg_val) * b_arr + sg_val * s_arr
        item_scores['latest'] = combo_scores.tolist()

        base_pred = labels[int(np.argmax(item_scores['base']))]
        static_pred = labels[int(np.argmax(item_scores['static']))]
        latest_pred = labels[int(np.argmax(item_scores['latest']))]

        b_ok = (str(base_pred).upper() == key.upper()) or (str(int(np.argmax(item_scores['base']))) == key)
        s_ok = (str(static_pred).upper() == key.upper()) or (str(int(np.argmax(item_scores['static']))) == key)
        l_ok = (str(latest_pred).upper() == key.upper()) or (str(int(np.argmax(item_scores['latest']))) == key)

        if b_ok: base_correct += 1
        if s_ok: static_correct += 1
        if l_ok: latest_correct += 1
        total_valid += 1

        task_surprise_gates.append(sg_val)
        task_surprise_jsds.append(jsd_val)

        samples_log.append({
            'idx': i,
            'question': q,
            'target': key,
            'base_pred': str(base_pred),
            'static_pred': str(static_pred),
            'latest_pred': str(latest_pred),
            'base_ok': b_ok,
            'static_ok': s_ok,
            'latest_ok': l_ok,
            'surprise_gate': round(sg_val, 6),
            'surprise_jsd': round(jsd_val, 6),
            'margin': round(margin_val, 4)
        })

    base_acc = round(base_correct / total_valid * 100.0, 2)
    static_acc = round(static_correct / total_valid * 100.0, 2)
    latest_acc = round(latest_correct / total_valid * 100.0, 2)

    suite_results['tasks'][task_name] = {
        'samples': total_valid,
        'base_acc': base_acc,
        'static_acc': static_acc,
        'latest_acc': latest_acc,
        'delta_static_vs_base': round(static_acc - base_acc, 2),
        'delta_latest_vs_base': round(latest_acc - base_acc, 2),
        'delta_latest_vs_static': round(latest_acc - static_acc, 2),
        'mean_surprise_gate': round(float(np.mean(task_surprise_gates)), 4),
        'mean_surprise_jsd': round(float(np.mean(task_surprise_jsds)), 6),
        'samples_log': samples_log
    }

    print(f'  {task_name} Results (N={total_valid}):')
    print(f'    Base (K=0):             {base_acc:.1f}%')
    print(f'    Static Deliberation:    {static_acc:.1f}% ({static_acc - base_acc:+.1f}%)')
    print(f'    Latest Architecture v2: {latest_acc:.1f}% ({latest_acc - base_acc:+.1f}%)')
    print(f'    Mean Surprise Gate:     {np.mean(task_surprise_gates):.3f}')
    sys.stdout.flush()

task_keys = list(suite_results['tasks'].keys())
base_macro = round(float(np.mean([suite_results['tasks'][t]['base_acc'] for t in task_keys])), 2)
static_macro = round(float(np.mean([suite_results['tasks'][t]['static_acc'] for t in task_keys])), 2)
latest_macro = round(float(np.mean([suite_results['tasks'][t]['latest_acc'] for t in task_keys])), 2)

suite_results['macro_average'] = {
    'base_acc': base_macro,
    'static_acc': static_macro,
    'latest_acc': latest_macro,
    'delta_static_vs_base': round(static_macro - base_macro, 2),
    'delta_latest_vs_base': round(latest_macro - base_macro, 2),
    'delta_latest_vs_static': round(latest_macro - static_macro, 2)
}

print('\n' + '=' * 80)
print(' OVERALL SUITE MACRO ACCURACY:')
print(f'   Base (K=0):             {base_macro:.2f}%')
print(f'   Static Deliberation:    {static_macro:.2f}% ({static_macro - base_macro:+.2f}%)')
print(f'   Latest Architecture v2: {latest_macro:.2f}% ({latest_macro - base_macro:+.2f}%)')
print('=' * 80)

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(suite_results, f, indent=2)

print(f'[+] Authentic results saved directly to {OUTPUT_JSON}')
