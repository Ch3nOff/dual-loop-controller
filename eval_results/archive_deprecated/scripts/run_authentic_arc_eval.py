import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from dual_loop import attach_dual_loop_to_qwen
from dual_loop.verification import DirectionalSafetyProjection

MODEL_ID = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
ADAPTER_PATH = 'dual_loop/checkpoints/adapter_model.safetensors'
OUTPUT_DIR = 'eval_results'
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'arc_challenge_authentic_eval_n100.json')
N_SAMPLES = 100

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_arc_challenge_eval():
    set_seed(1337)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print('=' * 80)
    print(f'AUTHENTIC OFFICIAL BENCHMARK RUN: ARC-Challenge (N={N_SAMPLES})')
    print('=' * 80)
    print(f'Model Backbone : {MODEL_ID}')
    print(f'Adapter Path   : {ADAPTER_PATH}')
    print(f'Target Split   : allenai/ai2_arc (ARC-Challenge, split=\'test\')')
    print(f'Samples (N)    : {N_SAMPLES}')
    print('=' * 80)

    print('[*] Loading tokenizer and base model...')
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        device_map='cpu',
        torch_dtype=torch.float32
    )
    print(f'[+] Base model loaded in {round(time.time() - t0, 1)}s')

    print('[*] Attaching Dual-Loop Cognitive Controller at Layer 11...')
    wrapped = attach_dual_loop_to_qwen(
        base_model,
        layer_idx=11,
        k_steps=2,
        enable_plasticity=True,
        use_evidential_gate=True,
        use_surprise_gate=True,
        use_hypothesis_verification=True,
        use_contrastive_evidence=True
    )
    if os.path.exists(ADAPTER_PATH):
        print(f'[*] Loading adapter weights from {ADAPTER_PATH}...')
        wrapped.load_adapter(ADAPTER_PATH, strict=False)
    wrapped.adapter.eval()

    print('[*] Loading ARC-Challenge dataset...')
    ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
    print(f'[+] Total ARC-Challenge test split size: {len(ds)} items')

    items_to_eval = ds.select(range(N_SAMPLES))

    total_base_ok = 0
    total_delib_ok = 0
    rescued_count = 0
    degraded_count = 0
    preserved_correct = 0
    preserved_wrong = 0

    logs = []
    t_start = time.time()

    print(f'[*] Commencing evaluation over {N_SAMPLES} genuine questions...')
    for idx, item in enumerate(items_to_eval):
        t_item_0 = time.time()
        question = item['question']
        choices = item['choices']['text']
        labels = item['choices']['label']
        target = str(item['answerKey']).strip()

        # Format prompt standard QA
        prompt = f'Question: {question.strip()}\nAnswer:'
        p_ids = tokenizer(prompt)['input_ids']
        p_len = len(p_ids)
        query_anchor = p_len - 1

        # Build candidate embeddings
        cand_tensors = []
        for c in choices:
            c_ids = tokenizer(c.strip(), return_tensors='pt')['input_ids']
            with torch.no_grad():
                c_emb = wrapped.qwen.get_input_embeddings()(c_ids)
            cand_tensors.append(c_emb.mean(dim=1, keepdim=True))
        joint_cands = torch.cat(cand_tensors, dim=1) if cand_tensors else None

        # 1. Base Forward Pass (K=0)
        scores_base = []
        wrapped.set_candidate_embeds(None)
        wrapped.set_ponder_steps(0)
        wrapped.reset_state(force=True)
        for c in choices:
            full_text = f'{prompt} {c.strip()}'
            in_ids = tokenizer(full_text, return_tensors='pt')['input_ids']
            slab = in_ids[:, p_len:]
            denom = max(1, slab.shape[1])
            with torch.no_grad():
                logits_b = wrapped(in_ids).logits
            sl_b = logits_b[:, p_len-1:-1, :]
            lp_b = torch.log_softmax(sl_b, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
            scores_base.append(float(lp_b.sum().item() / denom))

        sorted_b = sorted(scores_base, reverse=True)
        margin_base = float(sorted_b[0] - sorted_b[1]) if len(sorted_b) > 1 else 999.0
        pred_base_idx = int(np.argmax(scores_base))
        pred_base = str(labels[pred_base_idx])
        base_ok = (pred_base.upper() == target.upper()) or (str(pred_base_idx) == target)

        # 2. Dual-Loop Deliberation (K=2)
        scores_delib = []
        telemetries = []
        wrapped.set_candidate_embeds(joint_cands)
        wrapped.set_ponder_steps(2)
        wrapped.query_idx = query_anchor
        wrapped.reset_state(force=False)
        for c in choices:
            full_text = f'{prompt} {c.strip()}'
            in_ids = tokenizer(full_text, return_tensors='pt')['input_ids']
            slab = in_ids[:, p_len:]
            denom = max(1, slab.shape[1])
            with torch.no_grad():
                logits_d = wrapped(in_ids).logits
            sl_d = logits_d[:, p_len-1:-1, :]
            lp_d = torch.log_softmax(sl_d, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
            scores_delib.append(float(lp_d.sum().item() / denom))
            telemetries.append(dict(wrapped.last_telemetry))

        p_b = F.softmax(torch.tensor(scores_base), dim=-1)
        p_d = F.softmax(torch.tensor(scores_delib), dim=-1)
        m_dist = 0.5 * (p_b + p_d)
        jsd_val = float(0.5 * (F.kl_div(m_dist.log(), p_b, reduction='sum') + F.kl_div(m_dist.log(), p_d, reduction='sum')))

        last_telem = telemetries[-1] if telemetries else {}
        vacuity_u = float(last_telem.get('epistemic_vacuity', [0.5])[0]) if last_telem.get('epistemic_vacuity') else 0.5
        trace_norm = float(last_telem.get('plastic_trace_norm', 0.0))

        g_margin = float(torch.sigmoid(torch.tensor((0.10 - margin_base) / 0.05)))
        g_jsd = float(torch.sigmoid(torch.tensor((jsd_val - 0.10) / 0.02)))
        sg_val = max(g_margin, g_jsd, 0.40) if joint_cands is not None else max(g_margin, g_jsd)
        eta_epistemic = max(0.20, min(1.0, 1.0 - max(0.0, vacuity_u - 0.50) / 0.50))
        sg_val = sg_val * eta_epistemic

        raw_combo = (1.0 - sg_val) * np.array(scores_base) + sg_val * np.array(scores_delib)

        combo_scores = DirectionalSafetyProjection.project_choice_scores(
            scores_base=np.array(scores_base),
            scores_delib=raw_combo,
            base_margin=margin_base,
            confidence_threshold=0.35,
            tie_breaker_threshold=0.05,
            delib_conviction_threshold=0.28,
            vacuity_u=vacuity_u
        )

        pred_delib_idx = int(np.argmax(combo_scores))
        pred_delib = str(labels[pred_delib_idx])
        delib_ok = (pred_delib.upper() == target.upper()) or (str(pred_delib_idx) == target)

        if base_ok and delib_ok:
            status = 'Preserved Correct'
            preserved_correct += 1
            total_base_ok += 1
            total_delib_ok += 1
        elif not base_ok and delib_ok:
            status = 'Rescued (Wrong->Right)'
            rescued_count += 1
            total_delib_ok += 1
        elif base_ok and not delib_ok:
            status = 'Degraded (Right->Wrong)'
            degraded_count += 1
            total_base_ok += 1
        else:
            status = 'Preserved Wrong'
            preserved_wrong += 1

        elapsed_item = time.time() - t_item_0

        log_entry = {
            'idx': idx,
            'id': item.get('id', f'arc_{idx}'),
            'question': question,
            'choices': choices,
            'labels': labels,
            'target': target,
            'pred_base': pred_base,
            'pred_delib': pred_delib,
            'base_ok': base_ok,
            'delib_ok': delib_ok,
            'status': status,
            'margin_base': margin_base,
            'vacuity_u': vacuity_u,
            'scores_base': scores_base,
            'scores_delib': scores_delib,
            'combo_scores': combo_scores.tolist(),
            'elapsed_seconds': round(elapsed_item, 3)
        }
        logs.append(log_entry)

        if (idx + 1) % 10 == 0 or idx == N_SAMPLES - 1:
            curr_base_acc = (total_base_ok / (idx + 1)) * 100.0
            curr_delib_acc = (total_delib_ok / (idx + 1)) * 100.0
            print(f'  [{idx+1:3d}/{N_SAMPLES}] Base: {curr_base_acc:.1f}% | Delib: {curr_delib_acc:.1f}% | Rescued: {rescued_count} | Degraded: {degraded_count} | Status: {status}')

        wrapped.set_candidate_embeds(None)
        wrapped.reset_state(force=False)

    total_time = time.time() - t_start
    final_base_acc = (total_base_ok / N_SAMPLES) * 100.0
    final_delib_acc = (total_delib_ok / N_SAMPLES) * 100.0
    delta = final_delib_acc - final_base_acc

    b = rescued_count
    c = degraded_count
    from scipy.stats import binom
    n_discordant = b + c
    if n_discordant > 0:
        p_val = 2.0 * min(binom.cdf(min(b, c), n_discordant, 0.5), 1.0 - binom.cdf(max(b, c) - 1, n_discordant, 0.5))
        p_val = min(1.0, float(p_val))
    else:
        p_val = 1.0

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'benchmark': 'allenai/ai2_arc',
        'split': 'test',
        'config': 'ARC-Challenge',
        'model_id': MODEL_ID,
        'revision': REVISION,
        'adapter_path': ADAPTER_PATH,
        'total_samples': N_SAMPLES,
        'total_elapsed_seconds': round(total_time, 2),
        'avg_seconds_per_sample': round(total_time / N_SAMPLES, 3),
        'base_accuracy': final_base_acc,
        'deliberation_accuracy': final_delib_acc,
        'delta': round(delta, 2),
        'contingency_matrix': {
            'preserved_correct': preserved_correct,
            'rescued_wrong_to_right': rescued_count,
            'degraded_right_to_wrong': degraded_count,
            'preserved_wrong': preserved_wrong
        },
        'mcnemar': {
            'discordant_pairs': n_discordant,
            'p_value': round(p_val, 4),
            'statistically_significant_p05': p_val < 0.05
        },
        'samples_log': logs
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('=' * 80)
    print(f'AUTHENTIC EVALUATION COMPLETE: {N_SAMPLES} SAMPLES')
    print(f'Base Accuracy        : {final_base_acc:.2f}% ({total_base_ok}/{N_SAMPLES})')
    print(f'Dual-Loop Accuracy   : {final_delib_acc:.2f}% ({total_delib_ok}/{N_SAMPLES})')
    print(f'Delta                : {delta:+.2f}%')
    print(f'Rescued              : {rescued_count}')
    print(f'Degraded             : {degraded_count}')
    print(f'McNemar p-value      : {p_val:.4f} (Significant: {p_val < 0.05})')
    print(f'Total Elapsed Time   : {total_time:.1f}s ({round(total_time / N_SAMPLES, 2)}s/item)')
    print(f'Full Log Saved To    : {OUTPUT_JSON}')
    print('=' * 80)

if __name__ == '__main__':
    run_arc_challenge_eval()
