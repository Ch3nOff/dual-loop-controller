import os, sys, time, json, random, numpy as np, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dual_loop import attach_dual_loop_to_qwen, CognitiveMatrixHelper
from dual_loop.verification import DirectionalSafetyProjection

MODEL_ID = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
ADAPTER_PATH = 'dual_loop/checkpoints/adapter_model.safetensors'
OUTPUT_DIR = 'eval_results'
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'sciq_msqa_matrix_helper_eval_n100.json')
N_SAMPLES = 100

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_evaluation():
    set_seed(1337)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=' * 80)
    print(f'AUTHENTIC EVALUATION: SciQ MSQA and Matrix Helper (N={N_SAMPLES})')
    print('=' * 80)
    print(f'Backbone Model : {MODEL_ID}')
    print(f'Adapter Path   : {ADAPTER_PATH}')
    print(f'Benchmark      : allenai/sciq (Science QA, split=test)')
    print(f'Samples        : {N_SAMPLES}')
    print('=' * 80)

    print('[*] Loading tokenizer and model...')
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, device_map='cpu', torch_dtype=torch.float32)
    wrapped = attach_dual_loop_to_qwen(base_model, layer_idx=11, k_steps=2, enable_plasticity=True, use_evidential_gate=True, use_surprise_gate=True, use_hypothesis_verification=True, use_contrastive_evidence=True)
    if os.path.exists(ADAPTER_PATH):
        wrapped.load_adapter(ADAPTER_PATH, strict=False)
    wrapped.adapter.eval()
    print(f'[+] Model loaded in {round(time.time() - t0, 1)}s')

    matrix_helper = CognitiveMatrixHelper(elimination_threshold=0.12, min_survivors=2)
    print('[*] Loading allenai/sciq dataset...')
    ds = load_dataset('allenai/sciq', split='test')
    items = ds.select(range(N_SAMPLES))

    base_correct, delib_correct, matrix_correct = 0, 0, 0
    base_rescued, base_degraded = 0, 0
    matrix_rescued, matrix_degraded = 0, 0
    total_pruned_options = 0

    logs = []
    t_start = time.time()

    for idx, it in enumerate(items):
        t_item_0 = time.time()
        q = it['question'].strip()
        correct_text = it['correct_answer'].strip()
        raw_choices = [correct_text, it['distractor1'].strip(), it['distractor2'].strip(), it['distractor3'].strip()]
        
        rng = random.Random(1337 + idx)
        perm = list(range(4))
        rng.shuffle(perm)
        shuffled_choices = [raw_choices[p] for p in perm]
        labels = ['A', 'B', 'C', 'D']
        target_idx = perm.index(0)
        target_label = labels[target_idx]

        prompt = f'Question: {q}\nAnswer:'
        p_ids = tokenizer(prompt)['input_ids']
        p_len = len(p_ids)
        query_anchor = p_len - 1

        cand_tensors = []
        for c in shuffled_choices:
            c_ids = tokenizer(c, return_tensors='pt')['input_ids']
            with torch.no_grad():
                c_emb = wrapped.qwen.get_input_embeddings()(c_ids)
            cand_tensors.append(c_emb.mean(dim=1, keepdim=True))
        joint_cands = torch.cat(cand_tensors, dim=1) if cand_tensors else None

        # 1. Base Forward Pass (K=0)
        scores_base = []
        wrapped.set_candidate_embeds(None)
        wrapped.set_ponder_steps(0)
        wrapped.reset_state(force=True)
        for c in shuffled_choices:
            in_ids = tokenizer(f'{prompt} {c}', return_tensors='pt')['input_ids']
            slab = in_ids[:, p_len:]
            denom = max(1, slab.shape[1])
            with torch.no_grad():
                logits_b = wrapped(in_ids).logits
            sl_b = logits_b[:, p_len-1:-1, :]
            lp_b = torch.log_softmax(sl_b, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
            scores_base.append(float(lp_b.sum().item() / denom))

        margin_base = float(sorted(scores_base, reverse=True)[0] - sorted(scores_base, reverse=True)[1])
        pred_base_idx = int(np.argmax(scores_base))
        pred_base = labels[pred_base_idx]
        base_ok = (pred_base_idx == target_idx)

        # 2. Dual-Loop Deliberation (K=2)
        scores_delib = []
        telemetries = []
        wrapped.set_candidate_embeds(joint_cands)
        wrapped.set_ponder_steps(2)
        wrapped.query_idx = query_anchor
        wrapped.reset_state(force=False)
        for c in shuffled_choices:
            in_ids = tokenizer(f'{prompt} {c}', return_tensors='pt')['input_ids']
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
        g_margin = float(torch.sigmoid(torch.tensor((0.10 - margin_base) / 0.05)))
        g_jsd = float(torch.sigmoid(torch.tensor((jsd_val - 0.10) / 0.02)))
        sg_val = max(g_margin, g_jsd, 0.40) if joint_cands is not None else max(g_margin, g_jsd)
        eta_epistemic = max(0.20, min(1.0, 1.0 - max(0.0, vacuity_u - 0.50) / 0.50))
        sg_val = sg_val * eta_epistemic
        raw_combo = (1.0 - sg_val) * np.array(scores_base) + sg_val * np.array(scores_delib)
        combo_scores = DirectionalSafetyProjection.project_choice_scores(
            scores_base=np.array(scores_base), scores_delib=raw_combo, base_margin=margin_base, confidence_threshold=0.35, tie_breaker_threshold=0.05, delib_conviction_threshold=0.28, vacuity_u=vacuity_u
        )
        pred_delib_idx = int(np.argmax(combo_scores))
        pred_delib = labels[pred_delib_idx]
        delib_ok = (pred_delib_idx == target_idx)

        # 3. Dual-Loop + Cognitive Matrix Helper
        matrix_info = matrix_helper.build_evidence_matrix(scores_base, labels=labels)
        survivors = matrix_info['survivors']
        eliminated = matrix_info['eliminated_indices']
        total_pruned_options += len(eliminated)

        surv_cands = joint_cands[:, survivors, :] if joint_cands is not None else None
        scores_delib_surv = []
        wrapped.set_candidate_embeds(surv_cands)
        wrapped.set_ponder_steps(2)
        wrapped.reset_state(force=False)
        for s_idx in survivors:
            c = shuffled_choices[s_idx]
            in_ids = tokenizer(f'{prompt} {c}', return_tensors='pt')['input_ids']
            slab = in_ids[:, p_len:]
            denom = max(1, slab.shape[1])
            with torch.no_grad():
                logits_m = wrapped(in_ids).logits
            sl_m = logits_m[:, p_len-1:-1, :]
            lp_m = torch.log_softmax(sl_m, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
            scores_delib_surv.append(float(lp_m.sum().item() / denom))

        fused_matrix_scores = matrix_helper.fuse_scores(
            scores_base=scores_base, scores_delib_survivors=scores_delib_surv, survivor_indices=survivors, lambda_delib=0.85
        )
        pred_matrix_idx = int(np.argmax(fused_matrix_scores))
        pred_matrix = labels[pred_matrix_idx]
        matrix_ok = (pred_matrix_idx == target_idx)

        if base_ok: base_correct += 1
        if delib_ok: delib_correct += 1
        if matrix_ok: matrix_correct += 1

        if not base_ok and delib_ok: base_rescued += 1
        if base_ok and not delib_ok: base_degraded += 1
        if not base_ok and matrix_ok: matrix_rescued += 1
        if base_ok and not matrix_ok: matrix_degraded += 1

        elapsed_item = time.time() - t_item_0

        entry = {
            'idx': idx, 'question': q, 'target': target_label, 'target_text': correct_text,
            'choices': shuffled_choices, 'labels': labels,
            'pred_base': pred_base, 'pred_delib': pred_delib, 'pred_matrix': pred_matrix,
            'base_ok': base_ok, 'delib_ok': delib_ok, 'matrix_ok': matrix_ok,
            'survivor_labels': matrix_info['survivor_labels'],
            'eliminated_labels': matrix_info['eliminated_labels'],
            'elapsed_seconds': round(elapsed_item, 3)
        }
        logs.append(entry)

        if (idx + 1) % 10 == 0 or idx == N_SAMPLES - 1:
            n_curr = idx + 1
            print(f'  [{n_curr:3d}/{N_SAMPLES}] Base: {(base_correct/n_curr)*100:.1f}% | Dual-Loop: {(delib_correct/n_curr)*100:.1f}% | Matrix: {(matrix_correct/n_curr)*100:.1f}% (Pruned avg: {total_pruned_options/n_curr:.1f} options)')

        wrapped.set_candidate_embeds(None)
        wrapped.reset_state(force=False)

    total_time = time.time() - t_start
    base_acc = (base_correct / N_SAMPLES) * 100.0
    delib_acc = (delib_correct / N_SAMPLES) * 100.0
    matrix_acc = (matrix_correct / N_SAMPLES) * 100.0

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'benchmark': 'allenai/sciq',
        'category': 'Science Question Answering (MSQA)',
        'split': 'test',
        'total_samples': N_SAMPLES,
        'total_elapsed_seconds': round(total_time, 2),
        'base_accuracy': base_acc,
        'deliberation_accuracy': delib_acc,
        'matrix_helper_accuracy': matrix_acc,
        'delib_delta': round(delib_acc - base_acc, 2),
        'matrix_delta': round(matrix_acc - base_acc, 2),
        'delib_rescued': base_rescued,
        'delib_degraded': base_degraded,
        'matrix_rescued': matrix_rescued,
        'matrix_degraded': matrix_degraded,
        'avg_options_pruned_per_item': round(total_pruned_options / N_SAMPLES, 2),
        'samples_log': logs
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('=' * 80)
    print(f'AUTHENTIC MSQA / SciQ BENCHMARK RESULTS (N={N_SAMPLES})')
    print('=' * 80)
    print(f'1. Base Qwen3.5-2B (K=0)       : {base_acc:.2f}% ({base_correct}/{N_SAMPLES})')
    print(f'2. Dual-Loop Controller (K=2)  : {delib_acc:.2f}% ({delib_correct}/{N_SAMPLES}) [Delta: {delib_acc - base_acc:+.2f}%]')
    print(f'3. Dual-Loop + Matrix Helper   : {matrix_acc:.2f}% ({matrix_correct}/{N_SAMPLES}) [Delta: {matrix_acc - base_acc:+.2f}%]')
    print(f'   - Rescued by Matrix Helper  : {matrix_rescued}')
    print(f'   - Degraded by Matrix Helper : {matrix_degraded}')
    print(f'   - Avg Distractors Pruned    : {total_pruned_options / N_SAMPLES:.2f} options / question')
    print(f'Total Elapsed Time             : {total_time:.1f}s ({round(total_time / N_SAMPLES, 2)}s/item)')
    print(f'Log File                       : {OUTPUT_JSON}')
    print('=' * 80)

if __name__ == '__main__':
    run_evaluation()
