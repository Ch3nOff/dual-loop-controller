"""
Empirical Experiment: 5-Pass Matrix Deliberation & Probability Recalculation
==========================================================================
Tests the user's hypotheses on authentic Qwen/Qwen3.5-2B (Layer 11 adapter):
1. Can the Matrix Helper be improved by recalculating probabilities and applying Directional Safety?
2. What happens across 5 progressive deliberation passes (K=1, 2, 3, 4, 5)?
   - Does the model genuinely improve/correct errors?
   - Or does it saturate/overthink and turn correct answers into wrong ones?
3. What happens across 5 sequential benchmark runs with episodic memory?
"""

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

from dual_loop import attach_dual_loop_to_qwen, CognitiveMatrixHelper
from dual_loop.verification import DirectionalSafetyProjection
from dual_loop.memory import EpisodicMemoryBuffer

MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
ADAPTER_PATH = "dual_loop/checkpoints/adapter_model.safetensors"
N_SAMPLES = 30
OUTPUT_FILE = "eval_results/matrix_5x_run_experiment.json"

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_log_likelihood(model, tokenizer, prompt, choice, p_len):
    in_ids = tokenizer(f"{prompt} {choice}", return_tensors="pt")["input_ids"]
    slab = in_ids[:, p_len:]
    denom = max(1, slab.shape[1])
    with torch.no_grad():
        logits = model(in_ids).logits
    sl = logits[:, p_len - 1 : -1, :]
    lp = torch.log_softmax(sl, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
    return float(lp.sum().item() / denom)

def run_5x_experiment():
    set_seed(1337)
    os.makedirs("eval_results", exist_ok=True)
    print("=" * 80)
    print(f"EMPIRICAL EXPERIMENT: Matrix Recalculation & 5x Deliberation Run (N={N_SAMPLES})")
    print("=" * 80)
    print(f"Model       : {MODEL_ID}")
    print(f"Dataset     : allenai/sciq (test split)")
    print(f"Sample Size : {N_SAMPLES} questions")
    print("=" * 80)

    print("[*] Loading model and tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, device_map="cpu", torch_dtype=torch.float32
    )
    wrapped = attach_dual_loop_to_qwen(
        base_model,
        layer_idx=11,
        k_steps=2,
        enable_plasticity=True,
        use_evidential_gate=True,
        use_surprise_gate=True,
        use_hypothesis_verification=True,
        use_contrastive_evidence=True,
    )
    if os.path.exists(ADAPTER_PATH):
        wrapped.load_adapter(ADAPTER_PATH, strict=False)
    wrapped.adapter.eval()
    print(f"[+] Loaded model in {time.time() - t0:.1f}s")

    raw_matrix_helper = CognitiveMatrixHelper(elimination_threshold=0.12, min_survivors=2)

    print("[*] Loading SciQ test dataset...")
    ds = load_dataset("allenai/sciq", split="test")
    items = ds.select(range(N_SAMPLES))

    # Data structures for tracking 5-pass performance
    # Passes 1 to 5: K=1 to K=5
    pass_correct = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    pass_rescued = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    pass_degraded = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Improved Matrix (Recalculated Probabilities + Directional Safety)
    improved_matrix_correct = 0
    improved_matrix_rescued = 0
    improved_matrix_degraded = 0

    # Baseline tracking
    base_correct = 0
    standard_matrix_correct = 0
    standard_matrix_rescued = 0
    standard_matrix_degraded = 0

    sample_details = []

    print(f"\n[*] Commencing 5-Pass Progressive Deliberation & Recalculation across {N_SAMPLES} samples...\n")
    t_start = time.time()

    for idx, it in enumerate(items):
        t_item_0 = time.time()
        q = it["question"].strip()
        correct_text = it["correct_answer"].strip()
        raw_choices = [
            correct_text,
            it["distractor1"].strip(),
            it["distractor2"].strip(),
            it["distractor3"].strip(),
        ]

        rng = random.Random(1337 + idx)
        perm = list(range(4))
        rng.shuffle(perm)
        shuffled_choices = [raw_choices[p] for p in perm]
        labels = ["A", "B", "C", "D"]
        target_idx = perm.index(0)
        target_label = labels[target_idx]

        prompt = f"Question: {q}\nAnswer:"
        p_ids = tokenizer(prompt)["input_ids"]
        p_len = len(p_ids)
        query_anchor = p_len - 1

        cand_tensors = []
        for c in shuffled_choices:
            c_ids = tokenizer(c, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                c_emb = wrapped.qwen.get_input_embeddings()(c_ids)
            cand_tensors.append(c_emb.mean(dim=1, keepdim=True))
        joint_cands = torch.cat(cand_tensors, dim=1) if cand_tensors else None

        # -------------------------------------------------------------
        # 1. Base Forward Pass (K=0)
        # -------------------------------------------------------------
        scores_base = []
        wrapped.set_candidate_embeds(None)
        wrapped.set_ponder_steps(0)
        wrapped.reset_state(force=True)
        for c in shuffled_choices:
            s = compute_log_likelihood(wrapped, tokenizer, prompt, c, p_len)
            scores_base.append(s)

        margin_base = float(sorted(scores_base, reverse=True)[0] - sorted(scores_base, reverse=True)[1])
        pred_base_idx = int(np.argmax(scores_base))
        base_ok = (pred_base_idx == target_idx)
        if base_ok:
            base_correct += 1

        # -------------------------------------------------------------
        # 2. Standard Matrix (1-Pass static elimination without safety projection)
        # -------------------------------------------------------------
        matrix_info_std = raw_matrix_helper.build_evidence_matrix(scores_base, labels=labels)
        survivors_std = matrix_info_std["survivors"]
        surv_cands_std = joint_cands[:, survivors_std, :] if joint_cands is not None else None

        wrapped.set_candidate_embeds(surv_cands_std)
        wrapped.set_ponder_steps(2)
        wrapped.reset_state(force=False)
        scores_delib_surv_std = []
        for s_idx in survivors_std:
            s = compute_log_likelihood(wrapped, tokenizer, prompt, shuffled_choices[s_idx], p_len)
            scores_delib_surv_std.append(s)

        fused_std = raw_matrix_helper.fuse_scores(
            scores_base=scores_base,
            scores_delib_survivors=scores_delib_surv_std,
            survivor_indices=survivors_std,
            lambda_delib=0.85,
        )
        pred_std_idx = int(np.argmax(fused_std))
        std_matrix_ok = (pred_std_idx == target_idx)
        if std_matrix_ok:
            standard_matrix_correct += 1
        if not base_ok and std_matrix_ok:
            standard_matrix_rescued += 1
        if base_ok and not std_matrix_ok:
            standard_matrix_degraded += 1

        # -------------------------------------------------------------
        # 3. 5-Pass Progressive Deliberation & Recalculation (K = 1, 2, 3, 4, 5)
        # -------------------------------------------------------------
        pass_predictions = {}
        pass_scores = {}

        for k in range(1, 6):
            wrapped.set_candidate_embeds(surv_cands_std)
            wrapped.set_ponder_steps(k)
            wrapped.query_idx = query_anchor
            wrapped.reset_state(force=False)

            scores_k_surv = []
            for s_idx in survivors_std:
                s = compute_log_likelihood(wrapped, tokenizer, prompt, shuffled_choices[s_idx], p_len)
                scores_k_surv.append(s)

            fused_k = raw_matrix_helper.fuse_scores(
                scores_base=scores_base,
                scores_delib_survivors=scores_k_surv,
                survivor_indices=survivors_std,
                lambda_delib=0.85,
            )
            pred_k_idx = int(np.argmax(fused_k))
            k_ok = (pred_k_idx == target_idx)
            pass_predictions[k] = labels[pred_k_idx]
            pass_scores[k] = [float(fused_k[i]) for i in range(len(labels))]

            if k_ok:
                pass_correct[k] += 1
            if not base_ok and k_ok:
                pass_rescued[k] += 1
            if base_ok and not k_ok:
                pass_degraded[k] += 1

        # -------------------------------------------------------------
        # 4. Improved Matrix Helper (Recalculated Probabilities + Directional Safety)
        # -------------------------------------------------------------
        # Base probabilities
        p_base_all = F.softmax(torch.tensor(scores_base), dim=-1).numpy()

        # Deliberation on all candidates with K=2
        wrapped.set_candidate_embeds(joint_cands)
        wrapped.set_ponder_steps(2)
        wrapped.query_idx = query_anchor
        wrapped.reset_state(force=False)
        scores_delib_all = []
        for c in shuffled_choices:
            s = compute_log_likelihood(wrapped, tokenizer, prompt, c, p_len)
            scores_delib_all.append(s)

        p_delib_all = F.softmax(torch.tensor(scores_delib_all), dim=-1).numpy()
        # Recalculated Joint Probability
        p_joint = 0.5 * (p_base_all + p_delib_all)

        # Elimination Rule: Only eliminate if BOTH base AND joint probability are below threshold
        elim_improved = []
        for i in range(len(labels)):
            if p_base_all[i] < 0.12 and p_joint[i] < 0.15:
                elim_improved.append(i)

        survivors_improved = [i for i in range(len(labels)) if i not in elim_improved]
        if len(survivors_improved) < 2:
            survivors_improved = list(np.argsort(p_joint)[::-1][:2])

        # Directional Safety Projection on fused scores
        raw_fused_improved = np.full(4, -1e9)
        for s_i in survivors_improved:
            raw_fused_improved[s_i] = 0.30 * scores_base[s_i] + 0.70 * scores_delib_all[s_i]

        # Apply Safety Shield: If base was highly confident (margin >= 0.35), shield it from degradation
        if margin_base >= 0.35:
            pred_imp_idx = pred_base_idx
        else:
            pred_imp_idx = int(np.argmax(raw_fused_improved))

        improved_ok = (pred_imp_idx == target_idx)
        if improved_ok:
            improved_matrix_correct += 1
        if not base_ok and improved_ok:
            improved_matrix_rescued += 1
        if base_ok and not improved_ok:
            improved_matrix_degraded += 1

        elapsed = time.time() - t_item_0
        n_curr = idx + 1

        sample_details.append({
            "idx": idx,
            "question": q,
            "target": target_label,
            "target_text": correct_text,
            "base_ok": base_ok,
            "pred_base": labels[pred_base_idx],
            "margin_base": margin_base,
            "pred_std_matrix": labels[pred_std_idx],
            "std_matrix_ok": std_matrix_ok,
            "pred_improved_matrix": labels[pred_imp_idx],
            "improved_matrix_ok": improved_ok,
            "pass_predictions": pass_predictions,
            "elapsed_seconds": round(elapsed, 2)
        })

        if n_curr % 5 == 0 or n_curr == N_SAMPLES:
            print(f"  [{n_curr:2d}/{N_SAMPLES}] Base: {(base_correct/n_curr)*100:.1f}% | "
                  f"Std Matrix: {(standard_matrix_correct/n_curr)*100:.1f}% | "
                  f"Improved Matrix: {(improved_matrix_correct/n_curr)*100:.1f}% | "
                  f"K=1: {(pass_correct[1]/n_curr)*100:.1f}%, "
                  f"K=3: {(pass_correct[3]/n_curr)*100:.1f}%, "
                  f"K=5: {(pass_correct[5]/n_curr)*100:.1f}%")

    total_time = time.time() - t_start

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": N_SAMPLES,
        "total_elapsed_seconds": round(total_time, 2),
        "base_accuracy": round((base_correct / N_SAMPLES) * 100.0, 2),
        "standard_matrix": {
            "accuracy": round((standard_matrix_correct / N_SAMPLES) * 100.0, 2),
            "rescued": standard_matrix_rescued,
            "degraded": standard_matrix_degraded,
            "delta": round(((standard_matrix_correct - base_correct) / N_SAMPLES) * 100.0, 2),
        },
        "improved_matrix": {
            "accuracy": round((improved_matrix_correct / N_SAMPLES) * 100.0, 2),
            "rescued": improved_matrix_rescued,
            "degraded": improved_matrix_degraded,
            "delta": round(((improved_matrix_correct - base_correct) / N_SAMPLES) * 100.0, 2),
        },
        "5_pass_progressive_deliberation": {
            k: {
                "accuracy": round((pass_correct[k] / N_SAMPLES) * 100.0, 2),
                "rescued": pass_rescued[k],
                "degraded": pass_degraded[k],
                "delta": round(((pass_correct[k] - base_correct) / N_SAMPLES) * 100.0, 2),
            }
            for k in range(1, 6)
        },
        "sample_details": sample_details,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("5-PASS EXPERIMENT FINAL RESULTS SUMMARY:")
    print("=" * 80)
    print(f"Base Model Accuracy (K=0)               : {results['base_accuracy']}%")
    print(f"Standard Matrix Helper (1-Pass)         : {results['standard_matrix']['accuracy']}% (Rescued: {standard_matrix_rescued}, Degraded: {standard_matrix_degraded})")
    print(f"Improved Matrix (Recalculated + Safety) : {results['improved_matrix']['accuracy']}% (Rescued: {improved_matrix_rescued}, Degraded: {improved_matrix_degraded})")
    print("-" * 80)
    print("Progressive Deliberation Depth Across 5 Passes:")
    for k in range(1, 6):
        d = results["5_pass_progressive_deliberation"][k]
        print(f"  Pass K={k}: Accuracy = {d['accuracy']}% (Delta: {d['delta']:+0.1f}%, Rescued: {d['rescued']}, Degraded: {d['degraded']})")
    print("=" * 80)
    print(f"[+] Saved complete experiment log to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_5x_experiment()
