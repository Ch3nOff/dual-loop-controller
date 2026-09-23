"""
Empirical Head-to-Head Benchmark: Standard Cold-Start vs Wrong Log Active
========================================================================
Authentic PyTorch evaluation on Qwen/Qwen3.5-2B (frozen) across 3 standard benchmarks:
  1. AllenAI SciQ (Science QA, test split, N=25)
  2. AI2 ARC-Challenge (Complex Science, test split, N=25)
  3. AllenAI OpenBookQA (Multi-Hop Science, test split, N=25)
Total N = 75 challenging questions.

Compares 3 Model Configurations under 2 Evaluation Modes:
  Mode 1: Standard Cold-Start (Single-shot, zero prior error history)
  Mode 2: Wrong Log Active (Adaptive session where prior failed choice is registered and banned)

Configurations:
  - Base Qwen3.5-2B (K=0)
  - Dual-Loop Normal (K=2 latent deliberation + directional safety)
  - Dual-Loop x Wrong Log Active (System 2 deliberation + Persistent Cognitive Matrix Wrong Log Bank)

Outputs:
  - Raw JSON Log: eval_results/side_by_side_wronglog_eval.json
  - Publication Graph: eval_results/side_by_side_benchmark_wronglog_graph.png
"""

import os
import sys
import time
import json
import random
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dual_loop import attach_dual_loop_to_qwen, CognitiveMatrixHelper
from dual_loop.verification import DirectionalSafetyProjection

MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
ADAPTER_PATH = "dual_loop/checkpoints/adapter_model.safetensors"
SAMPLES_PER_BENCH = 25
OUTPUT_JSON = "eval_results/side_by_side_wronglog_eval.json"
OUTPUT_PNG = "eval_results/side_by_side_benchmark_wronglog_graph.png"

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_question_key(prompt: str) -> str:
    return hashlib.md5(prompt.strip().encode("utf-8")).hexdigest()[:12]

def compute_log_likelihood(model, tokenizer, prompt, choice, p_len):
    in_ids = tokenizer(f"{prompt} {choice}", return_tensors="pt")["input_ids"]
    slab = in_ids[:, p_len:]
    denom = max(1, slab.shape[1])
    with torch.no_grad():
        logits = model(in_ids).logits
    sl = logits[:, p_len - 1 : -1, :]
    lp = torch.log_softmax(sl, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
    return float(lp.sum().item() / denom)

def load_evaluation_datasets(n_samples=25):
    print(f"[*] Loading 3 standard benchmarks ({n_samples} items each)...")
    suite = {}

    # 1. SciQ
    ds_sciq = load_dataset("allenai/sciq", split="test")
    items_sciq = []
    for idx in range(min(n_samples, len(ds_sciq))):
        it = ds_sciq[idx]
        q = it["question"].strip()
        correct_text = it["correct_answer"].strip()
        raw_choices = [correct_text, it["distractor1"].strip(), it["distractor2"].strip(), it["distractor3"].strip()]
        rng = random.Random(1337 + idx)
        perm = list(range(4))
        rng.shuffle(perm)
        shuffled = [raw_choices[p] for p in perm]
        labels = ["A", "B", "C", "D"]
        target_label = labels[perm.index(0)]
        items_sciq.append({
            "prompt": f"Question: {q}\nAnswer:",
            "choices": shuffled,
            "labels": labels,
            "target": target_label,
            "target_text": correct_text
        })
    suite["SciQ"] = {"name": "AllenAI SciQ", "items": items_sciq}

    # 2. ARC-Challenge
    ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items_arc = []
    for idx in range(min(n_samples, len(ds_arc))):
        it = ds_arc[idx]
        q = (it.get("question") or it.get("question_stem", "")).strip()
        choices = it["choices"]["text"]
        labels = it["choices"]["label"]
        tgt = str(it["answerKey"]).strip()
        # Normalization
        label_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        norm_labels = [label_map.get(l, l) for l in labels]
        norm_tgt = label_map.get(tgt, tgt)
        items_arc.append({
            "prompt": f"Question: {q}\nAnswer:",
            "choices": choices,
            "labels": norm_labels,
            "target": norm_tgt,
            "target_text": choices[norm_labels.index(norm_tgt)] if norm_tgt in norm_labels else ""
        })
    suite["ARC-Challenge"] = {"name": "AI2 ARC-Challenge", "items": items_arc}

    # 3. OpenBookQA
    ds_obqa = load_dataset("allenai/openbookqa", "main", split="test")
    items_obqa = []
    for idx in range(min(n_samples, len(ds_obqa))):
        it = ds_obqa[idx]
        q = (it.get("question") or it.get("question_stem", "")).strip()
        choices = it["choices"]["text"]
        labels = it["choices"]["label"]
        tgt = str(it["answerKey"]).strip()
        items_obqa.append({
            "prompt": f"Question: {q}\nAnswer:",
            "choices": choices,
            "labels": labels,
            "target": tgt,
            "target_text": choices[labels.index(tgt)] if tgt in labels else ""
        })
    suite["OpenBookQA"] = {"name": "AllenAI OpenBookQA", "items": items_obqa}

    return suite

def run_side_by_side_benchmark():
    set_seed(1337)
    os.makedirs("eval_results", exist_ok=True)
    print("=" * 80)
    print("SIDE-BY-SIDE BENCHMARK: Cold-Start vs Wrong Log Active")
    print(f"Model: {MODEL_ID} | Samples: {SAMPLES_PER_BENCH} per benchmark | Total: {SAMPLES_PER_BENCH * 3}")
    print("=" * 80)

    print("[*] Loading Qwen3.5-2B model and Dual-Loop adapter...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, device_map="cpu", torch_dtype=torch.float32
    )
    wrapped = attach_dual_loop_to_qwen(base_model, layer_idx=11, k_steps=2)
    if os.path.exists(ADAPTER_PATH):
        wrapped.load_adapter(ADAPTER_PATH, strict=False)
    wrapped.adapter.eval()
    print(f"[+] Loaded in {time.time() - t0:.1f}s")

    suite = load_evaluation_datasets(SAMPLES_PER_BENCH)
    matrix_helper = CognitiveMatrixHelper(elimination_threshold=0.12, min_survivors=2)

    results_by_bench = {}
    macro_scores = {
        "mode1_cold_start": {"base": 0, "dualloop_normal": 0, "dualloop_wronglog": 0},
        "mode2_wronglog_active": {"base": 0, "dualloop_normal": 0, "dualloop_wronglog": 0},
        "total_samples": 0
    }

    t_start = time.time()

    for bkey, bdata in suite.items():
        bname = bdata["name"]
        items = bdata["items"]
        n_items = len(items)
        print(f"\n---> Evaluating Benchmark: {bname} ({n_items} items)...")

        # Scores tracking for this benchmark
        bench_scores = {
            "mode1_cold_start": {
                "base_correct": 0,
                "dualloop_normal_correct": 0,
                "dualloop_matrix_correct": 0
            },
            "mode2_wronglog_active": {
                "base_re_correct": 0,
                "dualloop_normal_re_correct": 0,
                "dualloop_wronglog_correct": 0
            }
        }

        bench_details = []

        # Clear wrong log bank for fresh benchmark run
        matrix_helper.clear_wrong_logs()

        for idx, it in enumerate(items):
            prompt = it["prompt"]
            choices = it["choices"]
            labels = it["labels"]
            target = it["target"]
            target_idx = labels.index(target) if target in labels else 0
            q_key = get_question_key(f"{bkey}_{prompt}")

            p_ids = tokenizer(prompt)["input_ids"]
            p_len = len(p_ids)
            query_anchor = p_len - 1

            # Candidate embeddings
            cand_tensors = []
            for c in choices:
                c_ids = tokenizer(c, return_tensors="pt")["input_ids"]
                with torch.no_grad():
                    c_emb = wrapped.qwen.get_input_embeddings()(c_ids)
                cand_tensors.append(c_emb.mean(dim=1, keepdim=True))
            joint_cands = torch.cat(cand_tensors, dim=1) if cand_tensors else None

            # -------------------------------------------------------------
            # Compute Raw Base Scores (K=0)
            # -------------------------------------------------------------
            wrapped.set_candidate_embeds(None)
            wrapped.set_ponder_steps(0)
            wrapped.reset_state(force=True)
            scores_base = []
            for c in choices:
                s = compute_log_likelihood(wrapped, tokenizer, prompt, c, p_len)
                scores_base.append(s)

            margin_base = float(sorted(scores_base, reverse=True)[0] - sorted(scores_base, reverse=True)[1]) if len(scores_base) > 1 else 999.0
            pred_base_idx = int(np.argmax(scores_base))
            pred_base = labels[pred_base_idx]
            base_ok = (pred_base_idx == target_idx)

            # -------------------------------------------------------------
            # Compute Dual-Loop Normal Scores (K=2 Deliberation on all candidates)
            # -------------------------------------------------------------
            wrapped.set_candidate_embeds(joint_cands)
            wrapped.set_ponder_steps(2)
            wrapped.query_idx = query_anchor
            wrapped.reset_state(force=False)
            scores_delib_all = []
            for c in choices:
                s = compute_log_likelihood(wrapped, tokenizer, prompt, c, p_len)
                scores_delib_all.append(s)

            raw_combo = 0.30 * np.array(scores_base) + 0.70 * np.array(scores_delib_all)
            combo_normal = DirectionalSafetyProjection.project_choice_scores(
                scores_base=np.array(scores_base),
                scores_delib=raw_combo,
                base_margin=margin_base,
                confidence_threshold=0.35
            )
            pred_normal_idx = int(np.argmax(combo_normal))
            pred_normal = labels[pred_normal_idx]
            normal_ok = (pred_normal_idx == target_idx)

            # -------------------------------------------------------------
            # Mode 1: Dual-Loop Matrix (Cold-Start: No prior Wrong Log)
            # -------------------------------------------------------------
            matrix_info_cold = matrix_helper.build_evidence_matrix(
                scores_base=scores_base,
                labels=labels,
                query_key=q_key # Empty at start
            )
            surv_cold = matrix_info_cold["survivors"]
            fused_cold = matrix_helper.fuse_scores(
                scores_base=scores_base,
                scores_delib_survivors=[scores_delib_all[s] for s in surv_cold],
                survivor_indices=surv_cold,
                lambda_delib=0.85
            )
            # Directional Safety Shield
            if margin_base >= 0.35:
                pred_matrix_cold_idx = pred_base_idx
            else:
                pred_matrix_cold_idx = int(np.argmax(fused_cold))
            pred_matrix_cold = labels[pred_matrix_cold_idx]
            matrix_cold_ok = (pred_matrix_cold_idx == target_idx)

            # Record Mode 1 outcomes
            if base_ok:
                bench_scores["mode1_cold_start"]["base_correct"] += 1
            if normal_ok:
                bench_scores["mode1_cold_start"]["dualloop_normal_correct"] += 1
            if matrix_cold_ok:
                bench_scores["mode1_cold_start"]["dualloop_matrix_correct"] += 1

            # -------------------------------------------------------------
            # Simulation of Error Feedback Registration:
            # If the model failed in Mode 1, the failed guess is logged to Wrong Log!
            # -------------------------------------------------------------
            if not matrix_cold_ok:
                matrix_helper.register_wrong_choice(q_key, pred_matrix_cold)

            # -------------------------------------------------------------
            # Mode 2: Wrong Log Active Evaluation
            # -------------------------------------------------------------
            # 2A. Base x Wrong Log (Base excludes previously failed option if any)
            banned_logs = matrix_helper.get_wrong_choices(q_key)
            scores_base_re = [scores_base[i] if labels[i] not in banned_logs else -1e9 for i in range(len(labels))]
            pred_base_re_idx = int(np.argmax(scores_base_re))
            pred_base_re = labels[pred_base_re_idx]
            base_re_ok = (pred_base_re_idx == target_idx)

            # 2B. Dual-Loop Normal (Without Wrong Log - remains identical to show static model)
            normal_re_ok = normal_ok

            # 2C. Dual-Loop x Wrong Log Active (System 2 + Active Wrong Log Bank)
            matrix_info_active = matrix_helper.build_evidence_matrix(
                scores_base=scores_base,
                labels=labels,
                query_key=q_key # Now has the registered error!
            )
            surv_active = matrix_info_active["survivors"]
            fused_active = matrix_helper.fuse_scores(
                scores_base=scores_base,
                scores_delib_survivors=[scores_delib_all[s] for s in surv_active],
                survivor_indices=surv_active,
                lambda_delib=0.85
            )
            pred_wl_active_idx = int(np.argmax(fused_active))
            pred_wl_active = labels[pred_wl_active_idx]
            wl_active_ok = (pred_wl_active_idx == target_idx)

            # Record Mode 2 outcomes
            if base_re_ok:
                bench_scores["mode2_wronglog_active"]["base_re_correct"] += 1
            if normal_re_ok:
                bench_scores["mode2_wronglog_active"]["dualloop_normal_re_correct"] += 1
            if wl_active_ok:
                bench_scores["mode2_wronglog_active"]["dualloop_wronglog_correct"] += 1

            bench_details.append({
                "idx": idx,
                "q_key": q_key,
                "prompt": prompt[:70] + "...",
                "target": target,
                "mode1_cold_start": {
                    "pred_base": pred_base, "base_ok": base_ok,
                    "pred_normal": pred_normal, "normal_ok": normal_ok,
                    "pred_matrix": pred_matrix_cold, "matrix_ok": matrix_cold_ok
                },
                "wrong_logs": banned_logs,
                "mode2_wronglog_active": {
                    "pred_base_re": pred_base_re, "base_re_ok": base_re_ok,
                    "pred_normal_re": pred_normal, "normal_re_ok": normal_re_ok,
                    "pred_dualloop_wronglog": pred_wl_active, "dualloop_wronglog_ok": wl_active_ok
                }
            })

        # Calculate percentages
        m1 = bench_scores["mode1_cold_start"]
        m2 = bench_scores["mode2_wronglog_active"]
        b_summary = {
            "n_samples": n_items,
            "mode1_cold_start": {
                "base_accuracy": round((m1["base_correct"] / n_items) * 100.0, 2),
                "dualloop_normal_accuracy": round((m1["dualloop_normal_correct"] / n_items) * 100.0, 2),
                "dualloop_wronglog_accuracy": round((m1["dualloop_matrix_correct"] / n_items) * 100.0, 2)
            },
            "mode2_wronglog_active": {
                "base_re_accuracy": round((m2["base_re_correct"] / n_items) * 100.0, 2),
                "dualloop_normal_re_accuracy": round((m2["dualloop_normal_re_correct"] / n_items) * 100.0, 2),
                "dualloop_wronglog_accuracy": round((m2["dualloop_wronglog_correct"] / n_items) * 100.0, 2)
            },
            "sample_details": bench_details
        }
        results_by_bench[bkey] = b_summary

        macro_scores["mode1_cold_start"]["base"] += m1["base_correct"]
        macro_scores["mode1_cold_start"]["dualloop_normal"] += m1["dualloop_normal_correct"]
        macro_scores["mode1_cold_start"]["dualloop_wronglog"] += m1["dualloop_matrix_correct"]

        macro_scores["mode2_wronglog_active"]["base"] += m2["base_re_correct"]
        macro_scores["mode2_wronglog_active"]["dualloop_normal"] += m2["dualloop_normal_re_correct"]
        macro_scores["mode2_wronglog_active"]["dualloop_wronglog"] += m2["dualloop_wronglog_correct"]
        macro_scores["total_samples"] += n_items

        print(f"   Mode 1 (Cold-Start)      : Base {b_summary['mode1_cold_start']['base_accuracy']}% | Dual-Loop {b_summary['mode1_cold_start']['dualloop_normal_accuracy']}% | DL x Matrix {b_summary['mode1_cold_start']['dualloop_wronglog_accuracy']}%")
        print(f"   Mode 2 (WrongLog Active) : Base {b_summary['mode2_wronglog_active']['base_re_accuracy']}% | Dual-Loop {b_summary['mode2_wronglog_active']['dualloop_normal_re_accuracy']}% | DL x WrongLog {b_summary['mode2_wronglog_active']['dualloop_wronglog_accuracy']}%")

    tot = macro_scores["total_samples"]
    macro_summary = {
        "total_samples": tot,
        "mode1_cold_start": {
            "base_accuracy": round((macro_scores["mode1_cold_start"]["base"] / tot) * 100.0, 2),
            "dualloop_normal_accuracy": round((macro_scores["mode1_cold_start"]["dualloop_normal"] / tot) * 100.0, 2),
            "dualloop_wronglog_accuracy": round((macro_scores["mode1_cold_start"]["dualloop_wronglog"] / tot) * 100.0, 2)
        },
        "mode2_wronglog_active": {
            "base_accuracy": round((macro_scores["mode2_wronglog_active"]["base"] / tot) * 100.0, 2),
            "dualloop_normal_accuracy": round((macro_scores["mode2_wronglog_active"]["dualloop_normal"] / tot) * 100.0, 2),
            "dualloop_wronglog_accuracy": round((macro_scores["mode2_wronglog_active"]["dualloop_wronglog"] / tot) * 100.0, 2)
        }
    }

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_ID,
        "samples_per_task": SAMPLES_PER_BENCH,
        "total_samples": tot,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "macro_summary": macro_summary,
        "benchmarks": results_by_bench
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[+] Saved complete evaluation log to: {OUTPUT_JSON}")

    # Generate Publication Graph
    plot_side_by_side_comparison(final_payload, OUTPUT_PNG)

def plot_side_by_side_comparison(data, output_png):
    print(f"[*] Plotting side-by-side comparison graph to {output_png}...")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), facecolor="#0f172a")

    b_keys = list(data["benchmarks"].keys()) + ["Macro Mean"]
    
    # Mode 1 Accuracies
    m1_base = [data["benchmarks"][b]["mode1_cold_start"]["base_accuracy"] for b in data["benchmarks"]] + [data["macro_summary"]["mode1_cold_start"]["base_accuracy"]]
    m1_normal = [data["benchmarks"][b]["mode1_cold_start"]["dualloop_normal_accuracy"] for b in data["benchmarks"]] + [data["macro_summary"]["mode1_cold_start"]["dualloop_normal_accuracy"]]
    m1_wl = [data["benchmarks"][b]["mode1_cold_start"]["dualloop_wronglog_accuracy"] for b in data["benchmarks"]] + [data["macro_summary"]["mode1_cold_start"]["dualloop_wronglog_accuracy"]]

    # Mode 2 Accuracies
    m2_base = [data["benchmarks"][b]["mode2_wronglog_active"]["base_re_accuracy"] for b in data["benchmarks"]] + [data["macro_summary"]["mode2_wronglog_active"]["base_accuracy"]]
    m2_normal = [data["benchmarks"][b]["mode2_wronglog_active"]["dualloop_normal_re_accuracy"] for b in data["benchmarks"]] + [data["macro_summary"]["mode2_wronglog_active"]["dualloop_normal_accuracy"]]
    m2_wl = [data["benchmarks"][b]["mode2_wronglog_active"]["dualloop_wronglog_accuracy"] for b in data["benchmarks"]] + [data["macro_summary"]["mode2_wronglog_active"]["dualloop_wronglog_accuracy"]]

    x = np.arange(len(b_keys))
    width = 0.26

    # -------------------------------------------------------------
    # Panel A: Test 1 - Bench Awal Biasa (Cold-Start Single Pass)
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#1e293b")
    r1 = ax_a.bar(x - width, m1_base, width, label="Base Qwen3.5-2B (K=0)", color="#64748b", edgecolor="#94a3b8")
    r2 = ax_a.bar(x, m1_normal, width, label="Dual-Loop Normal (K=2)", color="#38bdf8", edgecolor="#7dd3fc")
    r3 = ax_a.bar(x + width, m1_wl, width, label="Dual-Loop x Matrix Helper", color="#10b981", edgecolor="#34d399")

    ax_a.set_title("A. Test 1: Standard Cold-Start Benchmark (Single-Pass, Zero Error History)", fontsize=11, fontweight="bold", color="#f8fafc", pad=10)
    ax_a.set_ylabel("Accuracy (%)", fontsize=10, color="#cbd5e1")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(b_keys, fontsize=9, color="#f1f5f9")
    ax_a.set_ylim(0, 105)
    ax_a.grid(True, linestyle="--", alpha=0.2, color="#94a3b8", axis="y")
    ax_a.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=8)

    for rects, col in [(r1, "#cbd5e1"), (r2, "#7dd3fc"), (r3, "#34d399")]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax_a.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                              textcoords="offset points", ha="center", va="bottom", fontsize=8, color=col, fontweight="bold")

    # -------------------------------------------------------------
    # Panel B: Test 2 - Bench WrongLog Active (Adaptive Session)
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#1e293b")
    rb1 = ax_b.bar(x - width, m2_base, width, label="Base + Wrong Log Filter", color="#f59e0b", edgecolor="#fbbf24")
    rb2 = ax_b.bar(x, m2_normal, width, label="Dual-Loop Normal (No Wrong Log)", color="#38bdf8", edgecolor="#7dd3fc")
    rb3 = ax_b.bar(x + width, m2_wl, width, label="Dual-Loop x Wrong Log Active", color="#a855f7", edgecolor="#c084fc")

    ax_b.set_title("B. Test 2: Wrong Log Active Benchmark (Adaptive Memory Session)", fontsize=11, fontweight="bold", color="#f8fafc", pad=10)
    ax_b.set_ylabel("Accuracy (%)", fontsize=10, color="#cbd5e1")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(b_keys, fontsize=9, color="#f1f5f9")
    ax_b.set_ylim(0, 105)
    ax_b.grid(True, linestyle="--", alpha=0.2, color="#94a3b8", axis="y")
    ax_b.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=8)

    for rects, col in [(rb1, "#fbbf24"), (rb2, "#7dd3fc"), (rb3, "#c084fc")]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax_b.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                              textcoords="offset points", ha="center", va="bottom", fontsize=8, color=col, fontweight="bold")

    # -------------------------------------------------------------
    # Panel C: Direct Side-by-Side Delta (Mode 1 vs Mode 2 for Dual-Loop)
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#1e293b")
    delta_width = 0.35
    rc1 = ax_c.bar(x - delta_width/2, m1_wl, delta_width, label="Mode 1: Cold-Start Dual-Loop", color="#10b981", edgecolor="#34d399")
    rc2 = ax_c.bar(x + delta_width/2, m2_wl, delta_width, label="Mode 2: Wrong Log Active Dual-Loop", color="#a855f7", edgecolor="#c084fc")

    ax_c.set_title("C. Side-by-Side Comparison: Dual-Loop Cold-Start vs Wrong Log Active", fontsize=11, fontweight="bold", color="#f8fafc", pad=10)
    ax_c.set_ylabel("Accuracy (%)", fontsize=10, color="#cbd5e1")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(b_keys, fontsize=9, color="#f1f5f9")
    ax_c.set_ylim(0, 115)
    ax_c.grid(True, linestyle="--", alpha=0.2, color="#94a3b8", axis="y")
    ax_c.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9)

    for rect1, rect2 in zip(rc1, rc2):
        h1 = rect1.get_height()
        h2 = rect2.get_height()
        diff = h2 - h1
        ax_c.annotate(f"{h1:.1f}%", xy=(rect1.get_x() + rect1.get_width()/2, h1), xytext=(0, 2),
                      textcoords="offset points", ha="center", fontsize=8, color="#34d399", fontweight="bold")
        ax_c.annotate(f"{h2:.1f}%\n(+{diff:.1f}%)", xy=(rect2.get_x() + rect2.get_width()/2, h2), xytext=(0, 2),
                      textcoords="offset points", ha="center", fontsize=8, color="#c084fc", fontweight="bold")

    # -------------------------------------------------------------
    # Panel D: Official Authentic Scoreboard Summary Table
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#1e293b")
    ax_d.axis("off")

    table_data = [
        ["Configuration", "Test 1 (Cold-Start)", "Test 2 (WrongLog Active)", "Net Memory Gain"],
        ["Base Qwen3.5-2B (K=0)", f"{data['macro_summary']['mode1_cold_start']['base_accuracy']:.1f}%", f"{data['macro_summary']['mode2_wronglog_active']['base_accuracy']:.1f}%", f"+{data['macro_summary']['mode2_wronglog_active']['base_accuracy'] - data['macro_summary']['mode1_cold_start']['base_accuracy']:.1f}%"],
        ["Dual-Loop Normal (K=2)", f"{data['macro_summary']['mode1_cold_start']['dualloop_normal_accuracy']:.1f}%", f"{data['macro_summary']['mode2_wronglog_active']['dualloop_normal_accuracy']:.1f}%", "0.0% (Static)"],
        ["Dual-Loop x Wrong Log", f"{data['macro_summary']['mode1_cold_start']['dualloop_wronglog_accuracy']:.1f}%", f"{data['macro_summary']['mode2_wronglog_active']['dualloop_wronglog_accuracy']:.1f}%", f"+{data['macro_summary']['mode2_wronglog_active']['dualloop_wronglog_accuracy'] - data['macro_summary']['mode1_cold_start']['dualloop_wronglog_accuracy']:.1f}%"],
    ]

    tbl = ax_d.table(
        cellText=table_data,
        loc="center",
        cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.2)

    # Style header and rows
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#334155")
        if row == 0:
            cell.set_facecolor("#0f172a")
            cell.set_text_props(color="#f8fafc", weight="bold")
        elif row == 3:
            cell.set_facecolor("#2e1065")
            cell.set_text_props(color="#e9d5ff", weight="bold")
        else:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="#f1f5f9")

    ax_d.set_title("D. Official Quantitative Leaderboard Scorecard (N=75 Total Samples)", fontsize=11, fontweight="bold", color="#f8fafc", pad=10)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Successfully generated: {output_png}")

if __name__ == "__main__":
    run_side_by_side_benchmark()
