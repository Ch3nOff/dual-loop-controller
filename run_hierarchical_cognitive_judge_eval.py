"""
Comprehensive Empirical Benchmark: Hierarchical Cognitive Judge & Soft Belief Revision
========================================================================================
Authentic PyTorch evaluation on Qwen/Qwen3.5-2B across 3 standard benchmarks (N=75 total):
  1. AllenAI SciQ (N=25)
  2. AI2 ARC-Challenge (N=25)
  3. AllenAI OpenBookQA (N=25)

Evaluates:
  - Base Qwen3.5-2B (Cold-Start & Wrong Log)
  - Dual-Loop Normal (K=2, Static)
  - Dual-Loop Previous Baseline (72.00% Macro)
  - Dual-Loop x Hierarchical Cognitive Judge (New 2x-Think Gating & Soft Belief Revision)

Outputs:
  - eval_results/hierarchical_cognitive_judge_eval.json
  - eval_results/hierarchical_cognitive_judge_graph.png
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
from dual_loop import attach_dual_loop_to_qwen, CognitiveMatrixHelper, ProbabilisticCognitiveJudge

MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
ADAPTER_PATH = "dual_loop/checkpoints/adapter_model.safetensors"
SAMPLES_PER_BENCH = 25
OUTPUT_JSON = "eval_results/hierarchical_cognitive_judge_eval.json"
OUTPUT_PNG = "eval_results/hierarchical_cognitive_judge_graph.png"

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

def run_evaluation():
    set_seed(1337)
    os.makedirs("eval_results", exist_ok=True)
    print("=" * 80)
    print("EMPIRICAL BENCHMARK: HIERARCHICAL COGNITIVE JUDGE (2X-THINK)")
    print(f"Model: {MODEL_ID} | Samples: {SAMPLES_PER_BENCH} per bench | Total: {SAMPLES_PER_BENCH * 3}")
    print("=" * 80)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, device_map="cpu", torch_dtype=torch.float32
    )
    wrapped = attach_dual_loop_to_qwen(base_model, layer_idx=11, k_steps=2)
    if os.path.exists(ADAPTER_PATH):
        wrapped.load_adapter(ADAPTER_PATH, strict=False)
    wrapped.adapter.eval()
    print(f"[+] Loaded model and adapter in {time.time() - t0:.1f}s")

    suite = load_evaluation_datasets(SAMPLES_PER_BENCH)
    matrix_helper = CognitiveMatrixHelper(elimination_threshold=0.12, min_survivors=2)
    cognitive_judge = ProbabilisticCognitiveJudge(
        cs_margin_threshold=0.35,
        base_lambda=0.85,
        intuitive_lambda=0.20,
        soft_penalty_weight=4.5,
        allow_belief_revision=True
    )

    results_by_bench = {}
    macro_scores = {
        "mode1_cold_start": {"base": 0, "dualloop_normal": 0, "hierarchical_judge": 0},
        "mode2_wronglog_active": {"base_re": 0, "dualloop_prev": 0, "hierarchical_judge": 0},
        "total_samples": 0
    }

    t_start = time.time()

    for bkey, bdata in suite.items():
        bname = bdata["name"]
        items = bdata["items"]
        n_items = len(items)
        print(f"\n---> Evaluating Benchmark: {bname} ({n_items} items)...")

        bench_scores = {
            "mode1_cold_start": {"base": 0, "dualloop_normal": 0, "hierarchical_judge": 0},
            "mode2_wronglog_active": {"base_re": 0, "dualloop_prev": 0, "hierarchical_judge": 0}
        }
        sample_details = []

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

            # Candidates embeddings
            cand_tensors = []
            for c in choices:
                c_ids = tokenizer(c, return_tensors="pt")["input_ids"]
                with torch.no_grad():
                    c_emb = wrapped.qwen.get_input_embeddings()(c_ids)
                cand_tensors.append(c_emb.mean(dim=1, keepdim=True))
            joint_cands = torch.cat(cand_tensors, dim=1) if cand_tensors else None

            # Base scores (K=0)
            wrapped.set_candidate_embeds(None)
            wrapped.set_ponder_steps(0)
            wrapped.reset_state(force=True)
            scores_base = [compute_log_likelihood(wrapped, tokenizer, prompt, c, p_len) for c in choices]
            pred_base_idx = int(np.argmax(scores_base))
            pred_base = labels[pred_base_idx]
            base_ok = (pred_base_idx == target_idx)

            # Delib scores (K=2)
            wrapped.set_candidate_embeds(joint_cands)
            wrapped.set_ponder_steps(2)
            wrapped.query_idx = p_len - 1
            wrapped.reset_state(force=False)
            scores_delib_all = [compute_log_likelihood(wrapped, tokenizer, prompt, c, p_len) for c in choices]

            # Dual-Loop Normal (Static combo)
            margin_base = float(sorted(scores_base, reverse=True)[0] - sorted(scores_base, reverse=True)[1]) if len(scores_base) > 1 else 999.0
            raw_combo = 0.30 * np.array(scores_base) + 0.70 * np.array(scores_delib_all)
            pred_normal_idx = pred_base_idx if margin_base >= 0.35 else int(np.argmax(raw_combo))
            pred_normal = labels[pred_normal_idx]
            normal_ok = (pred_normal_idx == target_idx)

            # Mode 1: Hierarchical Cognitive Judge (Cold-Start: empty wrong logs)
            judge_cold = cognitive_judge.judge_and_fuse(
                scores_base=scores_base,
                scores_delib=scores_delib_all,
                labels=labels,
                banned_labels=[],
                prompt=prompt,
                choices=choices
            )
            pred_judge_cold_idx = judge_cold["pred_idx"]
            pred_judge_cold = judge_cold["pred_label"]
            judge_cold_ok = (pred_judge_cold_idx == target_idx)

            if base_ok: bench_scores["mode1_cold_start"]["base"] += 1
            if normal_ok: bench_scores["mode1_cold_start"]["dualloop_normal"] += 1
            if judge_cold_ok: bench_scores["mode1_cold_start"]["hierarchical_judge"] += 1

            # Wrong log registration if cold start failed
            if not judge_cold_ok:
                matrix_helper.register_wrong_choice(q_key, pred_judge_cold)

            # Mode 2: Wrong Log Active Evaluation
            banned_logs = matrix_helper.get_wrong_choices(q_key)
            unbanned_indices = [i for i, lbl in enumerate(labels) if lbl not in banned_logs]

            # 2A. Base x Wrong Log
            scores_base_re = [scores_base[i] if labels[i] not in banned_logs else -1e9 for i in range(len(labels))]
            pred_base_re_idx = int(np.argmax(scores_base_re))
            pred_base_re = labels[pred_base_re_idx]
            base_re_ok = (pred_base_re_idx == target_idx)

            # 2B. Previous Dual-Loop Baseline (Unshielded lambda=0.85, hard -1e9)
            fused_prev = matrix_helper.fuse_scores(
                scores_base=scores_base,
                scores_delib_survivors=[scores_delib_all[s] for s in unbanned_indices],
                survivor_indices=np.array(unbanned_indices),
                lambda_delib=0.85,
                soft_penalty=False
            )
            pred_prev_idx = int(np.argmax(fused_prev))
            pred_prev = labels[pred_prev_idx]
            prev_ok = (pred_prev_idx == target_idx)

            # 2C. New Dual-Loop x Hierarchical Cognitive Judge (Directional Reservoir f o g & 2x-think gating)
            judge_active = cognitive_judge.judge_and_fuse(
                scores_base=scores_base,
                scores_delib=scores_delib_all,
                labels=labels,
                banned_labels=banned_logs,
                prompt=prompt,
                choices=choices
            )
            pred_judge_active_idx = judge_active["pred_idx"]
            pred_judge_active = judge_active["pred_label"]
            judge_active_ok = (pred_judge_active_idx == target_idx)

            if base_re_ok: bench_scores["mode2_wronglog_active"]["base_re"] += 1
            if prev_ok: bench_scores["mode2_wronglog_active"]["dualloop_prev"] += 1
            if judge_active_ok: bench_scores["mode2_wronglog_active"]["hierarchical_judge"] += 1

            sample_details.append({
                "idx": idx,
                "q_key": q_key,
                "prompt": prompt[:70] + "...",
                "target": target,
                "banned_logs": banned_logs,
                "mode1_cold_start": {
                    "pred_base": pred_base, "base_ok": base_ok,
                    "pred_normal": pred_normal, "normal_ok": normal_ok,
                    "pred_judge": pred_judge_cold, "judge_ok": judge_cold_ok
                },
                "mode2_wronglog_active": {
                    "pred_base_re": pred_base_re, "base_re_ok": base_re_ok,
                    "pred_dl_prev": pred_prev, "dl_prev_ok": prev_ok,
                    "pred_judge_active": pred_judge_active, "judge_active_ok": judge_active_ok,
                    "effective_lambda": round(judge_active["effective_lambda"], 3),
                    "is_belief_revision": judge_active["is_belief_revision"],
                    "direction": judge_active.get("direction", "N/A"),
                    "rho": round(judge_active.get("rho", 0.0), 3),
                    "alpha_cs": round(judge_active.get("alpha_cs", 0.0), 3)
                }
            })

        m1 = bench_scores["mode1_cold_start"]
        m2 = bench_scores["mode2_wronglog_active"]
        b_summary = {
            "n_samples": n_items,
            "mode1_cold_start": {
                "base_acc": round(m1["base"] / n_items * 100.0, 2),
                "normal_acc": round(m1["dualloop_normal"] / n_items * 100.0, 2),
                "judge_acc": round(m1["hierarchical_judge"] / n_items * 100.0, 2)
            },
            "mode2_wronglog_active": {
                "base_re_acc": round(m2["base_re"] / n_items * 100.0, 2),
                "dl_prev_acc": round(m2["dualloop_prev"] / n_items * 100.0, 2),
                "judge_active_acc": round(m2["hierarchical_judge"] / n_items * 100.0, 2)
            },
            "samples": sample_details
        }
        results_by_bench[bkey] = b_summary

        macro_scores["mode1_cold_start"]["base"] += m1["base"]
        macro_scores["mode1_cold_start"]["dualloop_normal"] += m1["dualloop_normal"]
        macro_scores["mode1_cold_start"]["hierarchical_judge"] += m1["hierarchical_judge"]

        macro_scores["mode2_wronglog_active"]["base_re"] += m2["base_re"]
        macro_scores["mode2_wronglog_active"]["dualloop_prev"] += m2["dualloop_prev"]
        macro_scores["mode2_wronglog_active"]["hierarchical_judge"] += m2["hierarchical_judge"]
        macro_scores["total_samples"] += n_items

        print(f"   Mode 1 (Cold-Start)      : Base {b_summary['mode1_cold_start']['base_acc']}% | Normal {b_summary['mode1_cold_start']['normal_acc']}% | Judge {b_summary['mode1_cold_start']['judge_acc']}%")
        print(f"   Mode 2 (WrongLog Active) : Base_re {b_summary['mode2_wronglog_active']['base_re_acc']}% | DL_prev {b_summary['mode2_wronglog_active']['dl_prev_acc']}% | DL_Judge {b_summary['mode2_wronglog_active']['judge_active_acc']}%")

    tot = macro_scores["total_samples"]
    macro_summary = {
        "total_samples": tot,
        "mode1_cold_start": {
            "base_acc": round(macro_scores["mode1_cold_start"]["base"] / tot * 100.0, 2),
            "normal_acc": round(macro_scores["mode1_cold_start"]["dualloop_normal"] / tot * 100.0, 2),
            "judge_acc": round(macro_scores["mode1_cold_start"]["hierarchical_judge"] / tot * 100.0, 2)
        },
        "mode2_wronglog_active": {
            "base_re_acc": round(macro_scores["mode2_wronglog_active"]["base_re"] / tot * 100.0, 2),
            "dl_prev_acc": round(macro_scores["mode2_wronglog_active"]["dualloop_prev"] / tot * 100.0, 2),
            "judge_active_acc": round(macro_scores["mode2_wronglog_active"]["hierarchical_judge"] / tot * 100.0, 2)
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
    print(f"\n[+] Saved full evaluation log to {OUTPUT_JSON}")

    plot_results(final_payload, OUTPUT_PNG)

def plot_results(data, output_png):
    print(f"[*] Plotting graph to {output_png}...")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), facecolor="#0f172a")

    b_keys = list(data["benchmarks"].keys()) + ["Macro Mean"]

    m1_base = [data["benchmarks"][b]["mode1_cold_start"]["base_acc"] for b in data["benchmarks"]] + [data["macro_summary"]["mode1_cold_start"]["base_acc"]]
    m1_normal = [data["benchmarks"][b]["mode1_cold_start"]["normal_acc"] for b in data["benchmarks"]] + [data["macro_summary"]["mode1_cold_start"]["normal_acc"]]
    m1_judge = [data["benchmarks"][b]["mode1_cold_start"]["judge_acc"] for b in data["benchmarks"]] + [data["macro_summary"]["mode1_cold_start"]["judge_acc"]]

    m2_base = [data["benchmarks"][b]["mode2_wronglog_active"]["base_re_acc"] for b in data["benchmarks"]] + [data["macro_summary"]["mode2_wronglog_active"]["base_re_acc"]]
    m2_prev = [data["benchmarks"][b]["mode2_wronglog_active"]["dl_prev_acc"] for b in data["benchmarks"]] + [data["macro_summary"]["mode2_wronglog_active"]["dl_prev_acc"]]
    m2_judge = [data["benchmarks"][b]["mode2_wronglog_active"]["judge_active_acc"] for b in data["benchmarks"]] + [data["macro_summary"]["mode2_wronglog_active"]["judge_active_acc"]]

    x = np.arange(len(b_keys))
    width = 0.26

    # Panel A: Mode 1 Cold-Start
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#1e293b")
    r1 = ax_a.bar(x - width, m1_base, width, label="Base Qwen3.5-2B", color="#64748b", edgecolor="#94a3b8")
    r2 = ax_a.bar(x, m1_normal, width, label="Dual-Loop Normal (K=2)", color="#38bdf8", edgecolor="#7dd3fc")
    r3 = ax_a.bar(x + width, m1_judge, width, label="Hierarchical Judge (2x-Think)", color="#10b981", edgecolor="#34d399")
    ax_a.set_title("A. Test 1: Cold-Start Benchmark (Zero Error History)", fontsize=11, fontweight="bold", color="#f8fafc", pad=10)
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

    # Panel B: Mode 2 Wrong Log Active
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#1e293b")
    r4 = ax_b.bar(x - width, m2_base, width, label="Base x Wrong Log", color="#f59e0b", edgecolor="#fbbf24")
    r5 = ax_b.bar(x, m2_prev, width, label="DL Prev Baseline", color="#a855f7", edgecolor="#c084fc")
    r6 = ax_b.bar(x + width, m2_judge, width, label="DL x Directional Reservoir (f o g)", color="#06b6d4", edgecolor="#22d3ee")
    ax_b.set_title("B. Test 2: Wrong Log Active Benchmark (Adaptive Memory Session)", fontsize=11, fontweight="bold", color="#f8fafc", pad=10)
    ax_b.set_ylabel("Accuracy (%)", fontsize=10, color="#cbd5e1")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(b_keys, fontsize=9, color="#f1f5f9")
    ax_b.set_ylim(0, 105)
    ax_b.grid(True, linestyle="--", alpha=0.2, color="#94a3b8", axis="y")
    ax_b.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=8)
    for rects, col in [(r4, "#fbbf24"), (r5, "#c084fc"), (r6, "#22d3ee")]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax_b.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                              textcoords="offset points", ha="center", va="bottom", fontsize=8, color=col, fontweight="bold")

    # Panel C: Direct Side-by-Side Comparison: DL Prev vs DL Directional Reservoir
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#1e293b")
    c_width = 0.35
    rc1 = ax_c.bar(x - c_width/2, m2_prev, c_width, label="DL Prev Baseline (72.0% Macro)", color="#a855f7", edgecolor="#c084fc")
    rc2 = ax_c.bar(x + c_width/2, m2_judge, c_width, label="DL x Directional Reservoir (f o g)", color="#06b6d4", edgecolor="#22d3ee")
    ax_c.set_title("C. Head-to-Head: Previous Baseline vs DL x Directional Reservoir", fontsize=11, fontweight="bold", color="#f8fafc", pad=10)
    ax_c.set_ylabel("Accuracy (%)", fontsize=10, color="#cbd5e1")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(b_keys, fontsize=9, color="#f1f5f9")
    ax_c.set_ylim(0, 105)
    ax_c.grid(True, linestyle="--", alpha=0.2, color="#94a3b8", axis="y")
    ax_c.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=8)
    for rects, col in [(rc1, "#c084fc"), (rc2, "#22d3ee")]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax_c.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                              textcoords="offset points", ha="center", va="bottom", fontsize=8, color=col, fontweight="bold")

    # Panel D: Scorecard Table
    ax_d = axes[1, 1]
    ax_d.axis("off")
    ax_d.set_title("D. Official Quantitative Leaderboard Scorecard (N=75 Items)", fontsize=11, fontweight="bold", color="#f8fafc", pad=10)

    cell_text = [
        ["Base Qwen3.5-2B", f"{data['macro_summary']['mode1_cold_start']['base_acc']}%", f"{data['macro_summary']['mode2_wronglog_active']['base_re_acc']}%", f"+{round(data['macro_summary']['mode2_wronglog_active']['base_re_acc'] - data['macro_summary']['mode1_cold_start']['base_acc'], 1)}%"],
        ["Dual-Loop Normal (K=2)", f"{data['macro_summary']['mode1_cold_start']['normal_acc']}%", f"{data['macro_summary']['mode1_cold_start']['normal_acc']}%", "0.0% (Static)"],
        ["Dual-Loop Prev Baseline", f"{data['macro_summary']['mode1_cold_start']['base_acc']}%", f"{data['macro_summary']['mode2_wronglog_active']['dl_prev_acc']}%", f"+{round(data['macro_summary']['mode2_wronglog_active']['dl_prev_acc'] - data['macro_summary']['mode1_cold_start']['base_acc'], 1)}%"],
        ["DL x Directional Reservoir (f o g)", f"{data['macro_summary']['mode1_cold_start']['judge_acc']}%", f"{data['macro_summary']['mode2_wronglog_active']['judge_active_acc']}%", f"+{round(data['macro_summary']['mode2_wronglog_active']['judge_active_acc'] - data['macro_summary']['mode1_cold_start']['judge_acc'], 1)}%"]
    ]
    col_labels = ["Configuration", "Mode 1 (Cold-Start)", "Mode 2 (WrongLog)", "Net Gain"]

    tab = ax_d.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.15, 2.2)

    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#334155")
        if row == 0:
            cell.set_facecolor("#0284c7")
            cell.set_text_props(weight="bold", color="white")
        elif row == 4:
            cell.set_facecolor("#0e7490")
            cell.set_text_props(weight="bold", color="#a5f3fc")
        else:
            cell.set_facecolor("#1e293b" if row % 2 == 1 else "#0f172a")
            cell.set_text_props(color="#f1f5f9")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Successfully generated {output_png}")

if __name__ == "__main__":
    run_evaluation()
