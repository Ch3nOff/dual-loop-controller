"""
Dual-Loop Cognitive Controller: Scaled Multi-Step Reasoning Benchmark (N=400)
=============================================================================
Evaluates the core thesis: "Recurrent latent deliberation assists multi-step deduction"
across 4 challenging multi-step domains (N=100 samples each):
  1. BBH - Logical Deduction (Three Objects)
  2. BBH - Date Understanding (Calendar Arithmetic & Reasoning)
  3. BBH - Tracking Shuffled Objects (State Maintenance & Swaps)
  4. ARC-Challenge (Multi-Hop Scientific Reasoning)

Conditions compared:
  - Base Qwen3.5-2B (K=0 feedforward)
  - Static Deliberation (K=2 un-gated recurrent loop)
  - Continuous Soft-Gated Deliberation (K=2 with dynamic uncertainty margin + choice JSD)

All logs and telemetries saved directly to JSON for fully verifiable auditing.
"""

import os
import sys
import time
import json
import re
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dual_loop import attach_dual_loop_to_qwen

OUTPUT_DIR = "eval_results"
DEFAULT_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "qwen35_2b_multistep_n400_eval.json")
MODEL_ID = "Qwen/Qwen3.5-2B"
ADAPTER_PATH = "dual_loop/checkpoints/qwen35_2b_deliberation_adapter.pt"

def parse_bbh_sample(item):
    text = item["input"]
    target_str = item["target"].strip()
    target_clean = target_str.strip("()").strip()
    
    lines = text.split("\n")
    prompt_lines = []
    options = {}
    for line in lines:
        line_str = line.strip()
        opt_match = re.match(r"^\(([A-Z])\)\s*(.*)", line_str)
        if opt_match:
            options[opt_match.group(1)] = opt_match.group(2)
        else:
            prompt_lines.append(line)
            
    prompt = "\n".join(prompt_lines).strip()
    if prompt.endswith("Options:"):
        prompt = prompt[:-8].strip()
        
    labels = list(options.keys())
    choices_text = [options[l] for l in labels]
    return prompt, labels, choices_text, target_clean

def parse_arc_sample(item):
    q = item.get("question", "")
    if not q and "question_stem" in item:
        q = item["question_stem"]
    prompt = str(q).strip()
    labels = [str(l).strip() for l in item["choices"]["label"]]
    choices_text = [str(c).strip() for c in item["choices"]["text"]]
    target = str(item.get("answerKey", "")).strip()
    return prompt, labels, choices_text, target

def compute_wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf((1 + confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - half_width) * 100.0, min(1.0, center + half_width) * 100.0

def mcnemar_exact_pvalue(b, c):
    """b = count where base ok, gated not ok; c = count where base not ok, gated ok"""
    n = b + c
    if n == 0:
        return 1.0
    # Exact two-sided binomial test
    k = min(b, c)
    return float(stats.binomtest(k, n, p=0.5, alternative='two-sided').pvalue)

def main():
    parser = argparse.ArgumentParser(description="Scaled Multi-Step Reasoning Benchmark (N=400)")
    parser.add_argument("--samples_per_task", type=int, default=100)
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_PATH)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f" SCALED MULTI-STEP REASONING BENCHMARK ON QWEN3.5-2B")
    print(f" Samples per task: {args.samples_per_task} | Total tasks: 4 | Total samples: {args.samples_per_task * 4}")
    print(f" Output target: {args.output_json}")
    print(f" Device: {device}")
    print("=" * 80)

    # 1. Load Tokenizer & Base Model
    print("\n[1/3] Loading Qwen3.5-2B backbone...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float32,
        device_map=None
    ).to(device)
    base_model.eval()

    # 2. Attach Dual-Loop Adapter
    print("\n[2/3] Attaching Dual-Loop Adapter (Layer 11 full-attention hook)...")
    wrapped_model = attach_dual_loop_to_qwen(
        base_model,
        layer_idx=11,
        k_steps=2,
        use_surprise_gate=True,
        use_contrastive_evidence=True,
        use_hypothesis_verification=True
    )
    wrapped_model.load_adapter(args.adapter_path, strict=False)
    wrapped_model.eval()

    # 3. Define Task Configurations
    tasks_config = [
        {
            "name": "BBH-LogicalDeduction",
            "type": "bbh",
            "dataset_name": "lukaemon/bbh",
            "subset": "logical_deduction_three_objects",
            "description": "Multi-Step Transitive Deduction & Relational Ordering"
        },
        {
            "name": "BBH-DateUnderstanding",
            "type": "bbh",
            "dataset_name": "lukaemon/bbh",
            "subset": "date_understanding",
            "description": "Calendar Arithmetic & Multi-Hop Temporal Reasoning"
        },
        {
            "name": "BBH-TrackingShuffledObjects",
            "type": "bbh",
            "dataset_name": "lukaemon/bbh",
            "subset": "tracking_shuffled_objects_three_objects",
            "description": "Sequential State Swap Tracking in Working Memory"
        },
        {
            "name": "ARC-Challenge",
            "type": "arc",
            "dataset_name": "allenai/ai2_arc",
            "subset": "ARC-Challenge",
            "description": "Hard Multi-Hop Science QA with Complex Salient Distractors"
        }
    ]

    all_task_results = {}
    macro_base_correct = 0
    macro_static_correct = 0
    macro_gated_correct = 0
    macro_total = 0

    suite_rescued = 0
    suite_degraded = 0

    print("\n[3/3] Executing Scaled Multi-Step Benchmark across tasks...\n")
    start_all = time.time()

    for task_info in tasks_config:
        t_name = task_info["name"]
        t_desc = task_info["description"]
        print(f"---> Evaluating {t_name} (N={args.samples_per_task} samples)...")
        print(f"     Domain: {t_desc}")

        if task_info["type"] == "bbh":
            raw_ds = load_dataset(task_info["dataset_name"], task_info["subset"], split=f"test[:{args.samples_per_task}]")
        else:
            raw_ds = load_dataset(task_info["dataset_name"], task_info["subset"], split=f"test[:{args.samples_per_task}]")

        base_correct = 0
        static_correct = 0
        gated_correct = 0
        valid_samples = 0

        task_gates = []
        task_jsds = []
        task_margins = []
        samples_log = []

        t0_task = time.time()

        for idx, item in enumerate(raw_ds):
            if task_info["type"] == "bbh":
                prompt, labels, choices_text, target = parse_bbh_sample(item)
            else:
                prompt, labels, choices_text, target = parse_arc_sample(item)

            if not prompt or len(choices_text) == 0:
                continue

            target_norm = target.upper().strip()
            if target_norm not in [l.upper().strip() for l in labels]:
                continue

            item_scores = {"base": [], "static": []}
            prompt_str = f"Question: {prompt}\nAnswer:"
            p_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            p_len = len(p_ids)
            query_anchor = p_len - 1

            # Candidate forward passes
            for choice_text in choices_text:
                full_text = f"{prompt_str} {choice_text}"
                enc = tokenizer(full_text, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)

                c_toks = tokenizer.encode(f" {choice_text}", add_special_tokens=False)
                c_len = len(c_toks)
                denom = max(1, c_len)

                # Condition 1: Base Qwen3.5-2B (K=0)
                wrapped_model.set_ponder_steps(0)
                wrapped_model.query_idx = None
                with torch.no_grad():
                    l_base = wrapped_model(input_ids).logits
                sl_base = l_base[:, p_len-1:-1, :]
                slab = input_ids[:, p_len:]
                lp_base = torch.log_softmax(sl_base, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
                norm_base = lp_base.sum().item() / denom
                item_scores["base"].append(norm_base)

                # Condition 2: Static Deliberation (K=2)
                wrapped_model.set_ponder_steps(2)
                wrapped_model.query_idx = query_anchor
                with torch.no_grad():
                    l_static = wrapped_model(input_ids).logits
                sl_static = l_static[:, p_len-1:-1, :]
                lp_static = torch.log_softmax(sl_static, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
                norm_static = lp_static.sum().item() / denom
                item_scores["static"].append(norm_static)

            # Compute choice-level JSD divergence & System 1 margin
            p_b = F.softmax(torch.tensor(item_scores["base"]), dim=-1)
            p_s = F.softmax(torch.tensor(item_scores["static"]), dim=-1)
            m_dist = 0.5 * (p_b + p_s)
            jsd_val = float(0.5 * (F.kl_div(m_dist.log(), p_b, reduction="sum") + F.kl_div(m_dist.log(), p_s, reduction="sum")))

            sorted_b = sorted(item_scores["base"], reverse=True)
            margin_val = float(sorted_b[0] - sorted_b[1]) if len(sorted_b) > 1 else 999.0

            # Condition 3: Continuous Soft-Gated Deliberation
            g_margin = float(torch.sigmoid(torch.tensor((0.10 - margin_val) / 0.05)))
            g_jsd = float(torch.sigmoid(torch.tensor((jsd_val - 0.10) / 0.02)))
            sg_val = max(g_margin, g_jsd) if margin_val < 0.5 else g_jsd

            b_arr = np.array(item_scores["base"])
            s_arr = np.array(item_scores["static"])
            combo_scores = (1.0 - sg_val) * b_arr + sg_val * s_arr
            item_scores["gated"] = combo_scores.tolist()

            base_pred = labels[int(np.argmax(item_scores["base"]))]
            static_pred = labels[int(np.argmax(item_scores["static"]))]
            gated_pred = labels[int(np.argmax(item_scores["gated"]))]

            b_ok = (base_pred.upper().strip() == target_norm)
            s_ok = (static_pred.upper().strip() == target_norm)
            g_ok = (gated_pred.upper().strip() == target_norm)

            if b_ok: base_correct += 1
            if s_ok: static_correct += 1
            if g_ok: gated_correct += 1
            valid_samples += 1

            if not b_ok and g_ok:
                suite_rescued += 1
            elif b_ok and not g_ok:
                suite_degraded += 1

            task_gates.append(sg_val)
            task_jsds.append(jsd_val)
            task_margins.append(margin_val)

            samples_log.append({
                "idx": idx,
                "question": prompt[:120],
                "target": target_norm,
                "base_pred": base_pred,
                "static_pred": static_pred,
                "gated_pred": gated_pred,
                "base_ok": b_ok,
                "static_ok": s_ok,
                "gated_ok": g_ok,
                "surprise_gate": round(sg_val, 6),
                "surprise_jsd": round(jsd_val, 6),
                "margin": round(margin_val, 4)
            })

            if (valid_samples % 20 == 0) or (valid_samples == args.samples_per_task):
                elapsed = time.time() - t0_task
                speed = valid_samples / elapsed
                print(f"  [{t_name}] {valid_samples}/{args.samples_per_task} completed ({speed:.2f} items/s) | Base: {base_correct/valid_samples*100:.1f}% | Gated: {gated_correct/valid_samples*100:.1f}%")

        base_acc = round(base_correct / valid_samples * 100.0, 2)
        static_acc = round(static_correct / valid_samples * 100.0, 2)
        gated_acc = round(gated_correct / valid_samples * 100.0, 2)

        base_ci = compute_wilson_ci(base_correct, valid_samples)
        static_ci = compute_wilson_ci(static_correct, valid_samples)
        gated_ci = compute_wilson_ci(gated_correct, valid_samples)

        b_deg = sum(1 for s in samples_log if s["base_ok"] and not s["gated_ok"])
        b_res = sum(1 for s in samples_log if not s["base_ok"] and s["gated_ok"])
        p_val = mcnemar_exact_pvalue(b_deg, b_res)

        macro_base_correct += base_correct
        macro_static_correct += static_correct
        macro_gated_correct += gated_correct
        macro_total += valid_samples

        all_task_results[t_name] = {
            "description": t_desc,
            "samples": valid_samples,
            "base_acc": base_acc,
            "static_acc": static_acc,
            "gated_acc": gated_acc,
            "delta_static_vs_base": round(static_acc - base_acc, 2),
            "delta_gated_vs_base": round(gated_acc - base_acc, 2),
            "delta_gated_vs_static": round(gated_acc - static_acc, 2),
            "base_ci95": [round(base_ci[0], 2), round(base_ci[1], 2)],
            "static_ci95": [round(static_ci[0], 2), round(static_ci[1], 2)],
            "gated_ci95": [round(gated_ci[0], 2), round(gated_ci[1], 2)],
            "mcnemar_pvalue": round(p_val, 5),
            "rescued_count": b_res,
            "degraded_count": b_deg,
            "mean_surprise_gate": round(float(np.mean(task_gates)), 4),
            "mean_surprise_jsd": round(float(np.mean(task_jsds)), 6),
            "mean_margin": round(float(np.mean(task_margins)), 4),
            "samples_log": samples_log
        }

        print(f"\n  === Results for {t_name} (N={valid_samples}) ===")
        print(f"    Base Qwen3.5-2B (K=0): {base_acc:5.2f}% [95% CI: {base_ci[0]:.1f}% - {base_ci[1]:.1f}%]")
        print(f"    Static Deliberation:   {static_acc:5.2f}% ({static_acc - base_acc:+5.2f}%)")
        print(f"    Continuous Gated:      {gated_acc:5.2f}% ({gated_acc - base_acc:+5.2f}%) [95% CI: {gated_ci[0]:.1f}% - {gated_ci[1]:.1f}%]")
        print(f"    Rescued: {b_res} | Degraded: {b_deg} | McNemar p-value: {p_val:.5f}")
        print(f"    Mean Gate: {np.mean(task_gates):.4f} | Mean JSD: {np.mean(task_jsds):.6f}\n")

    m_base = round(macro_base_correct / macro_total * 100.0, 2)
    m_static = round(macro_static_correct / macro_total * 100.0, 2)
    m_gated = round(macro_gated_correct / macro_total * 100.0, 2)

    total_time = time.time() - start_all
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "total_samples": macro_total,
        "elapsed_seconds": round(total_time, 2),
        "macro_average": {
            "base_acc": m_base,
            "static_acc": m_static,
            "gated_acc": m_gated,
            "delta_static_vs_base": round(m_static - m_base, 2),
            "delta_gated_vs_base": round(m_gated - m_base, 2),
            "total_rescued": suite_rescued,
            "total_degraded": suite_degraded
        },
        "tasks": all_task_results
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("=" * 80)
    print(" SCALED MULTI-STEP BENCHMARK COMPLETE (N=400 TOTAL)")
    print("=" * 80)
    print(f" Macro Base (K=0):             {m_base:.2f}%")
    print(f" Macro Static Deliberation:    {m_static:.2f}% ({m_static - m_base:+5.2f}%)")
    print(f" Macro Continuous Gated:       {m_gated:.2f}% ({m_gated - m_base:+5.2f}%)")
    print(f" Total Rescued: {suite_rescued} questions | Total Degraded: {suite_degraded} questions")
    print(f" Total Execution Time: {total_time/60.0:.2f} minutes")
    print(f"[+] All verifiable evaluation data written to: {args.output_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
