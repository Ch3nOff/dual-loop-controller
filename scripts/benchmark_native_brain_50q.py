#!/usr/bin/env python3
"""
Native Brain Benchmark: 20 Canonical Benchmarks across 50 Questions
===================================================================
Compares:
1. Model Dasar: Base Qwen/Qwen3.5-2B (100% Frozen Baseline)
2. Model Sebelumnya: X-Star-2B (HADL v4.5 Dual-Cup 27B-Grade Distillation via External Hooks)
3. Model Sekarang: Native Standalone Brain (dist/xstar_2b_native_brain)
   - 100% Native Hugging Face AutoModelForCausalLM
   - ZERO external scripts, ZERO hooks, ZERO custom logits processors!
   - Fused VGNI-LBU Epistemic Memory & HADL 27B-Grade Logic directly in Neural Weights!

Saves:
- JSON: eval_results/benchmark_native_brain_50q_comparison.json
- PNG: eval_results/benchmark_native_brain_50q_comparison.png
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.dual_cup_poly_engine import attach_hadl_v45_dualcup
from scripts.evaluate_20bench_50q import build_20bench_50q_suite

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
NATIVE_BRAIN_DIR = PROJECT_ROOT / "dist" / "xstar_2b_native_brain"
PREV_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "xstar_2b_omnireason_27b_grade_checkpoint.pt"
JSON_OUT = PROJECT_ROOT / "eval_results" / "benchmark_native_brain_50q_comparison.json"
GRAPH_OUT = PROJECT_ROOT / "eval_results" / "benchmark_native_brain_50q_comparison.png"


def sample_50q_across_20bench(full_suite: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = {}
    for it in full_suite:
        b = it["benchmark"]
        if b not in grouped:
            grouped[b] = []
        grouped[b].append(it)

    three_q_benchmarks = {
        "DROP", "MMLU", "AGIEval", "TriviaQA", "ARC-c",
        "HumanEval", "ARC-e", "HellaSwag", "PIQA", "BoolQ"
    }

    selected = []
    for b, items in grouped.items():
        take_n = 3 if b in three_q_benchmarks else 2
        selected.extend(items[:take_n])

    assert len(selected) == 50, f"Expected 50 questions, got {len(selected)}"
    return selected


def run_pure_hf_evaluation(
    model: nn.Module,
    tokenizer: Any,
    suite_50q: List[Dict[str, Any]],
    model_type: str
) -> Dict[str, Any]:
    print(f"\n[*] Running Pure HF Evaluation for: {model_type}...")
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    results_by_bench = {}
    details = []
    total_tokens = 0
    correct_count = 0

    for idx, item in enumerate(suite_50q, 1):
        prompt = item["prompt"]
        target = item["target"]
        check_fn = item["check"]
        bench = item["benchmark"]
        pillar = item["pillar"]
        q_id = item["id"]

        inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_len = inp.input_ids.shape[1]
        max_tok = 20 if bench in ["DROP", "MMLU", "ARC-c", "ARC-e", "PIQA"] else 35

        torch.cuda.synchronize()
        t0_q = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **inp,
                max_new_tokens=max_tok,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        t_q = time.perf_counter() - t0_q

        gen_tokens = out_ids.shape[1] - input_len
        total_tokens += gen_tokens

        out_text = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()
        clean_text = re.sub(r"<think>.*?</think>", "", out_text, flags=re.DOTALL).strip()
        is_correct = check_fn(clean_text) or check_fn(out_text)

        if is_correct:
            correct_count += 1

        if bench not in results_by_bench:
            results_by_bench[bench] = {"correct": 0, "total": 0, "pillar": pillar}
        results_by_bench[bench]["total"] += 1
        if is_correct:
            results_by_bench[bench]["correct"] += 1

        details.append({
            "idx": idx,
            "id": q_id,
            "benchmark": bench,
            "pillar": pillar,
            "correct": is_correct,
            "generated_tokens": gen_tokens,
            "latency_sec": round(t_q, 4),
            "tokens_per_sec": round(gen_tokens / (t_q + 1e-9), 2),
            "prediction": clean_text[:60].replace("\n", " "),
            "target": target
        })

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start
    throughput = total_tokens / (total_time + 1e-9)
    acc_pct = (correct_count / len(suite_50q)) * 100.0

    print(f"[+] Completed {model_type}: Score {correct_count}/50 ({acc_pct:.1f}%) | Speed: {throughput:.2f} tok/s | Time: {total_time:.2f}s")

    return {
        "model_type": model_type,
        "score": f"{correct_count}/50",
        "correct_count": correct_count,
        "total_questions": len(suite_50q),
        "accuracy_pct": round(acc_pct, 2),
        "total_tokens_generated": total_tokens,
        "total_latency_sec": round(total_time, 3),
        "mean_tokens_per_sec": round(throughput, 2),
        "mean_latency_per_q": round(total_time / len(suite_50q), 4),
        "results_by_benchmark": results_by_bench,
        "details": details
    }


def main():
    print("=" * 100)
    print("NATIVE BRAIN BENCHMARK: COMPARING PURE NATIVE MODEL AGAINST PREVIOUS & BASE")
    print("=" * 100)

    full_suite = build_20bench_50q_suite()
    suite_50q = sample_50q_across_20bench(full_suite)
    print(f"[*] Prepared 50 Canonical Questions across 20 Benchmarks.")

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR))

    # 1. EVALUATE MODEL DASAR (Base Qwen 2B Frozen)
    print("\n" + "-" * 80)
    print("[*] Loading Model Dasar from:", BASE_MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL_DIR), torch_dtype=torch.bfloat16).to(DEVICE)
    base_model.eval()
    res_base = run_pure_hf_evaluation(base_model, tokenizer, suite_50q, "Model Dasar (Base Qwen3.5-2B)")

    # 2. EVALUATE MODEL SEBELUMNYA (HADL v4.5 External Hooks)
    print("\n" + "-" * 80)
    print("[*] Attaching HADL v4.5 Controller for Model Sebelumnya...")
    hadl_model = attach_hadl_v45_dualcup(base_model=base_model, target_layer_idx=11, ghost_layer_idx=23)
    if PREV_CHECKPOINT.exists():
        ckpt_prev = torch.load(str(PREV_CHECKPOINT), map_location=DEVICE, weights_only=True)
        hadl_model.controller.load_state_dict(ckpt_prev["controller_state_dict"], strict=False)
        print(f"[+] Loaded HADL v4.5 Checkpoint from: {PREV_CHECKPOINT}")
    hadl_model.eval()
    res_prev = run_pure_hf_evaluation(hadl_model, tokenizer, suite_50q, "Model Sebelumnya (HADL v4.5 Dual-Cup)")

    # Clean up base model to free VRAM
    if hadl_model.hook_handle_target is not None:
        hadl_model.hook_handle_target.remove()
    if hadl_model.hook_handle_ghost is not None:
        hadl_model.hook_handle_ghost.remove()
    del hadl_model
    del base_model
    torch.cuda.empty_cache()

    # 3. EVALUATE MODEL SEKARANG (NATIVE STANDALONE BRAIN - ZERO WRAPPERS)
    print("\n" + "-" * 80)
    print("[*] Loading Native Standalone Brain directly from:", NATIVE_BRAIN_DIR)
    native_model = AutoModelForCausalLM.from_pretrained(str(NATIVE_BRAIN_DIR), torch_dtype=torch.bfloat16).to(DEVICE)
    native_model.eval()
    res_native = run_pure_hf_evaluation(native_model, tokenizer, suite_50q, "Model Sekarang (Native Standalone Brain)")

    del native_model
    torch.cuda.empty_cache()

    # 4. PRINT AGGREGATE SUMMARY TABLE
    print("\n" + "=" * 100)
    print("NATIVE BRAIN COMPREHENSIVE BENCHMARK SUMMARY (50 SOAL ACROSS 20 BENCHMARKS)")
    print("=" * 100)
    row_fmt = "{:<36} | {:<12} | {:<10} | {:<12} | {:<12} | {:<12}"
    print(row_fmt.format("Generasi Model", "Skor (Benar)", "Akurasi %", "Total Tokens", "Kecepatan", "Latensi Total"))
    print("-" * 100)
    print(row_fmt.format("1. Model Dasar (Base Frozen)", res_base["score"], f"{res_base['accuracy_pct']}%", f"{res_base['total_tokens_generated']} toks", f"{res_base['mean_tokens_per_sec']} t/s", f"{res_base['total_latency_sec']} s"))
    print(row_fmt.format("2. Model Sebelumnya (HADL v4.5)", res_prev["score"], f"{res_prev['accuracy_pct']}%", f"{res_prev['total_tokens_generated']} toks", f"{res_prev['mean_tokens_per_sec']} t/s", f"{res_prev['total_latency_sec']} s"))
    print(row_fmt.format("3. Model Sekarang (Native Brain)", res_native["score"], f"{res_native['accuracy_pct']}%", f"{res_native['total_tokens_generated']} toks", f"{res_native['mean_tokens_per_sec']} t/s", f"{res_native['total_latency_sec']} s"))
    print("=" * 100)

    # Pillar-wise breakdown
    pillars = ["Math & Quantitative", "Science & Academic", "Coding & Software", "Commonsense & Logic", "Instruction & Chat"]
    pillar_summary = {}
    print("\nBREAKDOWN PER PILLAR KEILMUAN:")
    print("-" * 90)
    print(f"{'Pillar Keilmuan':<26} | {'Model Dasar':<18} | {'Model Sebelumnya':<18} | {'Model Sekarang (Native)':<18}")
    print("-" * 90)
    for pil in pillars:
        b_c = sum(res_base["results_by_benchmark"][b]["correct"] for b in res_base["results_by_benchmark"] if res_base["results_by_benchmark"][b]["pillar"] == pil)
        p_c = sum(res_prev["results_by_benchmark"][b]["correct"] for b in res_prev["results_by_benchmark"] if res_prev["results_by_benchmark"][b]["pillar"] == pil)
        n_c = sum(res_native["results_by_benchmark"][b]["correct"] for b in res_native["results_by_benchmark"] if res_native["results_by_benchmark"][b]["pillar"] == pil)
        tot = sum(res_base["results_by_benchmark"][b]["total"] for b in res_base["results_by_benchmark"] if res_base["results_by_benchmark"][b]["pillar"] == pil)

        pillar_summary[pil] = {
            "total_questions": tot,
            "base_score": f"{b_c}/{tot}", "base_acc": round(b_c/tot*100, 1),
            "prev_score": f"{p_c}/{tot}", "prev_acc": round(p_c/tot*100, 1),
            "native_score": f"{n_c}/{tot}", "native_acc": round(n_c/tot*100, 1)
        }
        print(f"{pil:<26} | {b_c}/{tot} ({b_c/tot*100:.1f}%)         | {p_c}/{tot} ({p_c/tot*100:.1f}%)         | {n_c}/{tot} ({n_c/tot*100:.1f}%)")
    print("-" * 90)

    # Generate Comparative Graphs
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(pillars))
    w = 0.26
    b_scores = [pillar_summary[p]["base_acc"] for p in pillars]
    p_scores = [pillar_summary[p]["prev_acc"] for p in pillars]
    n_scores = [pillar_summary[p]["native_acc"] for p in pillars]

    ax1.bar(x - w, b_scores, width=w, label=f"Model Dasar ({res_base['accuracy_pct']}%)", color="#78909C")
    ax1.bar(x, p_scores, width=w, label=f"Model Sebelumnya ({res_prev['accuracy_pct']}%)", color="#D32F2F")
    ax1.bar(x + w, n_scores, width=w, label=f"Model Sekarang (Native) ({res_native['accuracy_pct']}%)", color="#1B5E20")

    ax1.set_title("Perbandingan Skor Akurasi per Pillar (50 Soal)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace(" & ", "\n& ") for p in pillars], fontsize=9)
    ax1.set_ylabel("Akurasi (%)", fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper right", fontsize=9)

    models_labels = ["Model Dasar", "Model Sebelumnya\n(HADL v4.5)", "Model Sekarang\n(Native Standalone)"]
    speeds = [res_base["mean_tokens_per_sec"], res_prev["mean_tokens_per_sec"], res_native["mean_tokens_per_sec"]]
    latencies = [res_base["total_latency_sec"], res_prev["total_latency_sec"], res_native["total_latency_sec"]]

    bars = ax2.bar(models_labels, speeds, color=["#546E7A", "#E53935", "#2E7D32"], width=0.45)
    ax2.set_title("Throughput Kecepatan Inferensi (tokens/sec)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Tokens / Detik", fontsize=10)
    ax2.set_ylim(0, max(speeds) * 1.25)

    for b, s, lat in zip(bars, speeds, latencies):
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 0.8, f"{s:.2f} tok/s\n({lat:.1f}s)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    GRAPH_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(GRAPH_OUT), dpi=300)
    plt.close()
    print(f"\n[+] Saved Comparison Graph to: {GRAPH_OUT}")

    # Save JSON Audit
    full_audit = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": DEVICE,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "models_evaluated": {
            "model_dasar": res_base,
            "model_sebelumnya": res_prev,
            "model_sekarang_native": res_native
        },
        "pillar_summary": pillar_summary
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(JSON_OUT), "w", encoding="utf-8") as f:
        json.dump(full_audit, f, indent=2)

    print(f"[+] Saved Detailed Audit Report to: {JSON_OUT}")


if __name__ == "__main__":
    main()
