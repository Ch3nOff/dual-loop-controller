#!/usr/bin/env python3
"""
Evaluate Autonomous Self-Evolving Native Brain on 50 Canonical Questions (20 Benchmarks)
========================================================================================
Tests dist/xstar_2b_autonomous_brain using:
- 100% Pure AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
- ZERO external hooks, ZERO external wrappers, ZERO custom logits processors!
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_20bench_50q import build_20bench_50q_suite
from scripts.benchmark_native_brain_50q import sample_50q_across_20bench

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_DIR = PROJECT_ROOT / "dist" / "xstar_2b_autonomous_brain"
JSON_OUT = PROJECT_ROOT / "eval_results" / "benchmark_autonomous_brain_50q.json"

print(f"[*] Loading Autonomous Model from: {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(DEVICE)
model.eval()

full_suite = build_20bench_50q_suite()
suite_50q = sample_50q_across_20bench(full_suite)
print(f"[*] Prepared 50 Canonical Questions across 20 Benchmarks.")

results_by_bench = {}
details = []
total_tokens = 0
correct_count = 0

torch.cuda.synchronize()
t_start = time.perf_counter()

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

print("\n" + "=" * 80)
print(f"[+] Autonomous Model Score: {correct_count}/50 ({acc_pct:.1f}%)")
print(f"[+] Throughput: {throughput:.2f} tok/s | Total Time: {total_time:.2f}s | Generated Tokens: {total_tokens}")
print("=" * 80)

# Pillar breakdown
pillars = ["Math & Quantitative", "Science & Academic", "Coding & Software", "Commonsense & Logic", "Instruction & Chat"]
print("\nBREAKDOWN PER PILLAR KEILMUAN:")
print("-" * 60)
for pil in pillars:
    c = sum(results_by_bench[b]["correct"] for b in results_by_bench if results_by_bench[b]["pillar"] == pil)
    tot = sum(results_by_bench[b]["total"] for b in results_by_bench if results_by_bench[b]["pillar"] == pil)
    print(f"{pil:<24} | {c}/{tot} ({c/tot*100:.1f}%)")
print("-" * 60)

res = {
    "model_name": "XStar-2B-Autonomous-Brain",
    "score": f"{correct_count}/50",
    "accuracy_pct": round(acc_pct, 2),
    "throughput_tok_per_sec": round(throughput, 2),
    "total_time_sec": round(total_time, 2),
    "total_tokens": total_tokens,
    "results_by_benchmark": results_by_bench,
    "details": details
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"[+] Saved results to {JSON_OUT}")
