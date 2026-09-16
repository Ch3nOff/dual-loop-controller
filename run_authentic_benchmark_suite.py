"""
Authentic Multi-Task Empirical Benchmark Suite: Qwen3.5-2B
============================================================
Evaluates 160 genuine test samples across 4 core reasoning benchmarks:
  1. ARC-Easy (40 samples)
  2. ARC-Challenge (40 samples)
  3. OpenBookQA (40 samples)
  4. PIQA (40 samples)

Compares:
  - Base Qwen3.5-2B (K=0)
  - Dual-Loop Controller (K=2, Properly Anchored at Question Token)

Measures BOTH Raw Accuracy (acc) and Length-Normalized Accuracy (acc_norm).
Saves authentic per-sample logs and scoreboard JSON.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dual_loop import attach_dual_loop_to_qwen

MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
ADAPTER_PATH = "dual_loop/checkpoints/qwen35_2b_deliberation_adapter.pt"
OUTPUT_DIR = "eval_results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "qwen35_2b_authentic_suite_n160.json")
SAMPLES_PER_TASK = 40

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print(" AUTHENTIC MULTI-TASK BENCHMARK: QWEN3.5-2B + DUAL-LOOP CONTROLLER")
print(f" Samples: {SAMPLES_PER_TASK} per task | Total: {SAMPLES_PER_TASK * 4} samples")
print(f" Adapter: {ADAPTER_PATH}")
print("=" * 80)
sys.stdout.flush()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, revision=REVISION)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("\n[Stage 1/3] Loading Qwen3.5-2B into CPU memory...")
sys.stdout.flush()
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    trust_remote_code=True,
    revision=REVISION,
    device_map="cpu"
)
print("[Stage 1/3] Model loaded successfully.")

wrapped_model = attach_dual_loop_to_qwen(base_model, layer_idx=11, k_steps=2)
wrapped_model.load_adapter(ADAPTER_PATH, strict=False)
gate_scale = float(torch.tanh(wrapped_model.adapter.gate_alpha).item())
print(f"[Stage 2/3] Adapter attached at Layer 11! ReZero Gate Scale: {gate_scale:.4f}")
sys.stdout.flush()

tasks = [
    {
        "name": "ARC-Easy",
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Easy",
        "split": f"test[:{SAMPLES_PER_TASK}]",
        "type": "multiple_choice"
    },
    {
        "name": "ARC-Challenge",
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "split": f"test[:{SAMPLES_PER_TASK}]",
        "type": "multiple_choice"
    },
    {
        "name": "OpenBookQA",
        "dataset": "allenai/openbookqa",
        "subset": "main",
        "split": f"test[:{SAMPLES_PER_TASK}]",
        "type": "multiple_choice"
    },
    {
        "name": "PIQA",
        "dataset": "lighteval/piqa",
        "subset": None,
        "split": f"validation[:{SAMPLES_PER_TASK}]",
        "type": "piqa"
    }
]

suite_results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model_id": MODEL_ID,
    "adapter_path": ADAPTER_PATH,
    "samples_per_task": SAMPLES_PER_TASK,
    "tasks": {},
    "macro_average": {}
}

print("\n[Stage 3/3] Executing Authentic Evaluations across 4 tasks...")
sys.stdout.flush()

for task in tasks:
    task_name = task["name"]
    print(f"\n---> Evaluating {task_name} ({SAMPLES_PER_TASK} samples)...")
    sys.stdout.flush()
    
    if task["subset"]:
        ds = load_dataset(task["dataset"], task["subset"], split=task["split"])
    else:
        ds = load_dataset(task["dataset"], split=task["split"])

    samples_log = []
    base_acc_correct = 0
    base_norm_correct = 0
    loop_acc_correct = 0
    loop_norm_correct = 0
    total_valid = 0

    for i, item in enumerate(ds):
        if task["type"] == "piqa":
            q = item.get("goal", "")
            choices = [item.get("sol1", ""), item.get("sol2", "")]
            labels = ["0", "1"]
            key = str(item.get("label", "")).strip()
            prompt = f"Question: {q}\nAnswer:"
        else:
            q = item.get("question", "")
            choices = item.get("choices", {}).get("text", [])
            labels = item.get("choices", {}).get("label", [])
            key = str(item.get("answerKey", "")).strip()
            prompt = f"Question: {q}\nAnswer:"

        if not choices or not key:
            continue

        prompt_ids = tokenizer(prompt)["input_ids"]
        p_len = len(prompt_ids)
        query_anchor = p_len - 1

        item_scores = {"base_raw": [], "base_norm": [], "loop_raw": [], "loop_norm": []}

        for c in choices:
            full_text = f"{prompt} {c.strip()}"
            inp = tokenizer(full_text, return_tensors="pt")
            input_ids = inp["input_ids"]

            # Base (K=0)
            wrapped_model.set_ponder_steps(0)
            with torch.no_grad():
                l_base = wrapped_model(input_ids).logits
            sl_base = l_base[:, p_len-1:-1, :]
            slab = input_ids[:, p_len:]
            lp_base = torch.log_softmax(sl_base, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
            raw_base = lp_base.sum().item()
            norm_base = raw_base / max(1, slab.shape[1])
            item_scores["base_raw"].append(raw_base)
            item_scores["base_norm"].append(norm_base)

            # Loop (K=2, Anchored at Query Token)
            wrapped_model.set_ponder_steps(2)
            wrapped_model.query_idx = query_anchor
            with torch.no_grad():
                l_loop = wrapped_model(input_ids).logits
            sl_loop = l_loop[:, p_len-1:-1, :]
            lp_loop = torch.log_softmax(sl_loop, dim=-1).gather(-1, slab.unsqueeze(-1)).squeeze(-1)
            raw_loop = lp_loop.sum().item()
            norm_loop = raw_loop / max(1, slab.shape[1])
            item_scores["loop_raw"].append(raw_loop)
            item_scores["loop_norm"].append(norm_loop)

        base_raw_pred = labels[int(np.argmax(item_scores["base_raw"]))]
        base_norm_pred = labels[int(np.argmax(item_scores["base_norm"]))]
        loop_raw_pred = labels[int(np.argmax(item_scores["loop_raw"]))]
        loop_norm_pred = labels[int(np.argmax(item_scores["loop_norm"]))]

        b_raw_ok = (str(base_raw_pred).upper() == key.upper()) or (str(int(np.argmax(item_scores["base_raw"]))) == key)
        b_norm_ok = (str(base_norm_pred).upper() == key.upper()) or (str(int(np.argmax(item_scores["base_norm"]))) == key)
        l_raw_ok = (str(loop_raw_pred).upper() == key.upper()) or (str(int(np.argmax(item_scores["loop_raw"]))) == key)
        l_norm_ok = (str(loop_norm_pred).upper() == key.upper()) or (str(int(np.argmax(item_scores["loop_norm"]))) == key)

        if b_raw_ok: base_acc_correct += 1
        if b_norm_ok: base_norm_correct += 1
        if l_raw_ok: loop_acc_correct += 1
        if l_norm_ok: loop_norm_correct += 1
        total_valid += 1

        samples_log.append({
            "idx": i,
            "question": q,
            "target": key,
            "base_norm_pred": str(base_norm_pred),
            "loop_norm_pred": str(loop_norm_pred),
            "base_norm_ok": b_norm_ok,
            "loop_norm_ok": l_norm_ok,
            "rescued": (not b_norm_ok and l_norm_ok),
            "degraded": (b_norm_ok and not l_norm_ok)
        })

    b_acc_pct = (base_acc_correct / total_valid) * 100.0
    b_norm_pct = (base_norm_correct / total_valid) * 100.0
    l_acc_pct = (loop_acc_correct / total_valid) * 100.0
    l_norm_pct = (loop_norm_correct / total_valid) * 100.0

    delta_raw = l_acc_pct - b_acc_pct
    delta_norm = l_norm_pct - b_norm_pct

    rescued_count = sum(1 for s in samples_log if s["rescued"])
    degraded_count = sum(1 for s in samples_log if s["degraded"])

    suite_results["tasks"][task_name] = {
        "samples": total_valid,
        "base_acc": round(b_acc_pct, 2),
        "loop_acc": round(l_acc_pct, 2),
        "delta_raw": round(delta_raw, 2),
        "base_acc_norm": round(b_norm_pct, 2),
        "loop_acc_norm": round(l_norm_pct, 2),
        "delta_norm": round(delta_norm, 2),
        "rescued_count": rescued_count,
        "degraded_count": degraded_count,
        "samples_detail": samples_log
    }

    print(f"  [{task_name}] RAW ACC:  Base={b_acc_pct:.1f}% -> Loop={l_acc_pct:.1f}% (Delta: {delta_raw:+.1f}%)")
    print(f"  [{task_name}] NORM ACC: Base={b_norm_pct:.1f}% -> Loop={l_norm_pct:.1f}% (Delta: {delta_norm:+.1f}%) | Rescued: {rescued_count} | Degraded: {degraded_count}")
    sys.stdout.flush()

all_tasks = list(suite_results["tasks"].keys())
macro_b_acc = float(np.mean([suite_results["tasks"][t]["base_acc"] for t in all_tasks]))
macro_l_acc = float(np.mean([suite_results["tasks"][t]["loop_acc"] for t in all_tasks]))
macro_b_norm = float(np.mean([suite_results["tasks"][t]["base_acc_norm"] for t in all_tasks]))
macro_l_norm = float(np.mean([suite_results["tasks"][t]["loop_acc_norm"] for t in all_tasks]))

suite_results["macro_average"] = {
    "total_samples": sum(suite_results["tasks"][t]["samples"] for t in all_tasks),
    "base_raw_mean": round(macro_b_acc, 2),
    "loop_raw_mean": round(macro_l_acc, 2),
    "delta_raw_mean": round(macro_l_acc - macro_b_acc, 2),
    "base_norm_mean": round(macro_b_norm, 2),
    "loop_norm_mean": round(macro_l_norm, 2),
    "delta_norm_mean": round(macro_l_norm - macro_b_norm, 2),
    "total_rescued": sum(suite_results["tasks"][t]["rescued_count"] for t in all_tasks),
    "total_degraded": sum(suite_results["tasks"][t]["degraded_count"] for t in all_tasks)
}

print("\n" + "=" * 80)
print(" AUTHENTIC SUITE MACRO RESULTS (N=160 SAMPLES)")
print(f" RAW ACCURACY:        Base={macro_b_acc:.2f}% | Loop={macro_l_acc:.2f}% | Delta: {macro_l_acc - macro_b_acc:+.2f}%")
print(f" NORMALIZED ACCURACY: Base={macro_b_norm:.2f}% | Loop={macro_l_norm:.2f}% | Delta: {macro_l_norm - macro_b_norm:+.2f}%")
print(f" TRANSITIONS:         Total Rescued: {suite_results['macro_average']['total_rescued']} | Total Degraded: {suite_results['macro_average']['total_degraded']}")
print("=" * 80)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(suite_results, f, indent=2)

print(f"\n[+] Verified benchmark results written to: {OUTPUT_JSON}")
