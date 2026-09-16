"""
Authentic Multi-Task Empirical Benchmark Runner: Qwen3.5-2B
============================================================
Evaluates Qwen3.5-2B across 4 real reasoning benchmarks:
  1. arc_easy
  2. arc_challenge
  3. openbookqa
  4. piqa
Zero mockups, zero predictions. 100% EleutherAI lm-eval execution.
"""

import os
import sys
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from dual_loop import attach_dual_loop_to_qwen

MODEL_ID = "Qwen/Qwen3.5-2B"
ADAPTER_PATH = "dual_loop/checkpoints/qwen35_2b_deliberation_adapter.pt"
OUTPUT_DIR = "eval_results"
TASKS = ["arc_easy", "arc_challenge", "openbookqa", "piqa"]
LIMIT = 40  # 40 samples per task across 4 tasks = 160 samples (640 forward queries)

os.makedirs(OUTPUT_DIR, exist_ok=True)

import argparse

parser = argparse.ArgumentParser(description="Empirical Benchmark Runner")
parser.add_argument("--trust_remote_code", action="store_true", default=False, help="Allow executing remote code")
args, _ = parser.parse_known_args()

print("=" * 80)
print(" AUTHENTIC MULTI-TASK BENCHMARK: QWEN3.5-2B + DUAL-LOOP CONTROLLER")
print("=" * 80)
print(f"Model ID:      {MODEL_ID}")
print(f"Adapter:       {ADAPTER_PATH}")
print(f"Tasks:         {TASKS}")
print(f"Limit:         {LIMIT} samples per task")
print("=" * 80)
sys.stdout.flush()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=args.trust_remote_code)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("\n[Stage 1/3] Loading Qwen3.5-2B base model into CPU memory...")
sys.stdout.flush()
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    trust_remote_code=args.trust_remote_code,
    device_map="cpu"
)
print("[Stage 1/3] Base model loaded!")
sys.stdout.flush()

# Attach Dual-Loop at Layer 11 (Full Attention)
print("\n[Stage 2/3] Attaching Dual-Loop Controller (Layer 11) and loading trained weights...")
dualloop_model = attach_dual_loop_to_qwen(base_model, layer_idx=11, k_steps=2)
dualloop_model.load_adapter(ADAPTER_PATH)
gate_scale = float(torch.tanh(dualloop_model.adapter.gate_alpha).item())
print(f"[Stage 2/3] Adapter attached! ReZero Gate Scale: {gate_scale:.4f}")
sys.stdout.flush()

# --- EVALUATION 1: Dual-Loop Augmented (K=2) ---
print("\n" + "=" * 80)
print(" STAGE 3A: EVALUATING DUAL-LOOP AUGMENTED MODEL (K=2)")
print("=" * 80)
sys.stdout.flush()

hflm_dualloop = HFLM(pretrained=dualloop_model, tokenizer=tokenizer, batch_size=1, device="cpu")
t0_k2 = time.time()
results_k2 = simple_evaluate(model=hflm_dualloop, tasks=TASKS, limit=LIMIT)
elapsed_k2 = time.time() - t0_k2

path_k2 = os.path.join(OUTPUT_DIR, "qwen35_2b_full_dualloop_k2.json")
with open(path_k2, "w", encoding="utf-8") as f:
    json.dump(results_k2, f, indent=2, default=str)
print(f"[+] K=2 Multi-Task Evaluation Finished in {elapsed_k2:.1f}s.")
print(f"    Results written to: {path_k2}")
sys.stdout.flush()

# --- EVALUATION 2: Base Model (K=0 Identity Bypass) ---
print("\n" + "=" * 80)
print(" STAGE 3B: EVALUATING BASE MODEL (K=0 IDENTITY BYPASS)")
print("=" * 80)
sys.stdout.flush()

dualloop_model.set_ponder_steps(0)
hflm_base = HFLM(pretrained=dualloop_model, tokenizer=tokenizer, batch_size=1, device="cpu")
t0_k0 = time.time()
results_k0 = simple_evaluate(model=hflm_base, tasks=TASKS, limit=LIMIT)
elapsed_k0 = time.time() - t0_k0

path_k0 = os.path.join(OUTPUT_DIR, "qwen35_2b_full_base_k0.json")
with open(path_k0, "w", encoding="utf-8") as f:
    json.dump(results_k0, f, indent=2, default=str)
print(f"[+] K=0 Multi-Task Evaluation Finished in {elapsed_k0:.1f}s.")
print(f"    Results written to: {path_k0}")
sys.stdout.flush()

# --- SUMMARY SCORECARD ---
print("\n" + "=" * 80)
print(" MULTI-TASK AUTHENTIC EMPIRICAL BENCHMARK SUMMARY")
print("=" * 80)
for task in TASKS:
    k0_acc = results_k0["results"][task]["acc,none"] * 100.0
    k2_acc = results_k2["results"][task]["acc,none"] * 100.0
    k0_norm = results_k0["results"][task].get("acc_norm,none", k0_acc / 100.0) * 100.0
    k2_norm = results_k2["results"][task].get("acc_norm,none", k2_acc / 100.0) * 100.0
    delta_acc = k2_acc - k0_acc
    delta_norm = k2_norm - k0_norm
    print(f"Task: {task:<15} | Base (K=0): {k0_acc:5.1f}% (norm {k0_norm:5.1f}%) | Dual-Loop (K=2): {k2_acc:5.1f}% (norm {k2_norm:5.1f}%) | Delta: {delta_acc:+5.1f}% (norm {delta_norm:+5.1f}%)")
print("=" * 80)
