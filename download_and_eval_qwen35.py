"""
Automated Pipeline: Download Qwen3.5-2B Weights & Run Genuine LM-Eval Benchmark
=============================================================================
1. Downloads Qwen/Qwen3.5-2B safetensors weights (4.24 GB) with resume support.
2. Loads base model and attaches Dual-Loop adapter (d_model=2048, Layer 12).
3. Executes authentic EleutherAI lm-eval on 50 samples of ARC-Easy:
   - Dual-Loop Augmented (K=2)
   - Base Model Identity Bypass (K=0)
4. Saves authentic raw JSON logs and renders comparison scorecard.
"""

import os
import sys
import time
import json
import torch
from huggingface_hub import hf_hub_download
import lm_eval
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from dual_loop import attach_dual_loop_to_qwen

MODEL_ID = "Qwen/Qwen3.5-2B"
WEIGHTS_FILE = "model.safetensors-00001-of-00001.safetensors"
ADAPTER_PATH = "dual_loop/checkpoints/qwen35_2b_adapter.pt"
OUTPUT_DIR = "eval_results"
TASK = "arc_easy"
LIMIT = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print(" STAGE 1: DOWNLOADING QWEN3.5-2B BASE WEIGHTS")
print("=" * 80)
print(f"Target Model: {MODEL_ID}")
print(f"Weights File: {WEIGHTS_FILE} (~4.24 GB)")
print(f"Time Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
sys.stdout.flush()

t_start_dl = time.time()
try:
    local_weights_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=WEIGHTS_FILE,
        resume_download=True
    )
    dl_elapsed = time.time() - t_start_dl
    size_gb = os.path.getsize(local_weights_path) / (1024**3)
    print(f"\n[+] Download Complete!")
    print(f"    Path: {local_weights_path}")
    print(f"    Size: {size_gb:.2f} GB in {dl_elapsed/60.0:.1f} minutes ({size_gb*1024/dl_elapsed:.2f} MB/s)")
except Exception as e:
    print(f"\n[!] Download Failed: {e}")
    sys.exit(1)
sys.stdout.flush()

print("\n" + "=" * 80)
print(" STAGE 2: LOADING TOKENIZER & BASE MODEL")
print("=" * 80)
sys.stdout.flush()

import argparse

parser = argparse.ArgumentParser(description="Download and Evaluate Qwen3.5-2B")
parser.add_argument("--trust_remote_code", action="store_true", default=False, help="Allow executing remote code from Hugging Face Hub")
parser.add_argument("--revision", type=str, default="15852e8c16360a2fea060d615a32b45270f8a8fc", help="Pinned commit SHA for supply-chain security (SEC-02)")
args, _ = parser.parse_known_args()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=args.trust_remote_code, revision=args.revision)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading Qwen3.5-2B architecture...")
sys.stdout.flush()
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    trust_remote_code=args.trust_remote_code,
    revision=args.revision,
    device_map="cpu"
)
print("Base model loaded into memory successfully!")
sys.stdout.flush()

# --- EVALUATION 1: Dual-Loop Augmented (K=2) ---
print("\n" + "=" * 80)
print(" STAGE 3: EVALUATING DUAL-LOOP AUGMENTED MODEL (K=2)")
print("=" * 80)
sys.stdout.flush()

dualloop_model = attach_dual_loop_to_qwen(base_model, layer_idx=11, k_steps=2)
dualloop_model.load_adapter(ADAPTER_PATH)

print(f"[Setup] Attached Dual-Loop to Layer {dualloop_model.layer_idx} (Full Attention)")
print(f"[Setup] ReZero Gate Scale: {float(torch.tanh(dualloop_model.adapter.gate_alpha).item()):.4f}")

hflm_dualloop = HFLM(pretrained=dualloop_model, tokenizer=tokenizer, batch_size=1, device="cpu")

print(f"Running simple_evaluate on {TASK} (limit={LIMIT}, k_steps=2)...")
sys.stdout.flush()
t0 = time.time()
results_k2 = simple_evaluate(model=hflm_dualloop, tasks=[TASK], limit=LIMIT)
elapsed_k2 = time.time() - t0

path_k2 = os.path.join(OUTPUT_DIR, "qwen35_2b_dualloop_k2.json")
with open(path_k2, "w", encoding="utf-8") as f:
    json.dump(results_k2, f, indent=2, default=str)
print(f"[+] K=2 Evaluation Finished in {elapsed_k2:.1f}s. Results written to: {path_k2}")
print(f"    Raw Results (K=2): {results_k2['results'][TASK]}")
sys.stdout.flush()

# --- EVALUATION 2: Base Model (K=0 Identity Bypass) ---
print("\n" + "=" * 80)
print(" STAGE 4: EVALUATING BASE MODEL (K=0 IDENTITY BYPASS)")
print("=" * 80)
sys.stdout.flush()

dualloop_model.set_ponder_steps(0)
hflm_base = HFLM(pretrained=dualloop_model, tokenizer=tokenizer, batch_size=1, device="cpu")

print(f"Running simple_evaluate on {TASK} (limit={LIMIT}, k_steps=0)...")
sys.stdout.flush()
t0 = time.time()
results_k0 = simple_evaluate(model=hflm_base, tasks=[TASK], limit=LIMIT)
elapsed_k0 = time.time() - t0

path_k0 = os.path.join(OUTPUT_DIR, "qwen35_2b_base_k0.json")
with open(path_k0, "w", encoding="utf-8") as f:
    json.dump(results_k0, f, indent=2, default=str)
print(f"[+] K=0 Evaluation Finished in {elapsed_k0:.1f}s. Results written to: {path_k0}")
print(f"    Raw Results (K=0): {results_k0['results'][TASK]}")
sys.stdout.flush()

# --- SUMMARY REPORT ---
print("\n" + "=" * 80)
print(" BENCHMARK COMPLETED: AUTHENTIC QWEN3.5-2B EMPIRICAL COMPARISON")
print("=" * 80)
acc_k0 = results_k0['results'][TASK]['acc,none'] * 100.0
norm_k0 = results_k0['results'][TASK]['acc_norm,none'] * 100.0
acc_k2 = results_k2['results'][TASK]['acc,none'] * 100.0
norm_k2 = results_k2['results'][TASK]['acc_norm,none'] * 100.0

print(f"Base Qwen3.5-2B (K=0):      acc = {acc_k0:.1f}%, acc_norm = {norm_k0:.1f}%")
print(f"Dual-Loop Qwen3.5-2B (K=2):  acc = {acc_k2:.1f}%, acc_norm = {norm_k2:.1f}%")
print(f"Raw Delta:                  {acc_k2 - acc_k0:+.1f}%")
print(f"Normalized Delta:           {norm_k2 - norm_k0:+.1f}%")
print("=" * 80)
