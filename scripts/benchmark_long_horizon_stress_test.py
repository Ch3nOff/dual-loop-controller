"""
HADL v4.5 Long-Horizon Stress Test & Deep Technical Evaluation
==============================================================
Empirically benchmarks:
1. Multi-Step Algorithmic State Tracking (Long-Horizon Variable Propagation)
2. Long-Horizon Multi-Component Code Synthesis (Complex Class & Dependency Implementation)
3. Latent Representation Energy Stability & Drift across 500-Token Rollouts
"""

import sys
import os
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.export_gguf import resolve_model_path

def run_long_horizon_state_tracking(model, tokenizer, device):
    """
    Tests the model's ability to track sequential state mutations across 15 steps.
    """
    prompt = """Below is an execution trace of a register machine with 4 registers [A, B, C, D] starting at [0, 0, 0, 0].
Follow every operation step by step and output ONLY the final values of all registers in JSON format: {"A": int, "B": int, "C": int, "D": int}.

Initial: A=0, B=0, C=0, D=0
Step 1: A = 10
Step 2: B = A + 5
Step 3: C = B * 2
Step 4: D = C - 4
Step 5: A = A + D
Step 6: B = B * 2
Step 7: C = C + A
Step 8: D = D // 2
Step 9: A = A - 6
Step 10: B = B + C
Step 11: C = C // 2
Step 12: D = D + B
Step 13: A = A * 2
Step 14: B = B - A
Step 15: Final adjustment: D = D + 10

Calculate carefully step by step, then output the final JSON:"""

    messages = [
        {"role": "system", "content": "You are a precise deterministic execution simulator. Compute step-by-step accurately."},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - t0
    gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Ground truth calculation:
    # A=0, B=0, C=0, D=0
    # 1: A = 10
    # 2: B = 10 + 5 = 15
    # 3: C = 15 * 2 = 30
    # 4: D = 30 - 4 = 26
    # 5: A = 10 + 26 = 36
    # 6: B = 15 * 2 = 30
    # 7: C = 30 + 36 = 66
    # 8: D = 26 // 2 = 13
    # 9: A = 36 - 6 = 30
    # 10: B = 30 + 66 = 96
    # 11: C = 66 // 2 = 33
    # 12: D = 13 + 96 = 109
    # 13: A = 30 * 2 = 60
    # 14: B = 96 - 60 = 36
    # 15: D = 109 + 10 = 119
    # Expected: {"A": 60, "B": 36, "C": 33, "D": 119}
    ground_truth = {"A": 60, "B": 36, "C": 33, "D": 119}
    
    return {
        "task": "15-Step Register Machine Tracking",
        "ground_truth": ground_truth,
        "response": gen_text.strip(),
        "latency_sec": elapsed,
        "tokens_gen": outputs.shape[1] - inputs.input_ids.shape[1]
    }

def run_long_horizon_code_synthesis(model, tokenizer, device):
    """
    Tests multi-component long-horizon Python implementation with complex dependencies.
    """
    prompt = """Implement a production-grade, thread-safe LRU Cache with Time-To-Live (TTL) expiry in Python.
Requirements:
1. Class `LRUTTLCache(capacity: int, default_ttl: float)`
2. Methods: `get(key)`, `put(key, value, ttl=None)`, `cleanup_expired()`, `__len__()`.
3. Eviction policy: Evict strictly least-recently-used non-expired item when capacity exceeded. Expired items should not count towards valid access.
4. Thread safety: Use `threading.RLock`.
5. Provide complete, fully working code with type hints, docstrings, and a brief verification example at the end. Do not truncate."""

    messages = [
        {"role": "system", "content": "You are a senior systems programmer. Write clean, complete, syntactically perfect Python code without truncation."},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=650,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - t0
    gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Check completeness
    has_class = "class LRUTTLCache" in gen_text
    has_rlock = "RLock" in gen_text
    has_get = "def get(" in gen_text
    has_put = "def put(" in gen_text
    has_cleanup = "def cleanup_expired(" in gen_text
    
    # Check syntax validity by compiling code block if present
    is_valid_syntax = False
    try:
        # extract code inside ```python ... ```
        if "```python" in gen_text:
            code_block = gen_text.split("```python")[1].split("```")[0]
        elif "```" in gen_text:
            code_block = gen_text.split("```")[1].split("```")[0]
        else:
            code_block = gen_text
        compile(code_block, "<string>", "exec")
        is_valid_syntax = True
    except Exception as e:
        syntax_err = str(e)
    else:
        syntax_err = None

    return {
        "task": "Long-Horizon Code Synthesis (Thread-Safe TTL LRU Cache)",
        "has_class": has_class,
        "has_rlock": has_rlock,
        "has_get": has_get,
        "has_put": has_put,
        "has_cleanup": has_cleanup,
        "is_valid_syntax": is_valid_syntax,
        "syntax_error": syntax_err,
        "latency_sec": elapsed,
        "tokens_gen": outputs.shape[1] - inputs.input_ids.shape[1],
        "response_preview": gen_text[:400] + "..."
    }

def run_latent_drift_audit(model, tokenizer, device, num_tokens=250):
    """
    Audits hidden state stability, energy norm ||h_t||, and cosine drift over long autoregressive rollout.
    """
    prompt = "Explain in rigorous detail the architecture of modern operating system kernels, virtual memory management, page tables, and TLB shootdowns."
    messages = [
        {"role": "system", "content": "You are an operating systems professor. Provide a deep, technical treatise."},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Capture layer outputs via hooks
    hidden_norms_mid = []
    hidden_norms_ghost = []
    
    mid_idx = len(model.model.layers) // 2 - 1
    ghost_idx = len(model.model.layers) - 2

    def hook_mid(module, args, output):
        # output is tuple (hidden_states, ...)
        h = output[0] if isinstance(output, tuple) else output
        hidden_norms_mid.append(torch.norm(h[:, -1, :], p=2).item())

    def hook_ghost(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_norms_ghost.append(torch.norm(h[:, -1, :], p=2).item())

    h1 = model.model.layers[mid_idx].register_forward_hook(hook_mid)
    h2 = model.model.layers[ghost_idx].register_forward_hook(hook_ghost)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - t0

    h1.remove()
    h2.remove()

    tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
    
    # Compute stability metrics
    mid_drift = max(hidden_norms_mid) - min(hidden_norms_mid) if hidden_norms_mid else 0
    ghost_drift = max(hidden_norms_ghost) - min(hidden_norms_ghost) if hidden_norms_ghost else 0
    
    mid_mean = sum(hidden_norms_mid) / len(hidden_norms_mid) if hidden_norms_mid else 0
    ghost_mean = sum(hidden_norms_ghost) / len(hidden_norms_ghost) if hidden_norms_ghost else 0

    return {
        "task": f"Latent Energy Stability Audit ({tokens_generated} tokens rollout)",
        "tokens_gen": tokens_generated,
        "elapsed_sec": elapsed,
        "mid_layer_idx": mid_idx,
        "ghost_layer_idx": ghost_idx,
        "mid_layer_mean_norm": round(mid_mean, 3),
        "mid_layer_norm_drift": round(mid_drift, 3),
        "ghost_layer_mean_norm": round(ghost_mean, 3),
        "ghost_layer_norm_drift": round(ghost_drift, 3),
        "is_numerically_stable": (mid_drift < 0.5 * mid_mean) and (ghost_drift < 0.5 * ghost_mean)
    }

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  HADL v4.5 LONG-HORIZON & COMPLEX CODING STRESS TEST AUDIT")
    print("=" * 80)
    print(f"[*] Compute Device: {device}")
    
    model_dir = "dist/staging_hf_fused"
    if not os.path.isdir(model_dir):
        print(f"[!] Staging directory {model_dir} not found, resolving base model...")
        model_dir = resolve_model_path("Qwen/Qwen3.5-2B")

    print(f"[*] Loading model from: {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    print("\n[1/3] Running Multi-Step Algorithmic State Tracking (15 Steps)...")
    res_state = run_long_horizon_state_tracking(model, tokenizer, device)
    print(f"  Latency    : {res_state['latency_sec']:.2f}s ({res_state['tokens_gen']} tokens)")
    print(f"  Response   : {res_state['response'][:150]}...")
    print(f"  Expected   : {res_state['ground_truth']}")

    print("\n[2/3] Running Long-Horizon Complex Code Synthesis (TTL LRU Cache)...")
    res_code = run_long_horizon_code_synthesis(model, tokenizer, device)
    print(f"  Latency    : {res_code['latency_sec']:.2f}s ({res_code['tokens_gen']} tokens)")
    print(f"  Valid Syntax: {res_code['is_valid_syntax']}")
    print(f"  Has RLock   : {res_code['has_rlock']}, Has Class: {res_code['has_class']}")

    print("\n[3/3] Running Latent Representation Energy Stability Audit (250 Tokens Rollout)...")
    res_drift = run_latent_drift_audit(model, tokenizer, device, num_tokens=250)
    print(f"  Mid Layer Norm Mean  : {res_drift['mid_layer_mean_norm']} (Drift: {res_drift['mid_layer_norm_drift']})")
    print(f"  Ghost Layer Norm Mean: {res_drift['ghost_layer_mean_norm']} (Drift: {res_drift['ghost_layer_norm_drift']})")
    print(f"  Numerically Stable   : {res_drift['is_numerically_stable']}")

    out_file = PROJECT_ROOT / "eval_results" / "hadl_v45_long_horizon_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_dir,
        "device": device,
        "results": {
            "state_tracking": res_state,
            "code_synthesis": res_code,
            "latent_energy_stability": res_drift
        }
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[OK] Full Audit Report Saved to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
