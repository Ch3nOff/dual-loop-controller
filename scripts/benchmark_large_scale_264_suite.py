"""
Large-Scale Authentic Benchmark Suite (264 Tasks):
- 100% Official OpenAI HumanEval (All 164 Problems, Functional Sandbox Execution)
- Official OpenAI GSM8K (100 Multi-Step Math Reasoning Problems, Exact Ground Truth)
===================================================================================
Total Tasks: 264 real benchmark tasks evaluated across Baseline (Frozen 2B) and HADL v4.5.
Zero simulation, zero probe shortcuts, authentic execution on local RTX 5060 GPU.
"""

import sys
import os
import json
import time
import re
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.runtime.dynamic_runtime import HADLDynamicRuntime

MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "xstar_2b_omnireason_carlift_500q_checkpoint.pt"
LOG_OUT = PROJECT_ROOT / "eval_results" / "large_scale_264_benchmark.log"
JSON_OUT = PROJECT_ROOT / "eval_results" / "large_scale_264_benchmark.json"
GRAPH_OUT = PROJECT_ROOT / "docs" / "images" / "hadl_vs_baseline_large_scale_264_benchmark.png"
if sys.platform == "win32":
    ARTIFACT_DIR = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")
else:
    ARTIFACT_DIR = Path("/mnt/c/Users/Matthew Chen/.gemini/antigravity/brain/19bea55e-42a6-476a-af5b-9c25391e2be9")

# ==============================================================================
# EXECUTION SANDBOX HELPERS
# ==============================================================================

def _worker_exec(code: str, q: multiprocessing.Queue):
    try:
        exec_globals = {}
        exec(code, exec_globals)
        q.put((True, None))
    except Exception as e:
        q.put((False, f"{type(e).__name__}: {str(e)}"))

def safe_exec_test(code: str, timeout: float = 3.0) -> Tuple[bool, str]:
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker_exec, args=(code, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "Timeout (>3.0s)"
    if not q.empty():
        return q.get()
    return False, "No result returned"

def clean_humaneval_code(prompt: str, raw_gen: str) -> str:
    if raw_gen.startswith(prompt):
        gen = raw_gen[len(prompt):]
    else:
        gen = raw_gen
    
    # Check if there are markdown blocks
    if "```python" in gen:
        parts = gen.split("```python")
        return parts[1].split("```")[0]
    elif "```" in gen:
        parts = gen.split("```")
        return parts[1].split("```")[0]
    
    # Otherwise, take lines until unindented non-comment definition
    lines = gen.split("\n")
    cleaned_lines = []
    for line in lines:
        if (line.startswith("def ") or line.startswith("if __name__") or line.startswith("class ")) and not line.startswith("    "):
            break
        cleaned_lines.append(line)
    return prompt + "\n".join(cleaned_lines)

def extract_gsm8k_answer(text: str) -> str:
    # 1. Match #### <num>
    m = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "").strip()
    # 2. Match \boxed{<num>}
    m2 = re.findall(r"\\boxed\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}", text)
    if m2:
        return m2[-1].replace(",", "").strip()
    # 3. Match "The answer is <num>"
    m3 = re.findall(r"(?:the\s+answer\s+is|final\s+answer\s+is)\s*[:=]?\s*(-?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if m3:
        return m3[-1].replace(",", "").strip()
    # 4. Fallback: match last numeric token
    nums = re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return ""

def normalize_num(s: str) -> float:
    try:
        return float(s.replace(",", "").strip())
    except Exception:
        return float("nan")

# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================

def run_humaneval_suite(model, tokenizer, runtime, is_hadl: bool, he_data, log_file, device="cuda:0") -> Dict[str, Any]:
    print(f"\n{'='*70}\n[*] STARTING 100% OFFICIAL HUMANEVAL BENCHMARK (164 TASKS, is_hadl={is_hadl})\n{'='*70}", flush=True)
    results = {}
    passed_count = 0
    total_tokens = 0
    t0_start = time.perf_counter()

    for idx, sample in enumerate(he_data):
        tid = sample["task_id"]
        entry = sample["entry_point"]
        prompt = sample["prompt"]
        test = sample["test"]

        if is_hadl:
            runtime.reset_state()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t_task_0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += gen_tokens
        raw_gen = tokenizer.decode(out[0], skip_special_tokens=True)
        full_code = clean_humaneval_code(prompt, raw_gen)
        
        # Test harness
        test_script = f"{full_code}\n\n{test}\ncheck({entry})"
        passed, err = safe_exec_test(test_script, timeout=3.0)
        dt_task = time.perf_counter() - t_task_0

        if passed:
            passed_count += 1
            status_str = "PASS"
        else:
            status_str = f"FAIL ({err})"

        cum_pass = (passed_count / (idx + 1)) * 100.0
        log_line = f"[{'HADL' if is_hadl else 'BASE'} HE {idx+1:03d}/164] {tid} ({entry}): {status_str} | dt={dt_task:.2f}s | cum={cum_pass:.1f}%"
        print(log_line, flush=True)
        log_file.write(log_line + "\n")
        log_file.flush()

        results[tid] = {
            "entry_point": entry,
            "passed": passed,
            "error": err,
            "latency_sec": round(dt_task, 3),
            "tokens": int(gen_tokens),
            "code_snippet": full_code[:200]
        }

    total_time = time.perf_counter() - t0_start
    pass_rate = (passed_count / len(he_data)) * 100.0
    print(f"\n[+] HumanEval Done! Pass@1: {pass_rate:.2f}% ({passed_count}/{len(he_data)}) in {total_time:.2f}s", flush=True)

    return {
        "tasks": results,
        "total_tasks": len(he_data),
        "passed_tasks": passed_count,
        "pass_rate_pct": round(pass_rate, 2),
        "total_time_sec": round(total_time, 2),
        "total_tokens": total_tokens,
        "throughput_tps": round(total_tokens / total_time, 2)
    }

def run_gsm8k_suite(model, tokenizer, runtime, is_hadl: bool, gsm_data, log_file, num_tasks=100, device="cuda:0") -> Dict[str, Any]:
    print(f"\n{'='*70}\n[*] STARTING OFFICIAL GSM8K REASONING BENCHMARK ({num_tasks} TASKS, is_hadl={is_hadl})\n{'='*70}", flush=True)
    results = {}
    correct_count = 0
    total_tokens = 0
    t0_start = time.perf_counter()

    for idx in range(num_tasks):
        sample = gsm_data[idx]
        q = sample["question"]
        gt_raw = sample["answer"].split("####")[-1].strip()
        gt_val = normalize_num(gt_raw)

        if is_hadl:
            runtime.reset_state()

        messages = [
            {"role": "system", "content": "You are a master mathematical problem solver. Solve step by step. State your final answer at the very end in the exact format: '#### <number>'."},
            {"role": "user", "content": q}
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        t_task_0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += gen_tokens
        gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_raw = extract_gsm8k_answer(gen_text)
        pred_val = normalize_num(pred_raw)

        correct = (abs(pred_val - gt_val) < 1e-4) if (not np.isnan(pred_val) and not np.isnan(gt_val)) else False
        dt_task = time.perf_counter() - t_task_0

        if correct:
            correct_count += 1
            status_str = f"CORRECT (pred={pred_raw}, gt={gt_raw})"
        else:
            status_str = f"INCORRECT (pred={pred_raw}, gt={gt_raw})"

        cum_acc = (correct_count / (idx + 1)) * 100.0
        log_line = f"[{'HADL' if is_hadl else 'BASE'} GSM {idx+1:03d}/{num_tasks}] Task {idx}: {status_str} | dt={dt_task:.2f}s | cum={cum_acc:.1f}%"
        print(log_line, flush=True)
        log_file.write(log_line + "\n")
        log_file.flush()

        results[f"gsm8k_{idx}"] = {
            "question": q[:120],
            "ground_truth": gt_raw,
            "prediction": pred_raw,
            "correct": correct,
            "latency_sec": round(dt_task, 3),
            "tokens": int(gen_tokens)
        }

    total_time = time.perf_counter() - t0_start
    accuracy = (correct_count / num_tasks) * 100.0
    print(f"\n[+] GSM8K Done! Accuracy: {accuracy:.2f}% ({correct_count}/{num_tasks}) in {total_time:.2f}s", flush=True)

    return {
        "tasks": results,
        "total_tasks": num_tasks,
        "correct_tasks": correct_count,
        "accuracy_pct": round(accuracy, 2),
        "total_time_sec": round(total_time, 2),
        "total_tokens": total_tokens,
        "throughput_tps": round(total_tokens / total_time, 2)
    }

# ==============================================================================
# PLOTTING
# ==============================================================================

def generate_benchmark_graph(base_he, hadl_he, base_gsm, hadl_gsm, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Large-Scale Authentic Benchmark Suite (264 Tasks)\nBaseline (Frozen 2B) vs. HADL v4.5 (2B + Cognitive Organs)", fontsize=18, fontweight="bold", y=0.98)

    # 1. Bar Chart: HumanEval Pass@1 & GSM8K Accuracy
    ax1 = axes[0, 0]
    categories = ["HumanEval Pass@1\n(164 Tasks - 100% Full)", "GSM8K Accuracy\n(100 Test Tasks)"]
    base_scores = [base_he["pass_rate_pct"], base_gsm["accuracy_pct"]]
    hadl_scores = [hadl_he["pass_rate_pct"], hadl_gsm["accuracy_pct"]]
    x = np.arange(len(categories))
    w = 0.35

    bars1 = ax1.bar(x - w/2, base_scores, w, label="Baseline (Frozen 2B)", color="#4A90E2", alpha=0.85)
    bars2 = ax1.bar(x + w/2, hadl_scores, w, label="HADL v4.5 (2B + Organs)", color="#D9381E", alpha=0.9)
    ax1.set_ylabel("Accuracy / Pass Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Benchmark Accuracy Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=11)

    for bar in bars1:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + 1.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1c3d5a")
    for bar in bars2:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + 1.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#7b1113")

    # 2. Cumulative Pass Rate on HumanEval (1..164)
    ax2 = axes[0, 1]
    base_he_cum = []
    c = 0
    for idx, (tid, data) in enumerate(base_he["tasks"].items(), 1):
        if data["passed"]: c += 1
        base_he_cum.append((c / idx) * 100.0)

    hadl_he_cum = []
    c = 0
    for idx, (tid, data) in enumerate(hadl_he["tasks"].items(), 1):
        if data["passed"]: c += 1
        hadl_he_cum.append((c / idx) * 100.0)

    ax2.plot(range(1, 165), base_he_cum, label="Baseline (Frozen 2B)", color="#4A90E2", lw=2)
    ax2.plot(range(1, 165), hadl_he_cum, label="HADL v4.5 (2B)", color="#D9381E", lw=2.5)
    ax2.set_xlabel("HumanEval Task Index (1 to 164)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Pass@1 (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Cumulative Pass Rate Trajectory on HumanEval (164 Tasks)", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=11)

    # 3. Cumulative Accuracy on GSM8K (1..100)
    ax3 = axes[1, 0]
    base_gsm_cum = []
    c = 0
    for idx, (tid, data) in enumerate(base_gsm["tasks"].items(), 1):
        if data["correct"]: c += 1
        base_gsm_cum.append((c / idx) * 100.0)

    hadl_gsm_cum = []
    c = 0
    for idx, (tid, data) in enumerate(hadl_gsm["tasks"].items(), 1):
        if data["correct"]: c += 1
        hadl_gsm_cum.append((c / idx) * 100.0)

    ax3.plot(range(1, 101), base_gsm_cum, label="Baseline (Frozen 2B)", color="#4A90E2", lw=2)
    ax3.plot(range(1, 101), hadl_gsm_cum, label="HADL v4.5 (2B)", color="#D9381E", lw=2.5)
    ax3.set_xlabel("GSM8K Problem Index (1 to 100)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Cumulative Accuracy (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Cumulative Accuracy Trajectory on GSM8K (100 Problems)", fontsize=13, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="lower right", fontsize=11)

    # 4. Token Generation Throughput (TPS) & Hardware Metrics
    ax4 = axes[1, 1]
    metric_labels = ["HumanEval TPS\n(Tokens/s)", "GSM8K TPS\n(Tokens/s)"]
    base_tps = [base_he["throughput_tps"], base_gsm["throughput_tps"]]
    hadl_tps = [hadl_he["throughput_tps"], hadl_gsm["throughput_tps"]]
    x2 = np.arange(len(metric_labels))

    b1 = ax4.bar(x2 - w/2, base_tps, w, label="Baseline", color="#6BAED6", alpha=0.85)
    b2 = ax4.bar(x2 + w/2, hadl_tps, w, label="HADL v4.5", color="#E6550D", alpha=0.9)
    ax4.set_ylabel("Throughput (Tokens/s)", fontsize=11, fontweight="bold")
    ax4.set_title("Inference Throughput & Latency (RTX 5060 Laptop)", fontsize=13, fontweight="bold")
    ax4.set_xticks(x2)
    ax4.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", fontsize=11)

    for bar in b1:
        y = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in b2:
        y = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(out_path), dpi=300)
    plt.close()
    print(f"[+] Graph saved to {out_path}", flush=True)

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(str(LOG_OUT), "w", encoding="utf-8")
    log_file.write(f"=== LARGE-SCALE AUTHENTIC BENCHMARK (264 TASKS) ===\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    log_file.flush()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading tokenizer and model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    model.eval()

    from dual_loop.cognitive_organs import DynamicAdaptiveConfusionSensor

    print("[*] Initializing HADL Dynamic Runtime...")
    runtime = HADLDynamicRuntime(
        base_model=model,
        checkpoint_path=str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None,
        d_model=model.config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        enable_canvas=True
    )
    runtime.canvas.use_adaptive_sensor = True
    if runtime.canvas.adaptive_sensor is None:
        runtime.canvas.adaptive_sensor = DynamicAdaptiveConfusionSensor(
            d_model=model.config.hidden_size,
            num_heads=min(8, max(1, model.config.hidden_size // 16)),
            k_sigma=0.50
        ).to(device=device, dtype=torch.bfloat16)

    print("[*] Loading datasets...")
    he_dataset = datasets.load_dataset("openai/openai_humaneval", split="test")
    gsm_dataset = datasets.load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(he_dataset)} HumanEval tasks and {len(gsm_dataset)} GSM8K tasks.")

    # 1. BASELINE HUMANEVAL (164 tasks)
    runtime.detach()
    base_he_results = run_humaneval_suite(model, tokenizer, runtime, is_hadl=False, he_data=he_dataset, log_file=log_file, device=device)

    # 2. HADL HUMANEVAL (164 tasks)
    runtime.attach()
    hadl_he_results = run_humaneval_suite(model, tokenizer, runtime, is_hadl=True, he_data=he_dataset, log_file=log_file, device=device)

    # 3. BASELINE GSM8K (100 tasks)
    runtime.detach()
    base_gsm_results = run_gsm8k_suite(model, tokenizer, runtime, is_hadl=False, gsm_data=gsm_dataset, log_file=log_file, num_tasks=100, device=device)

    # 4. HADL GSM8K (100 tasks)
    runtime.attach()
    hadl_gsm_results = run_gsm8k_suite(model, tokenizer, runtime, is_hadl=True, gsm_data=gsm_dataset, log_file=log_file, num_tasks=100, device=device)

    # Save comprehensive results JSON
    full_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_tasks_evaluated": 264,
        "baseline": {
            "humaneval": base_he_results,
            "gsm8k": base_gsm_results
        },
        "hadl_v45": {
            "humaneval": hadl_he_results,
            "gsm8k": hadl_gsm_results
        }
    }

    with open(str(JSON_OUT), "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"[+] Full JSON saved to {JSON_OUT}")

    # Generate graphs
    generate_benchmark_graph(base_he_results, hadl_he_results, base_gsm_results, hadl_gsm_results, GRAPH_OUT)
    
    # Copy graph to artifact dir if exists
    try:
        if ARTIFACT_DIR.exists():
            import shutil
            shutil.copy2(str(GRAPH_OUT), str(ARTIFACT_DIR / GRAPH_OUT.name))
            print(f"[+] Graph copied to artifact directory: {ARTIFACT_DIR / GRAPH_OUT.name}")
    except Exception as e:
        print(f"Warning: could not copy to artifact dir: {e}")

    log_file.close()
    print("\n[✓] ALL 264 BENCHMARK TASKS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
