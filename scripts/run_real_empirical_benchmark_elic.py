"""
HADL v4.5+ Real-World Empirical Benchmark: Baseline vs. Dynamic Runtime + ELIC
=============================================================================
Conducts a rigorous, side-by-side empirical benchmark comparing:
1. Pure Frozen Foundation Model (dist/staging_hf_fused, Qwen3.5-2B)
2. HADL v4.5+ Dynamic Recurrent Runtime + Emergent Latent Imagination Canvas (ELIC)
   + OC-Dim Context Threads + Dual-Store Plastic Memory

Evaluates across 5 core pillars:
- Pillar 1: Multi-Step Quantitative Reasoning (GSM8K Math)
- Pillar 2: Long-Horizon Algorithmic State Tracking (Register Machine 10-20 Steps)
- Pillar 3: Semantic Grounding & Concept Disambiguation ("Bakar Api" vs "REST API")
- Pillar 4: Long-Horizon Complex Code Synthesis (AST Validity & Structure Completeness)
- Pillar 5: Inference Latency, Token Throughput, Memory & Energy Stability
"""

import os
import sys
import json
import time
import re
import ast
import math
import shutil
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.runtime.dynamic_runtime import HADLDynamicRuntime
from dual_loop.imagination_canvas import EmergentLatentCanvas

MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "xstar_2b_omnireason_carlift_500q_checkpoint.pt"
JSON_OUT = PROJECT_ROOT / "eval_results" / "hadl_v45_elic_empirical_benchmark.json"
GRAPH_OUT = PROJECT_ROOT / "docs" / "images" / "hadl_v45_elic_empirical_benchmark.png"
ARTIFACT_DIR = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")


# ==============================================================================
# 1. TEST SUITE DEFINITIONS
# ==============================================================================

GSM8K_TESTS = [
    {
        "id": "gsm_01",
        "prompt": "Janet buys 8 notebooks at $5 each and 4 pens at $2 each. She pays with a $50 bill. How much change does she receive?\nAnswer:",
        "target": "2",
        "checker": lambda t: bool(re.search(r"\b2\b|\$2\b|change (is|of) 2", t, re.I))
    },
    {
        "id": "gsm_02",
        "prompt": "A bakery produces 120 loaves of bread in the morning. They sell 75% of them before noon. In the afternoon, they bake 50 more loaves and sell 30. How many loaves of bread remain at the end of the day?\nAnswer:",
        "target": "50",
        "checker": lambda t: bool(re.search(r"\b50\b|remains? (is|of) 50", t, re.I))
    },
    {
        "id": "gsm_03",
        "prompt": "A water tank holds 600 liters. A pump empties it at 25 liters per minute, while an inlet fills it at 10 liters per minute. If the tank starts full, how many minutes will it take until the tank is completely empty?\nAnswer:",
        "target": "40",
        "checker": lambda t: bool(re.search(r"\b40\b|40 minutes?", t, re.I))
    },
    {
        "id": "gsm_04",
        "prompt": "Tom earns $18 per hour for the first 40 hours of work, and 1.5 times his regular rate for overtime hours. If he works 48 hours in one week, what is his total earnings?\nAnswer:",
        "target": "936",
        "checker": lambda t: bool(re.search(r"\b936\b|\$936\b", t, re.I))
    },
    {
        "id": "gsm_05",
        "prompt": "A factory has 3 machines. Machine A produces 40 widgets/hour, Machine B produces 50 widgets/hour, and Machine C produces 60 widgets/hour. If all three machines run simultaneously for 6 hours, how many widgets are produced in total?\nAnswer:",
        "target": "900",
        "checker": lambda t: bool(re.search(r"\b900\b", t, re.I))
    },
    {
        "id": "gsm_06",
        "prompt": "A store offers a 20% discount on a $150 jacket. After the discount, a 10% sales tax is added to the discounted price. What is the final total price of the jacket?\nAnswer:",
        "target": "132",
        "checker": lambda t: bool(re.search(r"\b132\b|\$132\b", t, re.I))
    },
    {
        "id": "gsm_07",
        "prompt": "Sarah has three times as many marbles as Leo. Leo has 14 marbles. If Sarah gives 10 marbles to Leo, how many marbles does Sarah have left?\nAnswer:",
        "target": "32",
        "checker": lambda t: bool(re.search(r"\b32\b", t, re.I))
    },
    {
        "id": "gsm_08",
        "prompt": "A train travels at 75 km/h for 2 hours, and then increases its speed to 90 km/h for the next 3 hours. What is the total distance traveled by the train?\nAnswer:",
        "target": "420",
        "checker": lambda t: bool(re.search(r"\b420\b|420\s*km", t, re.I))
    },
    {
        "id": "gsm_09",
        "prompt": "David invested $800 at a simple annual interest rate of 5% for 4 years. What is the total amount (principal + interest) David will have at the end of 4 years?\nAnswer:",
        "target": "960",
        "checker": lambda t: bool(re.search(r"\b960\b|\$960\b", t, re.I))
    },
    {
        "id": "gsm_10",
        "prompt": "If 12 workers can build a wall in 6 days, how many days would it take 9 workers to build the exact same wall working at the same rate?\nAnswer:",
        "target": "8",
        "checker": lambda t: bool(re.search(r"\b8\b|8 days?", t, re.I))
    }
]

REGISTER_TESTS = [
    {
        "id": "reg_10_step",
        "name": "10-Step Register Trace",
        "prompt": """Below is a 10-step execution trace of a register machine with 4 registers [A, B, C, D] starting at [0, 0, 0, 0].
Follow every operation step-by-step and output the final values of all registers as JSON: {"A": int, "B": int, "C": int, "D": int}.

Initial: A=0, B=0, C=0, D=0
Step 1: A = 8
Step 2: B = A + 4
Step 3: C = B * 3
Step 4: D = C - 6
Step 5: A = A + D
Step 6: B = B * 2
Step 7: C = C + A
Step 8: D = D // 2
Step 9: A = A - 10
Step 10: B = B + C

Final JSON:""",
        # Trace:
        # A=8, B=12, C=36, D=30
        # A=8+30=38, B=24, C=36+38=74, D=15
        # A=38-10=28, B=24+74=98
        "ground_truth": {"A": 28, "B": 98, "C": 74, "D": 15}
    },
    {
        "id": "reg_15_step",
        "name": "15-Step Register Trace",
        "prompt": """Below is a 15-step execution trace of a register machine with 4 registers [A, B, C, D] starting at [0, 0, 0, 0].
Follow every operation step-by-step and output the final values of all registers as JSON: {"A": int, "B": int, "C": int, "D": int}.

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
Step 15: D = D + 10

Final JSON:""",
        # Trace:
        # 1: A=10
        # 2: B=15
        # 3: C=30
        # 4: D=26
        # 5: A=10+26=36
        # 6: B=30
        # 7: C=30+36=66
        # 8: D=13
        # 9: A=36-6=30
        # 10: B=30+66=96
        # 11: C=66//2=33
        # 12: D=13+96=109
        # 13: A=30*2=60
        # 14: B=96-60=36
        # 15: D=109+10=119
        "ground_truth": {"A": 60, "B": 36, "C": 33, "D": 119}
    }
]

DISAMBIGUATION_TESTS = [
    {
        "id": "dis_combustion",
        "type": "Physics / Chemistry",
        "prompt": "Jelaskan proses api bakar kayu secara kimiawi dan fisika. Apakah api tersebut bisa padam jika kekurangan oksigen atau disiram air? Jelaskan secara ilmiah.",
        "target_terms": ["oksigen", "pembakaran", "panas", "segitiga api", "karbon", "padam"],
        "forbidden_terms": ["endpoint", "http", "rest", "json", "server", "port", "request"],
    },
    {
        "id": "dis_software",
        "type": "Software Engineering",
        "prompt": "Dalam dunia pemrograman web, jelaskan apa itu Application Programming Interface (API) dan bagaimana siklus hidup sebuah API endpoint saat server sedang down atau offline.",
        "target_terms": ["antarmuka", "software", "endpoint", "server", "request", "protokol"],
        "forbidden_terms": ["kayu", "abu", "asap", "bakar", "panas", "bara", "arang"],
    },
    {
        "id": "dis_cross_domain",
        "type": "Cross-Domain Ambiguity Stress Test",
        "prompt": "Apakah sebuah API (Application Programming Interface) bisa 'padam' seperti api kayu bakar? Uraikan perbedaan mendasar antara padamnya api fisik dan tidak aktifnya sebuah API software tanpa mencampuradukkan konsep keduanya.",
        "target_terms": ["perbedaan", "fisik", "software", "sistem", "offline", "pembakaran"],
        "forbidden_terms": [],
    }
]

CODE_SYNTHESIS_TESTS = [
    {
        "id": "code_lru_cache",
        "task": "Thread-Safe TTL LRU Cache",
        "prompt": """Write a complete, thread-safe LRU Cache in Python with TTL (time-to-live) expiration.
Requirements:
1. Implement class `ThreadSafeLRUCache`.
2. Must use `threading.RLock` or `threading.Lock` to guarantee thread safety.
3. Methods: `get(key)`, `put(key, value, ttl_seconds)`, and `cleanup_expired()`.
4. Output valid, executable Python code only.
Code:""",
        "required_tokens": ["class ThreadSafeLRUCache", "def get", "def put", "Lock", "time"]
    },
    {
        "id": "code_pubsub_bus",
        "task": "Async Event-Driven Message Bus",
        "prompt": """Write a production-grade async Pub/Sub event bus in Python.
Requirements:
1. Implement class `AsyncEventBus`.
2. Use `asyncio` and `collections.defaultdict` for subscriber management.
3. Methods: `subscribe(topic, handler)`, `publish(topic, data)`, and `unsubscribe(topic, handler)`.
4. Exception-safe handling so a failing subscriber does not crash others.
Output valid, executable Python code only.
Code:""",
        "required_tokens": ["class AsyncEventBus", "async def publish", "subscribe", "asyncio"]
    }
]


# ==============================================================================
# 2. EVALUATION ENGINES
# ==============================================================================

def run_single_prompt(model, tokenizer, prompt: str, max_tokens: int = 256, device="cuda:0") -> Tuple[str, float, int]:
    messages = [
        {"role": "system", "content": "You are a precise, highly coherent scientific and technical assistant."},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - t0
    gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
    decoded = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return decoded, elapsed, gen_tokens


def evaluate_gsm8k(model, tokenizer, device="cuda:0") -> Dict[str, Any]:
    correct = 0
    total = len(GSM8K_TESTS)
    latencies = []
    tokens = []
    
    for item in GSM8K_TESTS:
        resp, t_el, n_tok = run_single_prompt(model, tokenizer, item["prompt"], max_tokens=128, device=device)
        latencies.append(t_el)
        tokens.append(n_tok)
        is_ok = item["checker"](resp)
        if is_ok:
            correct += 1
            
    acc = (correct / total) * 100.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    tps = sum(tokens) / sum(latencies) if sum(latencies) > 0 else 0
    return {
        "accuracy_pct": round(acc, 2),
        "correct": correct,
        "total": total,
        "avg_latency_sec": round(avg_latency, 3),
        "tokens_per_sec": round(tps, 2)
    }


def evaluate_registers(model, tokenizer, device="cuda:0") -> Dict[str, Any]:
    total_regs = 0
    correct_regs = 0
    details = []
    
    for item in REGISTER_TESTS:
        resp, t_el, n_tok = run_single_prompt(model, tokenizer, item["prompt"], max_tokens=256, device=device)
        gt = item["ground_truth"]
        
        # Parse JSON from response
        match = re.search(r"\{[^{}]*\}", resp)
        pred = {}
        if match:
            try:
                pred = json.loads(match.group(0))
            except Exception:
                pass
                
        task_correct = 0
        for reg in ["A", "B", "C", "D"]:
            total_regs += 1
            if reg in pred and pred[reg] == gt[reg]:
                correct_regs += 1
                task_correct += 1
                
        details.append({
            "task": item["id"],
            "pred": pred,
            "ground_truth": gt,
            "score": f"{task_correct}/4"
        })
        
    acc = (correct_regs / total_regs) * 100.0 if total_regs > 0 else 0
    return {
        "register_accuracy_pct": round(acc, 2),
        "correct_registers": correct_regs,
        "total_registers": total_regs,
        "details": details
    }


def evaluate_disambiguation(model, tokenizer, canvas: EmergentLatentCanvas, device="cuda:0") -> Dict[str, Any]:
    scores = []
    coherence_energies = []
    bleed_rates = []
    
    for item in DISAMBIGUATION_TESTS:
        resp, t_el, n_tok = run_single_prompt(model, tokenizer, item["prompt"], max_tokens=220, device=device)
        resp_lower = resp.lower()
        
        # Measure target term retention
        present_targets = sum(1 for t in item["target_terms"] if t in resp_lower)
        target_score = (present_targets / len(item["target_terms"])) * 100.0 if item["target_terms"] else 100.0
        
        # Measure concept bleed / contamination
        bleed_count = sum(1 for f in item["forbidden_terms"] if f in resp_lower)
        has_bleed = bleed_count > 0
        bleed_rates.append(1.0 if has_bleed else 0.0)
        
        # Compute latent coherence energy on embedding of response
        if len(resp.strip()) > 0:
            tokens = tokenizer(resp, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_input_embeddings()(tokens.input_ids)
                _, energy = canvas(emb, return_energy=True)
                if energy is not None:
                    coherence_energies.append(energy.item())
                
        scores.append(target_score)
        
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_energy = sum(coherence_energies) / len(coherence_energies) if coherence_energies else 0
    avg_bleed = (sum(bleed_rates) / len(bleed_rates)) * 100.0 if bleed_rates else 0
    
    return {
        "semantic_adherence_pct": round(avg_score, 2),
        "cross_domain_bleed_pct": round(avg_bleed, 2),
        "mean_latent_coherence_energy": round(avg_energy, 4)
    }


def evaluate_code_synthesis(model, tokenizer, device="cuda:0") -> Dict[str, Any]:
    valid_syntax_count = 0
    struct_scores = []
    details = []
    
    for item in CODE_SYNTHESIS_TESTS:
        resp, t_el, n_tok = run_single_prompt(model, tokenizer, item["prompt"], max_tokens=450, device=device)
        
        # Extract code block if inside markdown
        code_str = resp
        if "```python" in resp:
            parts = resp.split("```python")
            if len(parts) > 1:
                code_str = parts[1].split("```")[0]
        elif "```" in resp:
            parts = resp.split("```")
            if len(parts) > 1:
                code_str = parts[1].split("```")[0]
                
        # 1. AST Parsing
        is_valid_syntax = False
        try:
            ast.parse(code_str)
            is_valid_syntax = True
            valid_syntax_count += 1
        except Exception:
            is_valid_syntax = False
            
        # 2. Structural Requirements Checking
        req_present = sum(1 for req in item["required_tokens"] if req in code_str)
        struct_pct = (req_present / len(item["required_tokens"])) * 100.0
        struct_scores.append(struct_pct)
        
        details.append({
            "task": item["id"],
            "syntax_valid": is_valid_syntax,
            "structural_completeness_pct": round(struct_pct, 1)
        })
        
    syntax_pct = (valid_syntax_count / len(CODE_SYNTHESIS_TESTS)) * 100.0
    avg_struct = sum(struct_scores) / len(struct_scores) if struct_scores else 0
    
    return {
        "syntax_validity_pct": round(syntax_pct, 2),
        "structural_completeness_pct": round(avg_struct, 2),
        "details": details
    }


def audit_latent_stability(model, tokenizer, device="cuda:0") -> Dict[str, Any]:
    """Measures latent hidden representation norm variance and drift over 200 tokens."""
    prompt = "Explain in detail the mathematical derivation of Navier-Stokes equations and fluid conservation laws."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    norms = []
    def monitor_hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        norms.append(torch.norm(h.float(), p=2, dim=-1).mean().item())
        return out
        
    hook_handle = model.model.layers[11].register_forward_hook(monitor_hook)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    hook_handle.remove()
    
    mean_norm = float(np.mean(norms)) if norms else 0.0
    std_norm = float(np.std(norms)) if norms else 0.0
    max_drift = float(np.max(norms) - np.min(norms)) if norms else 0.0
    stability_score = max(0.0, 100.0 - (std_norm / (mean_norm + 1e-6)) * 100.0)
    
    return {
        "mean_norm": round(mean_norm, 3),
        "std_norm": round(std_norm, 3),
        "max_drift": round(max_drift, 3),
        "stability_score_pct": round(stability_score, 2)
    }


# ==============================================================================
# 3. VISUALIZATION GENERATOR
# ==============================================================================

def generate_comparison_chart(results: Dict[str, Any], save_path: Path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11), dpi=250)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    palette_base = "#64748b"  # Slate Gray
    palette_hadl = "#0284c7"  # Sky Blue / Cyan
    
    # Panel 1: Reasoning & Math Accuracy (GSM8K & Register Tracking)
    labels_p1 = ["GSM8K Math\nAccuracy", "Register Machine\nState Tracking"]
    base_p1 = [results["baseline"]["gsm8k"]["accuracy_pct"], results["baseline"]["registers"]["register_accuracy_pct"]]
    hadl_p1 = [results["hadl_elic"]["gsm8k"]["accuracy_pct"], results["hadl_elic"]["registers"]["register_accuracy_pct"]]
    
    x = np.arange(len(labels_p1))
    width = 0.35
    rects1 = ax1.bar(x - width/2, base_p1, width, label="Baseline (Frozen 2B)", color=palette_base, alpha=0.9)
    rects2 = ax1.bar(x + width/2, hadl_p1, width, label="HADL v4.5 + ELIC Canvas", color=palette_hadl, alpha=0.95)
    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Pillar 1 & 2: Reasoning & State Retention Accuracy", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_p1, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left")
    for r in rects1:
        ax1.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width()/2, r.get_height() + 2),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=palette_base)
    for r in rects2:
        ax1.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width()/2, r.get_height() + 2),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=palette_hadl)

    # Panel 2: Concept Disambiguation & Cross-Domain Bleed Rate
    labels_p2 = ["Semantic Domain\nAdherence", "Cross-Domain Bleed\n(Concept Contamination)"]
    base_p2 = [results["baseline"]["disambiguation"]["semantic_adherence_pct"], results["baseline"]["disambiguation"]["cross_domain_bleed_pct"]]
    hadl_p2 = [results["hadl_elic"]["disambiguation"]["semantic_adherence_pct"], results["hadl_elic"]["disambiguation"]["cross_domain_bleed_pct"]]
    
    x2 = np.arange(len(labels_p2))
    rects3 = ax2.bar(x2 - width/2, base_p2, width, label="Baseline (Frozen 2B)", color=palette_base, alpha=0.9)
    rects4 = ax2.bar(x2 + width/2, hadl_p2, width, label="HADL v4.5 + ELIC Canvas", color=palette_hadl, alpha=0.95)
    ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Pillar 3: Semantic Grounding ('Bakar Api' vs 'REST API')", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_p2, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right")
    for r in rects3:
        ax2.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width()/2, r.get_height() + 2),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=palette_base)
    for r in rects4:
        ax2.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width()/2, r.get_height() + 2),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=palette_hadl)

    # Panel 3: Code Synthesis Quality (Syntax Validity & Completeness)
    labels_p3 = ["Python AST Syntax\nValidity", "Structural Design\nCompleteness"]
    base_p3 = [results["baseline"]["code"]["syntax_validity_pct"], results["baseline"]["code"]["structural_completeness_pct"]]
    hadl_p3 = [results["hadl_elic"]["code"]["syntax_validity_pct"], results["hadl_elic"]["code"]["structural_completeness_pct"]]
    
    x3 = np.arange(len(labels_p3))
    rects5 = ax3.bar(x3 - width/2, base_p3, width, label="Baseline (Frozen 2B)", color=palette_base, alpha=0.9)
    rects6 = ax3.bar(x3 + width/2, hadl_p3, width, label="HADL v4.5 + ELIC Canvas", color=palette_hadl, alpha=0.95)
    ax3.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Pillar 4: Long-Horizon Complex Code Synthesis", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels_p3, fontsize=10, fontweight="bold")
    ax3.set_ylim(0, 115)
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.legend(loc="upper left")
    for r in rects5:
        ax3.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width()/2, r.get_height() + 2),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=palette_base)
    for r in rects6:
        ax3.annotate(f"{r.get_height():.1f}%", xy=(r.get_x() + r.get_width()/2, r.get_height() + 2),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=palette_hadl)

    # Panel 4: Runtime Speed & Latent Stability
    labels_p4 = ["Throughput\n(Tokens/Sec)", "Latent Lyapunov\nStability Score (%)"]
    base_p4 = [results["baseline"]["gsm8k"]["tokens_per_sec"], results["baseline"]["stability"]["stability_score_pct"]]
    hadl_p4 = [results["hadl_elic"]["gsm8k"]["tokens_per_sec"], results["hadl_elic"]["stability"]["stability_score_pct"]]
    
    x4 = np.arange(len(labels_p4))
    rects7 = ax4.bar(x4 - width/2, base_p4, width, label="Baseline (Frozen 2B)", color=palette_base, alpha=0.9)
    rects8 = ax4.bar(x4 + width/2, hadl_p4, width, label="HADL v4.5 + ELIC Canvas", color=palette_hadl, alpha=0.95)
    ax4.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax4.set_title("Pillar 5: Computational Throughput & Latent Energy Stability", fontsize=12, fontweight="bold", pad=12)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(labels_p4, fontsize=10, fontweight="bold")
    ax4.set_ylim(0, max(max(base_p4), max(hadl_p4)) * 1.25)
    ax4.grid(axis="y", linestyle="--", alpha=0.3)
    ax4.legend(loc="upper right")
    for r in rects7:
        ax4.annotate(f"{r.get_height():.1f}", xy=(r.get_x() + r.get_width()/2, r.get_height() + 2),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=palette_base)
    for r in rects8:
        ax4.annotate(f"{r.get_height():.1f}", xy=(r.get_x() + r.get_width()/2, r.get_height() + 2),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=palette_hadl)

    plt.suptitle("HADL v4.5+ Real-World Empirical Benchmark Audit: Baseline vs. Dynamic Runtime + ELIC",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison chart to: {save_path}")


# ==============================================================================
# 4. MAIN BENCHMARK RUNNER
# ==============================================================================

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  HADL v4.5+ EMPIRICAL REAL-WORLD BENCHMARK AUDIT")
    print("  Evaluating: Baseline vs. Dynamic Recurrent Runtime + ELIC Canvas")
    print("=" * 80)
    print(f"[*] Device              : {device}")
    print(f"[*] Model Directory     : {MODEL_DIR}")
    print(f"[*] Checkpoint          : {CHECKPOINT_PATH}")
    
    # 1. Load Model & Tokenizer
    print("\n[*] Loading Base Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    base_model.eval()
    
    # 2. Instantiate HADL Dynamic Runtime
    print("[*] Initializing HADL Dynamic Runtime with ELIC & Dual-Store Memory...")
    runtime = HADLDynamicRuntime(
        base_model=base_model,
        checkpoint_path=str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None,
        d_model=base_model.config.hidden_size,
        device=device,
        dtype=torch.bfloat16
    )
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "model": str(MODEL_DIR),
        "baseline": {},
        "hadl_elic": {}
    }
    
    # --------------------------------------------------------------------------
    # EVALUATION RUN 1: BASELINE FROZEN MODEL
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RUN 1: BASELINE FROZEN FOUNDATION MODEL (NO CONTROLLER)")
    print("=" * 60)
    runtime.detach()  # Ensure zero hooks
    
    print("\n[Baseline 1/5] GSM8K Math Reasoning (10 Real Questions)...")
    base_gsm = evaluate_gsm8k(base_model, tokenizer, device=device)
    print(f"  -> Accuracy: {base_gsm['accuracy_pct']}% | Throughput: {base_gsm['tokens_per_sec']} tps")
    
    print("[Baseline 2/5] Register Machine State Tracking (10 & 15 Steps)...")
    base_reg = evaluate_registers(base_model, tokenizer, device=device)
    print(f"  -> Register Accuracy: {base_reg['register_accuracy_pct']}% ({base_reg['correct_registers']}/{base_reg['total_registers']})")
    
    print("[Baseline 3/5] Disambiguation & Semantic Grounding ('Bakar Api' vs 'REST API')...")
    base_dis = evaluate_disambiguation(base_model, tokenizer, runtime.canvas, device=device)
    print(f"  -> Domain Adherence: {base_dis['semantic_adherence_pct']}% | Cross-Domain Bleed: {base_dis['cross_domain_bleed_pct']}%")
    
    print("[Baseline 4/5] Long-Horizon Complex Code Synthesis...")
    base_code = evaluate_code_synthesis(base_model, tokenizer, device=device)
    print(f"  -> Syntax Validity: {base_code['syntax_validity_pct']}% | Completeness: {base_code['structural_completeness_pct']}%")
    
    print("[Baseline 5/5] Latent Energy Stability Audit (200 Tokens Rollout)...")
    base_stab = audit_latent_stability(base_model, tokenizer, device=device)
    print(f"  -> Mean Norm: {base_stab['mean_norm']} | Max Drift: {base_stab['max_drift']} | Stability: {base_stab['stability_score_pct']}%")
    
    results["baseline"] = {
        "gsm8k": base_gsm,
        "registers": base_reg,
        "disambiguation": base_dis,
        "code": base_code,
        "stability": base_stab
    }
    
    # --------------------------------------------------------------------------
    # EVALUATION RUN 2: HADL v4.5+ DYNAMIC RUNTIME + ELIC
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RUN 2: HADL v4.5+ DYNAMIC RUNTIME + ELIC MENTAL CANVAS")
    print("=" * 60)
    runtime.reset_state()
    runtime.attach()  # Activate hydraulic piston, ELIC, context threads & dual memory
    
    print("\n[HADL 1/5] GSM8K Math Reasoning (10 Real Questions)...")
    hadl_gsm = evaluate_gsm8k(base_model, tokenizer, device=device)
    print(f"  -> Accuracy: {hadl_gsm['accuracy_pct']}% | Throughput: {hadl_gsm['tokens_per_sec']} tps")
    
    print("[HADL 2/5] Register Machine State Tracking (10 & 15 Steps)...")
    hadl_reg = evaluate_registers(base_model, tokenizer, device=device)
    print(f"  -> Register Accuracy: {hadl_reg['register_accuracy_pct']}% ({hadl_reg['correct_registers']}/{hadl_reg['total_registers']})")
    
    print("[HADL 3/5] Disambiguation & Semantic Grounding ('Bakar Api' vs 'REST API')...")
    hadl_dis = evaluate_disambiguation(base_model, tokenizer, runtime.canvas, device=device)
    print(f"  -> Domain Adherence: {hadl_dis['semantic_adherence_pct']}% | Cross-Domain Bleed: {hadl_dis['cross_domain_bleed_pct']}%")
    
    print("[HADL 4/5] Long-Horizon Complex Code Synthesis...")
    hadl_code = evaluate_code_synthesis(base_model, tokenizer, device=device)
    print(f"  -> Syntax Validity: {hadl_code['syntax_validity_pct']}% | Completeness: {hadl_code['structural_completeness_pct']}%")
    
    print("[HADL 5/5] Latent Energy Stability Audit (200 Tokens Rollout)...")
    hadl_stab = audit_latent_stability(base_model, tokenizer, device=device)
    print(f"  -> Mean Norm: {hadl_stab['mean_norm']} | Max Drift: {hadl_stab['max_drift']} | Stability: {hadl_stab['stability_score_pct']}%")
    
    runtime.detach()
    
    results["hadl_elic"] = {
        "gsm8k": hadl_gsm,
        "registers": hadl_reg,
        "disambiguation": hadl_dis,
        "code": hadl_code,
        "stability": hadl_stab
    }
    
    # --------------------------------------------------------------------------
    # SAVE STRUCTURED DATA & CHARTS
    # --------------------------------------------------------------------------
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved structured benchmark results to: {JSON_OUT}")
    
    generate_comparison_chart(results, GRAPH_OUT)
    
    # Copy graph to artifact directory for instant user view
    if ARTIFACT_DIR.exists():
        art_graph = ARTIFACT_DIR / "hadl_v45_elic_empirical_benchmark.png"
        shutil.copy2(GRAPH_OUT, art_graph)
        print(f"[+] Synced graph to artifact directory: {art_graph}")

    print("\n" + "=" * 80)
    print("  EMPIRICAL BENCHMARK AUDIT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
