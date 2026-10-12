#!/usr/bin/env python3
"""
Tri-Model Benchmark: 20 Canonical Benchmarks across 50 Questions
================================================================
Compares 3 Model Generations:
1. Model Dasar: Base Qwen/Qwen3.5-2B (100% Frozen Baseline)
2. Model Sebelumnya: X-Star-2B (HADL v4.5 Dual-Cup 27B-Grade Distillation)
3. Model Sekarang: Qwen3.5-2B + Stabilized VGNI-LBU x KV-Cloud Engine

Measures:
- Accuracy / Score across 20 Benchmarks (50 Questions total)
- Generation Speed (tokens/sec) & Total Latency
- Generated Token Count & Verbosity Reduction
- Epistemic Grounding & Stability Telemetry

Saves:
- JSON: eval_results/benchmark_3models_50q_comparison.json
- PNG: eval_results/benchmark_3models_50q_comparison.png
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
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.dual_cup_poly_engine import attach_hadl_v45_dualcup
from dual_loop.vgni_lbu_cloud_engine import (
    MonolithicIncoherentSensitivityField,
    KVCloudCompilerDecompiler,
    VirtualWorldGeneralIntelligence,
    LogicBaseUnit
)
from scripts.evaluate_20bench_50q import build_20bench_50q_suite

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
PREV_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "xstar_2b_omnireason_27b_grade_checkpoint.pt"
CURR_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "vgni_lbu_stabilized_qwen2b.pt"
JSON_OUT = PROJECT_ROOT / "eval_results" / "benchmark_3models_50q_comparison.json"
GRAPH_OUT = PROJECT_ROOT / "eval_results" / "benchmark_3models_50q_comparison.png"


# ==============================================================================
# SELECTION: 50 SOAL ACROSS ALL 20 CANONICAL BENCHMARKS
# ==============================================================================
def sample_50q_across_20bench(full_suite: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Samples exactly 50 representative questions covering all 20 canonical benchmarks.
    Allocation: 10 benchmarks x 3 questions + 10 benchmarks x 2 questions = 50 questions.
    """
    # Group by benchmark
    grouped = {}
    for it in full_suite:
        b = it["benchmark"]
        if b not in grouped:
            grouped[b] = []
        grouped[b].append(it)

    # 10 benchmarks get 3 questions, 10 get 2 questions
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


# ==============================================================================
# VGNI-LBU CONTROLLER & LOGITS PROCESSOR (MODEL SEKARANG)
# ==============================================================================
class VGNIBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, bias_vector: Any, bias_scale: float):
        self.bias_vector = bias_vector
        self.bias_scale = bias_scale

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.bias_scale > 0.0 and self.bias_vector is not None:
            return scores + self.bias_scale * self.bias_vector.to(scores.device, scores.dtype)
        return scores


class VGNILBUController(nn.Module):
    def __init__(self, base_model: nn.Module, d_model: int = 2048, vgni_capacity: int = 64):
        super().__init__()
        self.base_model = base_model
        self.d_model = d_model
        self.misf = MonolithicIncoherentSensitivityField(d_model=d_model)
        self.vgni = VirtualWorldGeneralIntelligence(d_model=d_model, max_epistemic_entries=vgni_capacity)
        self.kv_cloud = KVCloudCompilerDecompiler(d_model=d_model, cloud_capacity=32)

    def extract_hidden(self, input_ids: torch.Tensor, layer_idx: int = 11) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            return outputs.hidden_states[layer_idx]

    def compute_guidance(self, input_ids: torch.Tensor) -> Tuple[Any, float, Dict[str, Any]]:
        with torch.no_grad():
            h_mid = self.extract_hidden(input_ids, layer_idx=11)
            viscosity, misf_telem = self.misf(h_mid)
            aligned_approach, affinity = self.vgni.query_prior_grounding(h_mid)

            if affinity > 0.30:
                lbu_score = self.vgni.lbu.evaluate_step_validity(aligned_approach, h_mid.mean(dim=1))
                bias_vec = self.base_model.lm_head(aligned_approach.to(device=self.base_model.device, dtype=self.base_model.dtype))
                bias_vec = F.normalize(bias_vec, dim=-1) * 2.0
                bias_scale = 0.20 * float(affinity) * float(lbu_score.mean().item())
            else:
                bias_vec = None
                bias_scale = 0.0
                lbu_score = torch.tensor([0.5])

            telem = {
                "vgni_affinity": round(float(affinity), 4),
                "lbu_conviction": round(float(lbu_score.mean().item()), 4),
                "bias_scale": round(float(bias_scale), 4)
            }
            return bias_vec, bias_scale, telem


# ==============================================================================
# EVALUATION RUNNER
# ==============================================================================
def run_evaluation_pass(
    model: nn.Module,
    tokenizer: Any,
    suite_50q: List[Dict[str, Any]],
    model_type: str,
    vgni_controller: Any = None
) -> Dict[str, Any]:
    print(f"\n[*] Running Evaluation Pass for: {model_type}...")
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

        # Handle different model modalities
        logits_proc = None
        telem = {}
        if model_type == "Model Sekarang (VGNI-LBU)" and vgni_controller is not None:
            bias_vec, bias_scale, telem = vgni_controller.compute_guidance(inp.input_ids)
            logits_proc = LogitsProcessorList([VGNIBiasLogitsProcessor(bias_vec, bias_scale)])

        torch.cuda.synchronize()
        t0_q = time.perf_counter()
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_tok,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id
            }
            if logits_proc is not None:
                gen_kwargs["logits_processor"] = logits_proc

            out_ids = model.generate(**inp, **gen_kwargs)
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
            "target": target,
            "telemetry": telem
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


# ==============================================================================
# MAIN COMPARISON PIPELINE
# ==============================================================================
def main():
    print("=" * 100)
    print("TRI-MODEL BENCHMARK: 20 CANONICAL BENCHMARKS x 50 SOAL")
    print("Model Dasar vs Model Sebelumnya (HADL v4.5) vs Model Sekarang (VGNI-LBU)")
    print("=" * 100)
    print(f"[*] Compute Device : {DEVICE}")
    print(f"[*] Base Model Path: {MODEL_DIR}")

    # Build 1,000 Questions suite, then sample exactly 50 questions across 20 benchmarks
    full_1000q_suite = build_20bench_50q_suite()
    suite_50q = sample_50q_across_20bench(full_1000q_suite)
    print(f"[+] Sampled {len(suite_50q)} questions covering all 20 canonical benchmarks.\n")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading Qwen3.5-2B backbone in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(DEVICE)
    base_model.eval()
    print("[+] Model loaded.")

    # -------------------------------------------------------------------------
    # 1. EVALUATE MODEL DASAR (BASE QWEN3.5-2B FROZEN)
    # -------------------------------------------------------------------------
    res_base = run_evaluation_pass(base_model, tokenizer, suite_50q, "Model Dasar (Base Qwen3.5-2B)")

    # -------------------------------------------------------------------------
    # 2. EVALUATE MODEL SEBELUMNYA (HADL v4.5 DUAL-CUP 27B-GRADE DISTILLED)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[*] Attaching HADL v4.5 Dual-Cup Controller for Model Sebelumnya...")
    hadl_model = attach_hadl_v45_dualcup(base_model=base_model, target_layer_idx=11, ghost_layer_idx=23)
    if PREV_CHECKPOINT.exists():
        ckpt_prev = torch.load(str(PREV_CHECKPOINT), map_location=DEVICE, weights_only=True)
        hadl_model.controller.load_state_dict(ckpt_prev["controller_state_dict"], strict=False)
        print(f"[+] Restored HADL v4.5 Checkpoint from: {PREV_CHECKPOINT}")
    else:
        print("[!] Warning: HADL v4.5 checkpoint not found, using initialized controller.")
    hadl_model.eval()

    res_prev = run_evaluation_pass(hadl_model, tokenizer, suite_50q, "Model Sebelumnya (HADL v4.5 Dual-Cup)")

    # Remove HADL hooks to restore clean backbone
    if hadl_model.hook_handle_target is not None:
        hadl_model.hook_handle_target.remove()
    if hadl_model.hook_handle_ghost is not None:
        hadl_model.hook_handle_ghost.remove()
    del hadl_model
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 3. EVALUATE MODEL SEKARANG (VGNI-LBU x KV-CLOUD ENGINE)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[*] Initializing VGNI-LBU Controller for Model Sekarang...")
    d_model = base_model.config.hidden_size
    vgni_controller = VGNILBUController(base_model=base_model, d_model=d_model, vgni_capacity=64).to(DEVICE)
    vgni_controller.to(dtype=torch.bfloat16)

    if CURR_CHECKPOINT.exists():
        ckpt_curr = torch.load(str(CURR_CHECKPOINT), map_location=DEVICE, weights_only=True)
        vgni_controller.vgni.concept_anchors.copy_(ckpt_curr["vgni_state"]["concept_anchors"].to(DEVICE))
        vgni_controller.vgni.approach_vectors.copy_(ckpt_curr["vgni_state"]["approach_vectors"].to(DEVICE))
        vgni_controller.vgni.usage_counts.copy_(ckpt_curr["vgni_state"]["usage_counts"].to(DEVICE))
        vgni_controller.vgni.num_stored.copy_(ckpt_curr["vgni_state"]["num_stored"].to(DEVICE))
        vgni_controller.vgni.lbu.u_logic.data.copy_(ckpt_curr["lbu_weights"]["u_logic"].to(DEVICE))
        vgni_controller.vgni.lbu.w_step.weight.data.copy_(ckpt_curr["lbu_weights"]["w_step"].to(DEVICE))
        vgni_controller.misf.w_freq.data.copy_(ckpt_curr["misf_weights"]["w_freq"].to(DEVICE))
        vgni_controller.misf.phase_shift.data.copy_(ckpt_curr["misf_weights"]["phase_shift"].to(DEVICE))
        vgni_controller.misf.viscosity_proj.weight.data.copy_(ckpt_curr["misf_weights"]["viscosity_proj"].to(DEVICE))
        absorbed_n = int(vgni_controller.vgni.num_stored.item())
        print(f"[+] Restored Stabilized VGNI-LBU Checkpoint from: {CURR_CHECKPOINT}")
        print(f"[+] VGNI Epistemic Saturation: {absorbed_n}/64 slots ({absorbed_n/64*100:.2f}%)")
    else:
        print("[!] Warning: VGNI-LBU checkpoint not found.")

    res_curr = run_evaluation_pass(base_model, tokenizer, suite_50q, "Model Sekarang (VGNI-LBU)", vgni_controller=vgni_controller)

    # -------------------------------------------------------------------------
    # 4. PRINT AGGREGATE SUMMARY TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TRI-MODEL BENCHMARK SUMMARY (50 SOAL ACROSS 20 BENCHMARKS)")
    print("=" * 100)
    row_fmt = "{:<32} | {:<12} | {:<10} | {:<12} | {:<12} | {:<12}"
    print(row_fmt.format("Model Generation", "Skor (Total)", "Akurasi %", "Total Tokens", "Speed (tok/s)", "Latency (s)"))
    print("-" * 100)
    print(row_fmt.format("1. Model Dasar (Base Qwen3.5-2B)", res_base["score"], f"{res_base['accuracy_pct']}%", f"{res_base['total_tokens_generated']} toks", f"{res_base['mean_tokens_per_sec']} t/s", f"{res_base['total_latency_sec']} s"))
    print(row_fmt.format("2. Model Sebelumnya (HADL v4.5)", res_prev["score"], f"{res_prev['accuracy_pct']}%", f"{res_prev['total_tokens_generated']} toks", f"{res_prev['mean_tokens_per_sec']} t/s", f"{res_prev['total_latency_sec']} s"))
    print(row_fmt.format("3. Model Sekarang (VGNI-LBU)", res_curr["score"], f"{res_curr['accuracy_pct']}%", f"{res_curr['total_tokens_generated']} toks", f"{res_curr['mean_tokens_per_sec']} t/s", f"{res_curr['total_latency_sec']} s"))
    print("=" * 100)

    # Pillar-wise breakdown
    pillars = ["Math & Quantitative", "Science & Academic", "Coding & Software", "Commonsense & Logic", "Instruction & Chat"]
    pillar_summary = {}
    print("\nBREAKDOWN PER PILLAR KEILMUAN:")
    print("-" * 90)
    print(f"{'Pillar Keilmuan':<26} | {'Model Dasar':<18} | {'Model Sebelumnya':<18} | {'Model Sekarang':<18}")
    print("-" * 90)
    for pil in pillars:
        b_c = sum(res_base["results_by_benchmark"][b]["correct"] for b in res_base["results_by_benchmark"] if res_base["results_by_benchmark"][b]["pillar"] == pil)
        p_c = sum(res_prev["results_by_benchmark"][b]["correct"] for b in res_prev["results_by_benchmark"] if res_prev["results_by_benchmark"][b]["pillar"] == pil)
        c_c = sum(res_curr["results_by_benchmark"][b]["correct"] for b in res_curr["results_by_benchmark"] if res_curr["results_by_benchmark"][b]["pillar"] == pil)
        tot = sum(res_base["results_by_benchmark"][b]["total"] for b in res_base["results_by_benchmark"] if res_base["results_by_benchmark"][b]["pillar"] == pil)

        pillar_summary[pil] = {
            "total_questions": tot,
            "base_score": f"{b_c}/{tot}", "base_acc": round(b_c/tot*100, 1),
            "prev_score": f"{p_c}/{tot}", "prev_acc": round(p_c/tot*100, 1),
            "curr_score": f"{c_c}/{tot}", "curr_acc": round(c_c/tot*100, 1)
        }
        print(f"{pil:<26} | {b_c}/{tot} ({b_c/tot*100:.1f}%)         | {p_c}/{tot} ({p_c/tot*100:.1f}%)         | {c_c}/{tot} ({c_c/tot*100:.1f}%)")
    print("-" * 90)

    # -------------------------------------------------------------------------
    # 5. GENERATE COMPARATIVE GRAPH
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Graph 1: Accuracy & Pillar Scores
    x = np.arange(len(pillars))
    w = 0.26
    b_scores = [pillar_summary[p]["base_acc"] for p in pillars]
    p_scores = [pillar_summary[p]["prev_acc"] for p in pillars]
    c_scores = [pillar_summary[p]["curr_acc"] for p in pillars]

    ax1.bar(x - w, b_scores, width=w, label=f"Model Dasar ({res_base['accuracy_pct']}%)", color="#78909C")
    ax1.bar(x, p_scores, width=w, label=f"Model Sebelumnya ({res_prev['accuracy_pct']}%)", color="#D32F2F")
    ax1.bar(x + w, c_scores, width=w, label=f"Model Sekarang ({res_curr['accuracy_pct']}%)", color="#1B5E20")

    ax1.set_title("Perbandingan Skor Akurasi per Pillar (50 Soal)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace(" & ", "\n& ") for p in pillars], fontsize=9)
    ax1.set_ylabel("Akurasi (%)", fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper right", fontsize=9)

    # Graph 2: Speed & Latency Comparison
    models_labels = ["Model Dasar", "Model Sebelumnya\n(HADL v4.5)", "Model Sekarang\n(VGNI-LBU)"]
    speeds = [res_base["mean_tokens_per_sec"], res_prev["mean_tokens_per_sec"], res_curr["mean_tokens_per_sec"]]
    latencies = [res_base["total_latency_sec"], res_prev["total_latency_sec"], res_curr["total_latency_sec"]]

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

    # -------------------------------------------------------------------------
    # 6. SAVE JSON AUDIT
    # -------------------------------------------------------------------------
    full_audit = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": DEVICE,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "suite_info": {
            "total_questions": len(suite_50q),
            "canonical_benchmarks_count": 20
        },
        "models_evaluated": {
            "model_dasar": {
                "name": "Base Qwen/Qwen3.5-2B",
                "summary": res_base
            },
            "model_sebelumnya": {
                "name": "X-Star-2B (HADL v4.5 Dual-Cup 27B-Grade Distillation)",
                "checkpoint": str(PREV_CHECKPOINT),
                "summary": res_prev
            },
            "model_sekarang": {
                "name": "Qwen3.5-2B + Stabilized VGNI-LBU x KV-Cloud Engine",
                "checkpoint": str(CURR_CHECKPOINT),
                "summary": res_curr
            }
        },
        "pillar_summary": pillar_summary
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(JSON_OUT), "w", encoding="utf-8") as f:
        json.dump(full_audit, f, indent=2)

    print(f"[+] Saved Detailed Audit Report to: {JSON_OUT}")
    print("[+] TRI-MODEL BENCHMARK COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
