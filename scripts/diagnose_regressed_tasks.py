"""
Deep Technical Diagnostic on the 3 Regressed Tasks:
- autorccar   (Baseline 95.0% vs HADL 93.3%)
- markdownify (Baseline 100.0% vs HADL 93.3%)
- retrying    (Baseline 86.7% vs HADL 80.0%)
"""

import sys
import os
import time
import ast
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.runtime.dynamic_runtime import HADLDynamicRuntime
from dual_loop.cognitive_organs import DynamicAdaptiveConfusionSensor
from scripts.benchmark_nl2repobench_20_grand_tasks import (
    BENCHMARK_20_TASKS,
    extract_files_from_output,
    evaluate_task_generation
)

MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "xstar_2b_omnireason_carlift_500q_checkpoint.pt"

TARGET_IDS = ["autorccar", "markdownify", "retrying"]
target_tasks = [t for t in BENCHMARK_20_TASKS if t["id"] in TARGET_IDS]

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  DIAGNOSTIC AUDIT: ROOT CAUSE ANALYSIS OF DROPPED BENCHMARKS")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    base_model.eval()

    runtime = HADLDynamicRuntime(
        base_model=base_model,
        checkpoint_path=str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None,
        d_model=base_model.config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        enable_canvas=True
    )
    runtime.canvas.use_adaptive_sensor = True
    if runtime.canvas.adaptive_sensor is None:
        runtime.canvas.adaptive_sensor = DynamicAdaptiveConfusionSensor(
            d_model=base_model.config.hidden_size,
            num_heads=min(8, max(1, base_model.config.hidden_size // 16)),
            k_sigma=0.50
        ).to(device=device, dtype=torch.bfloat16)

    diagnostics = {}

    for mode in ["baseline", "hadl"]:
        print(f"\n{'='*40} TESTING MODE: {mode.upper()} {'='*40}")
        if mode == "hadl":
            runtime.attach()
        else:
            runtime.detach()

        diagnostics[mode] = {}

        for task in target_tasks:
            tid = task["id"]
            if mode == "hadl":
                runtime.reset_state()

            messages = [
                {"role": "system", "content": "You are a precise, production-grade software engineer building repositories from scratch."},
                {"role": "user", "content": task["prompt"]}
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model_out = base_model.generate(
                    **inputs,
                    max_new_tokens=450,
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
            decoded = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            eval_metrics = evaluate_task_generation(task, decoded)
            extracted_files = extract_files_from_output(decoded, task=task)

            # Detailed AST syntax inspection per file
            file_ast_status = {}
            for fname, code in extracted_files.items():
                try:
                    ast.parse(code)
                    file_ast_status[fname] = {"valid": True, "error": None, "length_chars": len(code)}
                except Exception as e:
                    file_ast_status[fname] = {"valid": False, "error": str(e), "length_chars": len(code)}

            diagnostics[mode][tid] = {
                "metrics": eval_metrics,
                "gen_tokens": gen_tokens,
                "hit_max_tokens": gen_tokens >= 448,
                "file_ast_status": file_ast_status,
                "raw_text": decoded
            }

            print(f"\nTask: {tid} | Mode: {mode}")
            print(f"  Tokens: {gen_tokens} / 450 (Hit Max: {gen_tokens >= 448})")
            print(f"  Composite: {eval_metrics['composite_score_pct']}% | AST: {eval_metrics['ast_syntax_integrity_pct']}% | SAC: {eval_metrics['api_specification_compliance_pct']}%")
            print(f"  Files Generated: {list(extracted_files.keys())}")
            for fname, st in file_ast_status.items():
                valid_str = "VALID" if st["valid"] else f"FAILED: {st['error']}"
                print(f"    - {fname}: {valid_str} ({st['length_chars']} chars)")

    if mode == "hadl":
        runtime.detach()

    # Save diagnostic results
    out_file = PROJECT_ROOT / "eval_results" / "diagnostic_dropped_tasks.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        # Save without massive raw_text for readability, keep snippet
        clean_diag = {}
        for m, tasks_data in diagnostics.items():
            clean_diag[m] = {}
            for tid, tdata in tasks_data.items():
                clean_diag[m][tid] = {
                    "metrics": tdata["metrics"],
                    "gen_tokens": tdata["gen_tokens"],
                    "hit_max_tokens": tdata["hit_max_tokens"],
                    "file_ast_status": tdata["file_ast_status"],
                    "last_150_chars": tdata["raw_text"][-150:]
                }
        json.dump(clean_diag, f, indent=2)

    print(f"\n[+] Diagnostic saved to: {out_file}")

if __name__ == "__main__":
    main()
