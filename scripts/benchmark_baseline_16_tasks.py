"""
Benchmark Baseline Qwen3.5-2B on Full 16 Tasks
Provides strict empirical head-to-head comparison against HADL v4.3.
"""

import sys
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrain_qwen_27b_grade_dualloop import MODEL_ID
from scripts.train_and_evaluate_hadl_v4_empirical import (
    EXPANDED_TEST_SUITE,
    EXPANDED_TEST_SUITE_PART2,
    check_expanded_eval_correctness
)

OUTPUT_FILE = PROJECT_ROOT / "eval_results" / "baseline_qwen35_2b_16_tasks_report.json"

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading raw Base Model {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    all_tasks = EXPANDED_TEST_SUITE + EXPANDED_TEST_SUITE_PART2
    print(f"[*] Running Base Model across {len(all_tasks)} Benchmark Tasks...")

    results = []
    correct_count = 0

    for idx, task in enumerate(all_tasks, 1):
        messages = [{"role": "user", "content": task["prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=600,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.perf_counter() - t0

        generated_tokens = output_ids[0][prompt_len:]
        num_new_tokens = len(generated_tokens)
        speed = num_new_tokens / max(elapsed, 1e-4)

        resp_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        eval_check = check_expanded_eval_correctness(task["id"], task["ground_truth"], resp_text)

        if eval_check["is_correct"]:
            correct_count += 1
            status_tag = "[PASS]"
        else:
            status_tag = "[FAIL]"

        print(f"[{idx:2d}/16] {status_tag} {task['name']:<50} | Truth: {task['ground_truth']:<10} | Got: {eval_check['extracted_answer']:<10} | Tokens: {num_new_tokens:<3}")

        results.append({
            "task_id": task["id"],
            "name": task["name"],
            "domain": task["domain"],
            "is_correct": eval_check["is_correct"],
            "ground_truth": task["ground_truth"],
            "extracted": eval_check["extracted_answer"],
            "num_tokens": num_new_tokens,
            "speed_tok_s": round(speed, 2),
            "generated_preview": resp_text.strip()[:140],
            "full_output": resp_text.strip()
        })

    accuracy = (correct_count / len(all_tasks)) * 100.0
    print(f"\n[>] Raw Base Model Accuracy: {accuracy:.1f}% ({correct_count}/{len(all_tasks)})")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": MODEL_ID,
            "accuracy_pct": accuracy,
            "passed": correct_count,
            "total": len(all_tasks),
            "task_results": results
        }, f, indent=2)
    print(f"[OK] Saved baseline report to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
