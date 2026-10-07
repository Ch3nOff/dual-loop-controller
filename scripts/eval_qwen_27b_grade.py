import sys
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop.square_cloud_engine import SquareCloudModelWrapper
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.retrain_qwen_27b_grade_dualloop import (
    MODEL_ID,
    CHECKPOINT_OUTPUT,
    TEST_SUITE,
    check_eval_correctness,
    plot_retrain_benchmark,
    DOCS_IMG_OUTPUT,
    EVAL_RESULTS_OUTPUT,
    ARTIFACT_DIR
)

def evaluate_suite(model, tokenizer, test_tasks, device, desc="Model"):
    print(f"\n[*] Evaluating {desc} with max_new_tokens=450...")
    eval_results = []
    correct_count = 0
    total_speed = []

    for task in test_tasks:
        messages = [{"role": "user", "content": task["prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=450,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.perf_counter() - t0

        generated_tokens = output_ids[0][prompt_len:]
        num_new_tokens = len(generated_tokens)
        speed = num_new_tokens / max(elapsed, 1e-4)
        total_speed.append(speed)

        resp_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        eval_check = check_eval_correctness(task["id"], task["ground_truth"], resp_text)

        if eval_check["is_correct"]:
            correct_count += 1
            status_tag = "[PASS]"
        else:
            status_tag = "[FAIL]"

        print(f"  {status_tag} {task['name']:<48} | Ground Truth: {task['ground_truth']} | Extracted: {eval_check['extracted_answer']} | Speed: {speed:.1f} tok/s")
        print(f"      Full Output Snippet: {resp_text.strip()[-120:]}")

        eval_results.append({
            "task_id": task["id"],
            "name": task["name"],
            "domain": task["domain"],
            "is_correct": eval_check["is_correct"],
            "ground_truth": task["ground_truth"],
            "extracted": eval_check["extracted_answer"],
            "speed_tok_s": round(speed, 2),
            "generated_preview": resp_text.strip()[:140],
            "full_output": resp_text.strip()
        })

    accuracy = (correct_count / len(test_tasks)) * 100.0
    avg_speed = float(np.mean(total_speed)) if 'np' in globals() else sum(total_speed)/len(total_speed)
    print(f"[>] {desc} Summary: Accuracy = {accuracy:.1f}% ({correct_count}/{len(test_tasks)}) | Average Speed = {avg_speed:.1f} tok/s")
    return accuracy, eval_results

def main():
    import numpy as np
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    for p in base_model.parameters():
        p.requires_grad = False

    # 1. Base Model Eval
    base_acc, base_tasks = evaluate_suite(base_model, tokenizer, TEST_SUITE, device, desc="Baseline Qwen3.5-2B (Unaugmented)")

    # 2. Attach Dual-Loop and load trained checkpoint
    sq_model = SquareCloudModelWrapper(base_model, target_layer_idx=11)
    sq_model.engine.load_state_dict(torch.load(CHECKPOINT_OUTPUT, weights_only=True))
    sq_model.engine.to(device=device, dtype=torch.bfloat16)
    sq_model.engine.eval()

    dl_acc, dl_tasks = evaluate_suite(base_model, tokenizer, TEST_SUITE, device, desc="Dual-Loop Qwen3.5-2B (27B-Grade Retrained)")

    # Read existing report to preserve loss history
    with open(EVAL_RESULTS_OUTPUT, "r", encoding="utf-8") as f:
        existing_report = json.load(f)

    existing_report["scoreboard"]["baseline_accuracy_pct"] = base_acc
    existing_report["scoreboard"]["dualloop_accuracy_pct"] = dl_acc
    existing_report["scoreboard"]["net_gain_pct"] = round(dl_acc - base_acc, 2)
    existing_report["baseline_tasks"] = base_tasks
    existing_report["dualloop_tasks"] = dl_tasks

    with open(EVAL_RESULTS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(existing_report, f, indent=2)

    plot_retrain_benchmark(
        base_acc, dl_acc,
        existing_report["loss_progression"]["history"],
        dl_tasks,
        DOCS_IMG_OUTPUT
    )

    if ARTIFACT_DIR and Path(ARTIFACT_DIR).exists():
        art_path = Path(ARTIFACT_DIR) / "hadl_v34_27b_grade_retrain_benchmark.png"
        plot_retrain_benchmark(
            base_acc, dl_acc,
            existing_report["loss_progression"]["history"],
            dl_tasks,
            art_path
        )
        print(f"[OK] Mirrored visual scorecard to {art_path}")

    print("\n" + "="*80)
    print(f"FINAL AUDIT: Base Acc: {base_acc:.1f}% -> Dual-Loop Acc: {dl_acc:.1f}% (+{dl_acc - base_acc:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    main()
