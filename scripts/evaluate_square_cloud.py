import os
import sys
import json
import time
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('.')
from dual_loop.square_cloud_engine import SquareCloudModelWrapper
from scripts.evaluate_option1_extended_tokens import TEST_TASKS, check_answer_correctness

MODEL_ID = "Qwen/Qwen3.5-2B"
OUTPUT_JSON = "eval_results/square_cloud_empirical_benchmark.json"

def main():
    print("=" * 80)
    print("EMPIRICAL BENCHMARK: SQUARECLOUD DYNAMIC ENGINE ON QWEN3.5-2B (GPU)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Wrap model with SquareCloud Dynamic Engine
    sq_model = SquareCloudModelWrapper(base_model, target_layer_idx=14)
    sq_model.engine.to(device=device, dtype=torch.bfloat16)
    sq_model.engine.eval()

    eval_results = []

    for idx, task in enumerate(TEST_TASKS, 1):
        print(f"\n--- Testing Task {idx}/3: {task['name']} (SquareCloud Engine) ---")
        messages = [{"role": "user", "content": task["prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = base_model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=False
            )
        lat = time.time() - t0
        gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
        toks = len(gen_ids)
        resp = tokenizer.decode(gen_ids, skip_special_tokens=True)

        eval_metrics = check_answer_correctness(task["id"], task["ground_truth"], resp)
        telem = sq_model.engine.last_telemetry

        print(f"Result: {eval_metrics['extracted_answer']} | Correct: {eval_metrics['is_correct']} | Tokens: {toks} | Latency: {lat:.2f}s ({toks/lat:.1f} tok/s)")
        print(f"Telemetry: Active Dim={telem.get('active_dimensions', 0)}/{sq_model.engine.d_model}, Judge Verdict={telem.get('judge_verdict', 1.0)}, Isometry Error={telem.get('isometry_error', 0.0):.6f}")
        print("Tail of response:")
        print("\n".join(resp.splitlines()[-4:]))

        eval_results.append({
            "task_id": task["id"],
            "task_name": task["name"],
            "ground_truth": task["ground_truth"],
            "evaluation": eval_metrics,
            "generated_tokens": toks,
            "latency_s": round(lat, 3),
            "telemetry": telem,
            "tail": "\n".join(resp.splitlines()[-4:])
        })

    sq_model.remove_hook()

    correct_count = sum(1 for r in eval_results if r["evaluation"]["is_correct"])
    score_str = f"{correct_count}/3 ({correct_count/3*100:.1f}%)"

    print("\n" + "=" * 80)
    print("SQUARECLOUD EMPIRICAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"SquareCloud Final Score: {score_str}")

    out_data = {
        "experiment": "SquareCloud Dynamic Engine Empirical Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_name": MODEL_ID,
        "score": score_str,
        "correct_count": correct_count,
        "results": eval_results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"\nFull log saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
