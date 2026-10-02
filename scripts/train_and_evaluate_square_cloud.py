import os
import sys
import json
import time
import re
import random
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('.')
from dual_loop.square_cloud_engine import SquareCloudModelWrapper
from scripts.train_and_evaluate_6principles import TRAINING_SAMPLES, TEST_TASKS, check_answer_correctness

MODEL_ID = "Qwen/Qwen3.5-2B"
OUTPUT_CKPT = "checkpoints/tuned_square_cloud_engine.pt"
OUTPUT_JSON = "eval_results/tuned_square_cloud_report.json"

def main():
    print("=" * 80)
    print("TUNING & EMPIRICAL EVALUATION: SQUARECLOUD DYNAMIC ENGINE (GPU)")
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

    # Freeze base model parameters
    for p in base_model.parameters():
        p.requires_grad = False

    # Wrap model with SquareCloud Dynamic Engine
    sq_model = SquareCloudModelWrapper(base_model, target_layer_idx=11)
    sq_model.engine.to(device=device, dtype=torch.bfloat16)

    trainable_params = [p for p in sq_model.engine.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable SquareCloud Parameters: {trainable_count:,} (~{trainable_count/base_model.num_parameters()*100:.2f}% of model)")

    # Prepare tokenized training items
    train_items = []
    for item in TRAINING_SAMPLES:
        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["target"]}
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        enc = tokenizer(full_text, return_tensors="pt")
        input_ids = enc.input_ids.to(device)

        prompt_only = tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt_only).input_ids)

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        train_items.append((input_ids, labels))

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

    print("\n--------------------------------------------------------------------------------")
    print(f"PHASE 1: TUNING SQUARECLOUD ENGINE ON REASONING PATTERNS (12 EPOCHS)")
    print("--------------------------------------------------------------------------------")
    sq_model.engine.train()

    NUM_EPOCHS = 12
    loss_history = []
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        random.shuffle(train_items)
        epoch_loss = 0.0

        for input_ids, labels in train_items:
            optimizer.zero_grad()
            outputs = base_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_items)
        loss_history.append(round(avg_loss, 4))
        if epoch in (1, 2, 4, 6, 8, 10, 12):
            print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Causal Loss: {avg_loss:.4f}")

    train_duration = time.time() - t_start
    print(f"\nTuning Complete in {train_duration:.2f}s! Initial Loss: {loss_history[0]} -> Final Loss: {loss_history[-1]}")

    os.makedirs(os.path.dirname(OUTPUT_CKPT), exist_ok=True)
    torch.save(sq_model.engine.state_dict(), OUTPUT_CKPT)
    print(f"Tuned weights saved to: {OUTPUT_CKPT}")

    # -------------------------------------------------------------------------
    # PHASE 2: EVALUATION ON HELD-OUT TASKS
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("PHASE 2: EVALUATING POST-TUNED SQUARECLOUD ENGINE ON HELD-OUT TASKS")
    print("--------------------------------------------------------------------------------")
    sq_model.engine.eval()

    eval_results = []
    for idx, task in enumerate(TEST_TASKS, 1):
        print(f"\n--- Testing Task {idx}/3: {task['name']} ---")
        messages = [{"role": "user", "content": task["prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = base_model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=[248046, 248044]
            )
        lat = time.time() - t0
        gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
        toks = len(gen_ids)
        resp = tokenizer.decode(gen_ids, skip_special_tokens=True)

        eval_metrics = check_answer_correctness(task["id"], task["ground_truth"], resp)
        telem = sq_model.engine.last_telemetry

        print(f"Result: {eval_metrics['extracted_answer']} | Correct: {eval_metrics['is_correct']} | Latency: {lat:.2f}s ({toks/lat:.1f} tok/s)")
        print(f"Telemetry: Active Dim={telem.get('active_dimensions', 0)}/{sq_model.engine.d_model}, Judge Verdict={telem.get('judge_verdict', 1.0)}, Mean Rot Deg={telem.get('mean_rotation_deg', 0.0):.2f}, Isometry Error={telem.get('isometry_error', 0.0):.6f}")
        print("Tail of response:")
        print("\n".join(resp.splitlines()[-4:]))

        eval_results.append({
            "task_id": task["id"],
            "task_name": task["name"],
            "ground_truth": task["ground_truth"],
            "post_tuned_evaluation": eval_metrics,
            "generated_tokens": toks,
            "latency_s": round(lat, 3),
            "telemetry": telem,
            "tail": "\n".join(resp.splitlines()[-4:])
        })

    sq_model.remove_hook()

    correct_count = sum(1 for r in eval_results if r["post_tuned_evaluation"]["is_correct"])
    score_str = f"{correct_count}/3 ({correct_count/3*100:.1f}%)"

    print("\n" + "=" * 80)
    print("SQUARECLOUD EMPIRICAL COMPARISON")
    print("=" * 80)
    print(f"Untrained Baseline Score:  1/3 (33.3%)")
    print(f"Post-Tuned SquareCloud:    {score_str}")

    full_report = {
        "experiment": "Tuning SquareCloud Dynamic Engine",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_name": MODEL_ID,
        "trainable_parameters": trainable_count,
        "tuning_duration_s": round(train_duration, 2),
        "loss_history": loss_history,
        "final_score": score_str,
        "correct_count": correct_count,
        "detailed_tasks": eval_results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\nComplete log written to:\n{OUTPUT_JSON}")

if __name__ == "__main__":
    main()
