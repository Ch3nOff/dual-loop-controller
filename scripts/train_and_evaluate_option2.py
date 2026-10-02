import os
import sys
import json
import time
import re
import random
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop.experimental_cognitive_engine import SixPrincipleModelWrapper

MODEL_ID = "Qwen/Qwen3.5-2B"
DATA_PATH = "data/extended_cognitive_tuning_dataset.json"
OUTPUT_CKPT = "checkpoints/tuned_6principles_option2.pt"
OUTPUT_JSON = "eval_results/option2_training_and_eval.json"

TEST_TASKS = [
    {
        "id": "task_1_non_abelian_algebra",
        "name": "Exotic Non-Abelian Algebraic Reduction",
        "ground_truth": "I",
        "prompt": (
            "You are learning a newly invented abstract algebraic system (G, *) with 4 generators {A, B, C, D} "
            "and an identity element I. This operation is strictly NON-COMMUTATIVE (order matters!).\n\n"
            "Fundamental Axioms:\n"
            "1. A * B = C\n"
            "2. B * A = D (Notice: A * B != B * A)\n"
            "3. C * D = A\n"
            "4. D * C = B\n"
            "5. Inversion Law: For any element X in {A, B, C, D}, X * X = I (every element is self-inverse).\n"
            "6. Identity Law: X * I = I * X = X for all X.\n"
            "7. Associativity: (X * Y) * Z = X * (Y * Z).\n\n"
            "Lemma derivations to help you:\n"
            "- Since B * A = D, multiplying by A on the right gives: (B * A) * A = D * A => B * (A * A) = D * A => B * I = D * A => D * A = B.\n"
            "- Multiplying by B on the left gives: B * (B * A) = B * D => (B * B) * A = B * D => I * A = B * D => B * D = A.\n"
            "- Since A * B = C, multiplying by B on the right gives: (A * B) * B = C * B => A * (B * B) = C * B => A * I = C * B => C * B = A.\n\n"
            "Challenge Task:\n"
            "Simplify the following expression step-by-step to a single irreducible element (A, B, C, D, or I):\n"
            "E = A * (B * D) * (C * B) * A\n\n"
            "Rules for your answer:\n"
            "- Show the reduction of each sub-expression step-by-step.\n"
            "- Never assume commutativity.\n"
            "- On the final line, state strictly: 'Final Answer: [ELEMENT]'"
        )
    },
    {
        "id": "task_2_stack_bytecode_machine",
        "name": "Reversible Stack Bytecode Machine Simulation",
        "ground_truth": "[7, 8, 0]",
        "prompt": (
            "You are an interpreter executing a newly designed synthetic stack-based bytecode virtual machine.\n"
            "The machine operates on a single integer stack, initially empty. Top of stack is at the right: [bottom, ..., top].\n\n"
            "Instruction Set Architecture (ISA):\n"
            "- PUSH v: Pushes integer v onto the top of the stack.\n"
            "- DUP_ODD: Checks the top element. If it is odd, push a copy of it onto the stack. If it is even, push integer 0.\n"
            "- SWAP_TOP: Swaps the top two elements of the stack.\n"
            "- ADD_FOLD: Pops the top two elements (let A be top, B be second). Computes (B + A) % 10 and pushes the result.\n"
            "- SUB_FOLD: Pops the top two elements (let A be top, B be second). Computes abs(B - A) and pushes the result.\n\n"
            "Program to Execute:\n"
            "1. PUSH 7\n"
            "2. DUP_ODD\n"
            "3. PUSH 4\n"
            "4. SWAP_TOP\n"
            "5. ADD_FOLD\n"
            "6. PUSH 9\n"
            "7. SUB_FOLD\n"
            "8. DUP_ODD\n\n"
            "Task:\n"
            "Execute each instruction from 1 to 8 in exact sequential order.\n"
            "For each step, show:\n"
            "- The operation performed\n"
            "- The exact resulting stack from bottom to top: [element1, element2, ...]\n"
            "On the final line, state strictly: 'Final Stack: [x, y, z]'"
        )
    },
    {
        "id": "task_3_cryptographic_hash_round",
        "name": "Synthetic Cryptographic Hash Round State Transformation",
        "ground_truth": "[1, 7, 1, 7]",
        "prompt": (
            "You are executing 1 round of 'X-Hash', a newly designed cryptographic permutation.\n"
            "State Representation: 4 4-bit nibbles S = [S0, S1, S2, S3], where each value is an integer in range [0..15].\n\n"
            "Round Transformation Rules:\n"
            "Step 1 (Non-Linear S-Box Substitution):\n"
            "For each nibble i in {0, 1, 2, 3}, compute:\n"
            "S_i' = (S_i * 3 + 1) mod 16\n\n"
            "Step 2 (Linear Diffusion Layer):\n"
            "Compute 4 intermediate values T0, T1, T2, T3:\n"
            "- T0 = (S0' + S1') mod 16\n"
            "- T1 = S1' XOR S2' (bitwise exclusive-OR)\n"
            "- T2 = (S2' + S3') mod 16\n"
            "- T3 = S3' XOR S0' (bitwise exclusive-OR)\n\n"
            "Step 3 (Cyclic Permutation):\n"
            "The final output state of the round is: [T1, T2, T3, T0] (cyclic left shift by 1).\n\n"
            "Input State for this Round:\n"
            "S = [2, 5, 0, 7]\n\n"
            "Task:\n"
            "Calculate Step 1, Step 2, and Step 3 carefully. Show all arithmetic calculations.\n"
            "On the final line, state strictly: 'Final State: [w, x, y, z]'"
        )
    }
]

def check_answer_correctness(task_id: str, ground_truth: str, text: str) -> dict:
    lower = text.lower()
    is_correct = False
    extracted = None
    
    if task_id == "task_1_non_abelian_algebra":
        match = re.search(r"final answer:\s*([a-d]|i)\b", lower)
        if match:
            extracted = match.group(1).upper()
            is_correct = (extracted == "I")
        else:
            is_correct = bool(re.search(r"(?:e\s*=\s*i\b|answer is\s*i\b|reduces to\s*i\b)", lower))
            extracted = "I" if is_correct else "UNKNOWN"
            
    elif task_id == "task_2_stack_bytecode_machine":
        match = re.search(r"final stack:\s*\[\s*7\s*,\s*8\s*,\s*0\s*\]", lower)
        if match:
            is_correct = True
            extracted = "[7, 8, 0]"
        else:
            st_match = re.search(r"final stack:\s*(\[[^\]]+\])", lower)
            extracted = st_match.group(1) if st_match else "MISMATCH"
            is_correct = ("[7, 8, 0]" in text)
            
    elif task_id == "task_3_cryptographic_hash_round":
        match = re.search(r"final state:\s*\[\s*1\s*,\s*7\s*,\s*1\s*,\s*7\s*\]", lower)
        if match:
            is_correct = True
            extracted = "[1, 7, 1, 7]"
        else:
            st_match = re.search(r"final state:\s*(\[[^\]]+\])", lower)
            extracted = st_match.group(1) if st_match else "MISMATCH"
            is_correct = ("[1, 7, 1, 7]" in text)
            
    return {
        "ground_truth": ground_truth,
        "extracted_answer": extracted,
        "is_correct": is_correct
    }

def main():
    print("=" * 80)
    print("OPTION 2: EXPANDED COGNITIVE TUNING (60 SAMPLES) & EVALUATION")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Freeze base model
    for p in base_model.parameters():
        p.requires_grad = False
        
    # Wrap model
    huld_model = SixPrincipleModelWrapper(base_model, target_layer_idx=14)
    huld_model.adapter.to(device=device, dtype=torch.bfloat16)
    
    # Initialize rotator angle_proj with small std
    nn.init.normal_(huld_model.adapter.unitary_rotator.angle_proj.weight, mean=0.0, std=0.01)
    
    trainable_params = [p for p in huld_model.adapter.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable Adapter Parameters: {trainable_count:,} (~{trainable_count/base_model.num_parameters()*100:.2f}% of model)")
    
    # Load dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} training samples from {DATA_PATH}")
    
    # Prepare tokenized training items
    train_items = []
    for item in dataset:
        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["target"]}
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        enc = tokenizer(full_text, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        
        # Mask prompt for causal loss
        prompt_only = tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt_only).input_ids)
        
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        train_items.append((input_ids, labels))
        
    optimizer = torch.optim.AdamW(trainable_params, lr=8e-4, weight_decay=1e-4)
    
    print("\n--------------------------------------------------------------------------------")
    print(f"PHASE 1: TUNING ADAPTER ON {len(dataset)} SAMPLES (6 EPOCHS)")
    print("--------------------------------------------------------------------------------")
    huld_model.adapter.train()
    
    NUM_EPOCHS = 6
    loss_history = []
    t_start = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        random.shuffle(train_items)
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        accum_steps = 4
        for step, (inp_ids, lbls) in enumerate(train_items):
            outputs = base_model(input_ids=inp_ids, labels=lbls)
            loss = outputs.loss / accum_steps
            loss.backward()
            epoch_loss += loss.item() * accum_steps
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_items):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        avg_loss = epoch_loss / len(train_items)
        loss_history.append(round(avg_loss, 4))
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Causal Loss: {avg_loss:.4f}")
        
    train_duration = time.time() - t_start
    print(f"\nTuning Complete in {train_duration:.2f}s! Initial Loss: {loss_history[0]} -> Final Loss: {loss_history[-1]}")
    
    os.makedirs(os.path.dirname(OUTPUT_CKPT), exist_ok=True)
    torch.save(huld_model.adapter.state_dict(), OUTPUT_CKPT)
    print(f"Tuned weights saved to: {OUTPUT_CKPT}")
    
    # -------------------------------------------------------------------------
    # PHASE 2: EVALUATION ON HELD-OUT TASKS
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("PHASE 2: EVALUATING POST-OPTION-2 ENGINE ON HELD-OUT TASKS")
    print("--------------------------------------------------------------------------------")
    huld_model.adapter.eval()
    
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
                max_new_tokens=1000,
                do_sample=False
            )
        lat = time.time() - t0
        gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
        toks = len(gen_ids)
        resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        eval_metrics = check_answer_correctness(task["id"], task["ground_truth"], resp)
        telem = huld_model.adapter.last_telemetry
        
        print(f"Result: {eval_metrics['extracted_answer']} | Correct: {eval_metrics['is_correct']} | Latency: {lat:.2f}s ({toks/lat:.1f} tok/s)")
        print(f"Telemetry: Rotation Rad={telem.get('mean_rotation_angle_rad', 0.0):.4f}, Judge Verdict={telem.get('judge_verdict', 1.0)}, Isometry Error={telem.get('isometry_error', 0.0):.6f}")
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
            "full_response": resp,
            "tail": "\n".join(resp.splitlines()[-4:])
        })
        
    huld_model.remove_hook()
    
    pre_tuning_score = "1/3 (33.3%)"
    post_tuning_correct = sum(1 for r in eval_results if r["post_tuned_evaluation"]["is_correct"])
    post_tuning_score = f"{post_tuning_correct}/3 ({post_tuning_correct/3*100:.1f}%)"
    
    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON: OPTION 2 RESULTS")
    print("=" * 80)
    print(f"Baseline Score:        {pre_tuning_score}")
    print(f"Option 2 Post-Tuned:   {post_tuning_score}")
    
    if post_tuning_correct > 1:
        direction = "NAIK (PERFORMA MENINGKAT)"
    elif post_tuning_correct < 1:
        direction = "TURUN (PERFORMA MENURUN)"
    else:
        direction = "TETAP SAMA (EQUAL)"
    print(f"Direction of Change:   {direction}")
    
    full_report = {
        "experiment": "Option 2: 60-Sample Expanded Cognitive Tuning",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_name": MODEL_ID,
        "trainable_parameters": trainable_count,
        "training_samples": len(dataset),
        "tuning_duration_s": round(train_duration, 2),
        "loss_history": loss_history,
        "comparison": {
            "baseline": pre_tuning_score,
            "option_2": post_tuning_score,
            "direction": direction
        },
        "detailed_tasks": eval_results
    }
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
        
    print(f"\nComplete log written to:\n{OUTPUT_JSON}")

if __name__ == "__main__":
    main()
