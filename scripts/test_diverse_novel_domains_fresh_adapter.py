"""
Empirical Benchmark: In-Situ Teaching of Complex Novel Domains with Fresh (Untrained) Adapter
=============================================================================================
Evaluates whether a model can be taught completely novel, complex formal systems in-situ
(systems that typically require pretraining to understand) and solve multi-step problems,
comparing Base Model (System 1) vs Dual-Loop with Fresh Untrained Adapter (System 2).

Three Challenging Synthetic Domains:
1. Exotic Non-Abelian Algebraic Reduction (Strict Non-Commutative Semigroup)
   Ground Truth Target: I (Identity)
2. Reversible Bytecode Machine (Multi-Step Stack Execution Trace)
   Ground Truth Target: [7, 8, 0]
3. Synthetic Cryptographic Hash Round (Non-Linear S-Box + Mixed Diffusion + Permutation)
   Ground Truth Target: [1, 7, 1, 7]
"""

import os
import sys
import json
import time
import re
import torch

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
from dual_loop.adapters.qwen_adapter import DualLoopQwenModel

def _resolve_model_path(model_id: str = "Qwen/Qwen3.5-2B") -> str:
    hf_cache = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.isdir(hf_cache):
        sanitized = f"models--{model_id.replace('/', '--')}"
        snapshots_dir = os.path.join(hf_cache, sanitized, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = [os.path.join(snapshots_dir, s) for s in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, s))]
            if snapshots:
                return snapshots[0]
    return model_id

MODEL_PATH = _resolve_model_path()
OUTPUT_JSON = r"eval_results/diverse_novel_domains_fresh_adapter_log.json"

TASKS = [
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
    """Evaluates whether the generated text contains the exact ground truth."""
    lower = text.lower()
    gt_lower = ground_truth.lower()
    
    is_correct = False
    extracted = None
    
    if task_id == "task_1_non_abelian_algebra":
        # Look for Final Answer: I or = I
        match = re.search(r"final answer:\s*([a-d]|i)\b", lower)
        if match:
            extracted = match.group(1).upper()
            is_correct = (extracted == "I")
        else:
            # Fallback: check if ends with I or says e = i
            is_correct = bool(re.search(r"(?:e\s*=\s*i\b|answer is\s*i\b|reduces to\s*i\b)", lower))
            extracted = "I" if is_correct else "UNKNOWN"
            
    elif task_id == "task_2_stack_bytecode_machine":
        # Look for [7, 8, 0]
        match = re.search(r"final stack:\s*\[\s*7\s*,\s*8\s*,\s*0\s*\]", lower)
        if match:
            is_correct = True
            extracted = "[7, 8, 0]"
        else:
            # Check any occurrence of [7, 8, 0] at the end
            is_correct = "[7, 8, 0]" in text
            extracted = "[7, 8, 0]" if is_correct else "MISMATCH"
            
    elif task_id == "task_3_cryptographic_hash_round":
        # Look for [1, 7, 1, 7]
        match = re.search(r"final state:\s*\[\s*1\s*,\s*7\s*,\s*1\s*,\s*7\s*\]", lower)
        if match:
            is_correct = True
            extracted = "[1, 7, 1, 7]"
        else:
            is_correct = "[1, 7, 1, 7]" in text
            extracted = "[1, 7, 1, 7]" if is_correct else "MISMATCH"
            
    return {
        "ground_truth": ground_truth,
        "extracted_answer": extracted,
        "is_correct": is_correct
    }


def main():
    print("=" * 80)
    print("EXPERIMENT: TEACHING NOVEL COMPLEX DOMAINS (BASE VS FRESH ADAPTER)")
    print("=" * 80)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Loading Base Model: {MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    base_model.eval()
    print("Base model loaded into VRAM.")
    
    # 1. Instantiate Fresh Dual-Loop Model (Untrained Adapter)
    print("\nInstantiating DualLoopQwenModel with FRESH UNTRAINED ADAPTER at layer 11 (k_steps=2)...")
    dl_model = DualLoopQwenModel(base_model, k_steps=2)
    adapter_param_count = sum(p.numel() for p in dl_model.adapter.parameters())
    print(f"Fresh adapter created with {adapter_param_count:,} parameters (NO pre-trained checkpoint loaded).")
    
    # Initially disable hook for Base Model runs
    dl_model.enabled = False
    
    all_results = []
    
    for idx, task in enumerate(TASKS, 1):
        print("\n" + "#" * 80)
        print(f"EVALUATING TASK {idx}/3: {task['name']}")
        print("#" * 80)
        
        msg = [{"role": "user", "content": task["prompt"]}]
        formatted_prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        
        # ---------------------------------------------------------------------
        # RUN A: BASE MODEL (SYSTEM 1 STANDALONE)
        # ---------------------------------------------------------------------
        print("\n--- [A] RUNNING BASE MODEL (SYSTEM 1) ---")
        dl_model.enabled = False
        
        t0 = time.perf_counter()
        with torch.no_grad():
            out_base = base_model.generate(
                **inputs,
                max_new_tokens=700,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=[248046, 248044]
            )
        latency_base = time.perf_counter() - t0
        gen_tokens_base = out_base[0].shape[0] - prompt_len
        resp_base = tokenizer.decode(out_base[0][prompt_len:], skip_special_tokens=True).strip()
        eval_base = check_answer_correctness(task["id"], task["ground_truth"], resp_base)
        
        print(f"Base Model ({gen_tokens_base} tokens, {latency_base:.2f}s, {gen_tokens_base/latency_base:.1f} tok/s):")
        print(f"Result: {eval_base['extracted_answer']} | Correct: {eval_base['is_correct']}")
        print("Tail of response:")
        print("\n".join(resp_base.splitlines()[-6:]))
        
        # ---------------------------------------------------------------------
        # RUN B: DUAL-LOOP MODEL WITH FRESH UNTRAINED ADAPTER (SYSTEM 2)
        # ---------------------------------------------------------------------
        print("\n--- [B] RUNNING DUAL-LOOP MODEL (FRESH ADAPTER KOSONGAN) ---")
        dl_model.enabled = True
        
        t0 = time.perf_counter()
        with torch.no_grad():
            out_dl = dl_model.generate(
                **inputs,
                max_new_tokens=700,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=[248046, 248044]
            )
        latency_dl = time.perf_counter() - t0
        gen_tokens_dl = out_dl[0].shape[0] - prompt_len
        resp_dl = tokenizer.decode(out_dl[0][prompt_len:], skip_special_tokens=True).strip()
        eval_dl = check_answer_correctness(task["id"], task["ground_truth"], resp_dl)
        
        print(f"Dual-Loop Fresh ({gen_tokens_dl} tokens, {latency_dl:.2f}s, {gen_tokens_dl/latency_dl:.1f} tok/s):")
        print(f"Result: {eval_dl['extracted_answer']} | Correct: {eval_dl['is_correct']}")
        print("Tail of response:")
        print("\n".join(resp_dl.splitlines()[-6:]))
        
        task_record = {
            "task_id": task["id"],
            "task_name": task["name"],
            "ground_truth": task["ground_truth"],
            "base_model": {
                "generated_tokens": gen_tokens_base,
                "latency_s": round(latency_base, 3),
                "tokens_per_second": round(gen_tokens_base / latency_base, 2),
                "response": resp_base,
                "evaluation": eval_base
            },
            "dual_loop_fresh_adapter": {
                "generated_tokens": gen_tokens_dl,
                "latency_s": round(latency_dl, 3),
                "tokens_per_second": round(gen_tokens_dl / latency_dl, 2),
                "response": resp_dl,
                "evaluation": eval_dl
            }
        }
        all_results.append(task_record)
        
    dl_model.remove_hook()
    print("\nHook removed.")
    
    # Summary Table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 80)
    base_correct_count = sum(1 for r in all_results if r["base_model"]["evaluation"]["is_correct"])
    dl_correct_count = sum(1 for r in all_results if r["dual_loop_fresh_adapter"]["evaluation"]["is_correct"])
    
    print(f"Base Model Accuracy: {base_correct_count}/{len(TASKS)} ({base_correct_count/len(TASKS)*100:.1f}%)")
    print(f"Dual-Loop (Fresh Adapter) Accuracy: {dl_correct_count}/{len(TASKS)} ({dl_correct_count/len(TASKS)*100:.1f}%)")
    
    output_data = {
        "experiment_name": "In-Situ Teaching of Complex Novel Domains with Fresh (Untrained) Adapter",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_name": "Qwen/Qwen3.5-2B",
        "adapter_state": "FRESH_UNTRAINED_KOSONGAN",
        "adapter_parameters": adapter_param_count,
        "k_steps": 2,
        "target_layer_idx": 11,
        "summary": {
            "total_tasks": len(TASKS),
            "base_model_score": f"{base_correct_count}/{len(TASKS)}",
            "base_model_accuracy_pct": round(base_correct_count / len(TASKS) * 100, 1),
            "dual_loop_fresh_score": f"{dl_correct_count}/{len(TASKS)}",
            "dual_loop_fresh_accuracy_pct": round(dl_correct_count / len(TASKS) * 100, 1)
        },
        "detailed_results": all_results
    }
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nAll detailed records and full generation text written to:\n{OUTPUT_JSON}")


if __name__ == "__main__":
    main()
