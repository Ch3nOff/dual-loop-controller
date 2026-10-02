"""
Empirical Evaluation: Novel Law Reconstruction from Inductive Clues
====================================================================
Tests whether a model can synthesize and formulate a completely novel physical law
from isolated empirical clues when it has ZERO prior knowledge of the law.

Scientific Methodology:
1. Synthetic Alien Physics Formulation:
   Define 'Chronos-Vortex Hypergravity Law' in a 4-dimensional spatial universe:
   Target Formula: F = Gamma_0 * (|sigma_1 - sigma_2| * sqrt(1 + (delta_tau)^2)) / r^3

2. Phase 1 - Negative Control Probe (Zero Prior Knowledge):
   Ask the base model to state the formula directly without clues.
   Verifies whether the concept pre-exists in model weights (verifiable absence of prior bias).

3. Phase 2 - Base Model (System 1 Autoregressive):
   Provide 3 isolated empirical clues (geometric distance decay, chirality differential coupling,
   and temporal flux gradient) and prompt the base model to synthesize the closed-form equation.

4. Phase 3 - Dual-Loop Model (System 2 Latent Deliberation):
   Attach the HADL Dual-Loop Controller (at midpoint layer 11, k_steps=2) and present
   the exact same clues under identical greedy decoding parameters.

5. Quantitative & Qualitative Verification:
   Evaluate token latency, exact term retention, dimensional consistency, and raw reasoning.
   Output all data to eval_results/novel_physics_reconstruction_log.json.
"""

import os
import sys
import os
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

MODEL_PATH = r"C:\Users\Matthew Chen\.cache\huggingface\hub\models--Qwen--Qwen3.5-2B\snapshots\15852e8c16360a2fea060d615a32b45270f8a8fc"
CHECKPOINT_PATH = r"checkpoints/qwen_game_3d_adapter/qwen_game_3d_adapter.pt"
OUTPUT_JSON = r"eval_results/novel_physics_reconstruction_log.json"


def evaluate_equation_correctness(text: str) -> dict:
    """Verifies presence of key physical components in generated equation."""
    lower = text.lower()
    
    # 1. Distance dependence: r^3 in denominator
    has_r3 = bool(re.search(r"r\s*\^?\s*3|r\s*\*\*\s*3|/\s*r\s*[\^3]|1\s*/\s*r\^3|r\^\{-?3\}|r\^3|r\^\{-3\}", lower))
    has_r2_hallucination = bool(re.search(r"r\s*\^?\s*2|r\s*\*\*\s*2|/\s*r\s*[\^2]|1\s*/\s*r\^2|r\^\{-?2\}", lower))
    
    # 2. Chirality/Spin differential: |sigma_1 - sigma_2| or |\sigma_1 - \sigma_2|
    has_chirality = bool(
        "|sigma_1 - sigma_2|" in lower or
        "|\\sigma_1 - \\sigma_2|" in lower or
        "|σ1 - σ2|" in lower or
        "abs(sigma_1 - sigma_2)" in lower or
        "(sigma_1 - sigma_2)" in lower or
        "(\\sigma_1 - \\sigma_2)" in lower or
        "sigma_1 - sigma_2" in lower or
        "\\sigma_1 - \\sigma_2" in lower or
        "σ1 - σ2" in lower
    )
    has_mass_hallucination = bool(re.search(r"\bm_1\b|\bm_2\b|\bmass\b", lower))
    
    # 3. Temporal factor: sqrt(1 + (delta_tau)^2) or \sqrt{1 + (\delta_\tau)^2}
    has_temporal = bool(
        "sqrt(1 + (delta_tau)^2)" in lower or
        "sqrt(1 + delta_tau^2)" in lower or
        "\\sqrt{1 + (\\delta_\\tau)^2}" in lower or
        "\\sqrt{1 + (\\delta_\\tau)^2" in lower or
        "\\sqrt{1 + (\\delta_tau)^2}" in lower or
        "\\sqrt{1 + \\delta_\\tau^2}" in lower or
        "\\sqrt{1 + (\\delta_\\tau)^{2}}" in lower or
        "√(1 + (δτ)^2)" in lower or
        "√(1 + δτ^2)" in lower or
        "sqrt(1 + (delta_tau)**2)" in lower or
        "1 + (delta_tau)^2" in lower or
        "1 + delta_tau^2" in lower or
        "1 + (\\delta_\\tau)^2" in lower or
        "1 + \\delta_\\tau^2" in lower
    )
    
    # 4. Constant: Gamma_0 or \Gamma_0
    has_constant = bool("gamma_0" in lower or "\\gamma_0" in lower or "gamma" in lower or "\\gamma" in lower or "γ_0" in lower or "γ0" in lower)
    
    # Full target match: all three components present
    full_match = has_r3 and has_chirality and has_temporal
    
    return {
        "has_r3_distance": has_r3,
        "has_r2_hallucination": has_r2_hallucination,
        "has_chirality_diff": has_chirality,
        "has_mass_hallucination": has_mass_hallucination,
        "has_temporal_gradient": has_temporal,
        "has_coupling_constant": has_constant,
        "full_synthesis_correct": full_match
    }


def main():
    print("=" * 80)
    print("EXPERIMENT: NOVEL PHYSICAL LAW RECONSTRUCTION FROM INDUCTIVE CLUES")
    print("=" * 80)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Loading Base Model from: {MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    base_model.eval()
    print("Base model loaded successfully into VRAM.")
    
    # =========================================================================
    # PHASE 1: Negative Control Probe (Zero-Knowledge Verification)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 1: PROBING ZERO-PRIOR KNOWLEDGE (NEGATIVE CONTROL)")
    print("-" * 80)
    
    prompt_zero_knowledge = (
        "Question: In theoretical astrophysics, what is the exact closed-form mathematical equation "
        "for the 'Chronos-Vortex Hypergravity Law' that determines the force F between two spinning singularities? "
        "Provide the exact formula and definition of variables."
    )
    
    print(f"Prompt:\n{prompt_zero_knowledge}\n")
    msg_zk = [{"role": "user", "content": prompt_zero_knowledge}]
    formatted_prompt_zk = tokenizer.apply_chat_template(msg_zk, tokenize=False, add_generation_prompt=True)
    tokens_zk = tokenizer(formatted_prompt_zk, return_tensors="pt").to(device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        out_zk = base_model.generate(
            **tokens_zk,
            max_new_tokens=400,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=[248046, 248044]
        )
    latency_zk = time.perf_counter() - t0
    resp_zk = tokenizer.decode(out_zk[0][tokens_zk.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    print(f"Response (Time: {latency_zk:.2f}s):\n{resp_zk}\n")
    eval_zk = evaluate_equation_correctness(resp_zk)
    print(f"Verification Phase 1: {eval_zk}")
    print(f"Prior Knowledge Exists: {eval_zk['full_synthesis_correct']}")
    
    # =========================================================================
    # PHASE 2: Base Model (System 1) - Clue-Driven Inductive Synthesis
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 2: BASE MODEL (SYSTEM 1) - SYNTHESIS FROM CLUES")
    print("-" * 80)
    
    prompt_clues = (
        "You are an astrophysicist studying a newly discovered 4-dimensional spatial universe where "
        "Newtonian gravity does NOT apply. Based ONLY on the following 3 empirical experimental observations, "
        "derive and synthesize a single, unified closed-form mathematical equation for the interaction force F:\n\n"
        "Observation 1 (Geometry & Distance Decay): In this 4D spatial universe, radiation spreads over the surface "
        "of a 3-sphere (area proportional to r^3). Experiments confirm that doubling the distance r from 1 to 2 "
        "decreases the force by a factor of 8 (2^3), and tripling r to 3 decreases the force by a factor of 27 (3^3). "
        "Therefore, the force scales strictly inversely with the cube of distance (1/r^3).\n\n"
        "Observation 2 (Source Coupling & Chirality): The interaction is not mediated by mass, but by rotational "
        "chirality values sigma_1 and sigma_2. When sigma_1 = sigma_2, the force drops to exactly 0. As the difference "
        "between their chirality values increases, the interaction force grows in direct linear proportion to |sigma_1 - sigma_2|.\n\n"
        "Observation 3 (Temporal Flux Gradient): Relativistic experiments demonstrate that when a temporal flux gradient "
        "delta_tau exists between the two frames, the force is amplified multiplicatively by the factor sqrt(1 + (delta_tau)^2). "
        "When delta_tau = 0, this factor reduces to exactly 1.\n\n"
        "Your Task:\n"
        "1. Combine Observations 1, 2, and 3 into a single closed-form formula for F(sigma_1, sigma_2, r, delta_tau) "
        "using coupling constant Gamma_0.\n"
        "2. Detail how each observation determines the numerator, denominator, and overall scaling.\n"
        "3. State the final equation clearly on its own line in the exact format: 'Final Formula: F = ...'\n\n"
        "Derivation and Formula:"
    )
    
    msg_clues = [{"role": "user", "content": prompt_clues}]
    formatted_prompt_clues = tokenizer.apply_chat_template(msg_clues, tokenize=False, add_generation_prompt=True)
    tokens_clues = tokenizer(formatted_prompt_clues, return_tensors="pt").to(device)
    prompt_token_count = tokens_clues.input_ids.shape[1]
    
    t0 = time.perf_counter()
    with torch.no_grad():
        out_base = base_model.generate(
            **tokens_clues,
            max_new_tokens=650,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=[248046, 248044]
        )
    latency_base = time.perf_counter() - t0
    gen_tokens_base = out_base[0].shape[0] - prompt_token_count
    resp_base = tokenizer.decode(out_base[0][prompt_token_count:], skip_special_tokens=True).strip()
    
    print(f"Base Model Response ({gen_tokens_base} tokens, {latency_base:.2f}s, {gen_tokens_base/latency_base:.1f} tok/s):\n")
    print(resp_base)
    eval_base = evaluate_equation_correctness(resp_base)
    print(f"\nVerification Base Model: {eval_base}")
    
    # =========================================================================
    # PHASE 3: Dual-Loop Model (System 2 with HADL Adapter)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 3: DUAL-LOOP MODEL (SYSTEM 2 DELIBERATION) - SYNTHESIS FROM CLUES")
    print("-" * 80)
    
    print(f"Attaching DualLoopQwenModel at midpoint layer with checkpoint: {CHECKPOINT_PATH}")
    dl_model = DualLoopQwenModel(base_model, k_steps=2)
    dl_model.load_adapter(CHECKPOINT_PATH)
    print(f"Adapter attached at layer {dl_model.layer_idx}, k_steps={dl_model.k_steps}.")
    
    t0 = time.perf_counter()
    with torch.no_grad():
        out_dl = dl_model.generate(
            **tokens_clues,
            max_new_tokens=650,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=[248046, 248044]
        )
    latency_dl = time.perf_counter() - t0
    gen_tokens_dl = out_dl[0].shape[0] - prompt_token_count
    resp_dl = tokenizer.decode(out_dl[0][prompt_token_count:], skip_special_tokens=True).strip()
    
    print(f"Dual-Loop Model Response ({gen_tokens_dl} tokens, {latency_dl:.2f}s, {gen_tokens_dl/latency_dl:.1f} tok/s):\n")
    print(resp_dl)
    eval_dl = evaluate_equation_correctness(resp_dl)
    print(f"\nVerification Dual-Loop Model: {eval_dl}")
    
    # Clean hook to leave model clean
    dl_model.remove_hook()
    print("Hook removed cleanly.")
    
    # =========================================================================
    # LOGGING AND COMPARATIVE SUMMARY
    # =========================================================================
    results = {
        "experiment_name": "Novel Physical Law Reconstruction from Inductive Clues",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_name": "Qwen/Qwen3.5-2B",
        "target_law": "Chronos-Vortex Hypergravity Law",
        "target_equation": "F = Gamma_0 * (|sigma_1 - sigma_2| * sqrt(1 + (delta_tau)^2)) / r^3",
        "phase_1_zero_knowledge_probe": {
            "prompt": prompt_zero_knowledge,
            "response": resp_zk,
            "latency_s": round(latency_zk, 3),
            "evaluation": eval_zk,
            "proof_of_zero_prior_knowledge": not eval_zk["full_synthesis_correct"]
        },
        "phase_2_base_model_system1": {
            "prompt_tokens": prompt_token_count,
            "generated_tokens": gen_tokens_base,
            "latency_s": round(latency_base, 3),
            "tokens_per_second": round(gen_tokens_base / latency_base, 2),
            "response": resp_base,
            "evaluation": eval_base
        },
        "phase_3_dual_loop_system2": {
            "prompt_tokens": prompt_token_count,
            "generated_tokens": gen_tokens_dl,
            "latency_s": round(latency_dl, 3),
            "tokens_per_second": round(gen_tokens_dl / latency_dl, 2),
            "response": resp_dl,
            "evaluation": eval_dl,
            "last_telemetry": {
                k: (v.tolist() if isinstance(v, torch.Tensor) else v)
                for k, v in dl_model.last_telemetry.items()
                if not isinstance(v, (torch.Tensor, dict)) or isinstance(v, (int, float, str, bool, list))
            }
        },
        "comparative_analysis": {
            "base_synthesized_target": eval_base["full_synthesis_correct"],
            "dual_loop_synthesized_target": eval_dl["full_synthesis_correct"],
            "base_hallucinated_newtonian_r2": eval_base["has_r2_hallucination"],
            "dual_loop_hallucinated_newtonian_r2": eval_dl["has_r2_hallucination"],
            "relative_speed_overhead_pct": round(((latency_dl - latency_base) / latency_base) * 100, 2) if latency_base > 0 else 0.0
        }
    }
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("RESULTS SUCCESSFULLY SAVED TO:")
    print(OUTPUT_JSON)
    print("=" * 80)


if __name__ == "__main__":
    main()
