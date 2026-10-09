"""
Diagnostic Stress-Test: Systems of Reasoning (Penalaran) & Conversation (Percakapan)
=====================================================================================
Evaluates whether HADL v4.5 dynamic runtime or the underlying 2B foundation model
exhibits degradation, cognitive rigidity, language corruption, or style-bleed.

Evaluates 8 Comprehensive Multi-Turn & Single-Turn Benchmarks:
1. [Percakapan ID] Casual Greeting & Persona Check
2. [Percakapan ID] Empathy & Emotional Support Guidance
3. [Percakapan EN] Nuanced Recommendation & Creative Discourse
4. [Percakapan ID] Conceptual Pedagogical Metaphor (Internet for Kids)
5. [Penalaran Math] GSM8k Multi-Step Arithmetic Word Problem
6. [Penalaran Logic] Spatial & Positional Deductive Constraint Puzzle
7. [Penalaran Code] Semantic Root-Cause Bug Diagnosis (Mutable Default)
8. [Penalaran Algo] Algorithmic Trade-Off & Invariant Analysis (Merge vs Quick Sort)
"""

import sys
import os
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.runtime.dynamic_runtime import HADLDynamicRuntime
from dual_loop.cognitive_organs import DynamicAdaptiveConfusionSensor

MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "xstar_2b_omnireason_carlift_500q_checkpoint.pt"
JSON_OUT = PROJECT_ROOT / "eval_results" / "conversation_and_reasoning_audit.json"

TEST_PROMPTS = [
    {
        "id": "chat_01_id_persona",
        "category": "Percakapan",
        "language": "Indonesian",
        "prompt": "Halo! Siapa kamu, apa keahlian utamamu, dan bagaimana caramu bisa membantuku hari ini?",
        "checks": ["AI", "membantu", "asisten", "pemrograman"],
        "anti_patterns": ["### FILE:", "```python", "By Axiom", "Universe-K"]
    },
    {
        "id": "chat_02_id_empathy",
        "category": "Percakapan",
        "language": "Indonesian",
        "prompt": "Akhir-akhir ini saya merasa sangat burnout dan lelah dengan pekerjaan kantor. Rasanya seperti tidak punya waktu untuk diri sendiri. Apakah kamu punya saran praktis untuk menenangkan pikiran?",
        "checks": ["istirahat", "prioritas", "waktu", "kesehatan"],
        "anti_patterns": ["### FILE:", "def ", "class "]
    },
    {
        "id": "chat_03_en_creative",
        "category": "Percakapan",
        "language": "English",
        "prompt": "Could you suggest three creative, relaxing weekend hobbies for someone who spends all week looking at computer screens? Please explain why each is beneficial.",
        "checks": ["hobby", "screen", "relax", "creative"],
        "anti_patterns": ["### FILE:", "import ", "By Axiom"]
    },
    {
        "id": "chat_04_id_pedagogical",
        "category": "Percakapan",
        "language": "Indonesian",
        "prompt": "Tolong jelaskan secara sederhana dan menarik bagaimana internet bekerja kepada seorang anak berusia 10 tahun.",
        "checks": ["pesan", "surat", "komputer", "kabel", "jaringan"],
        "anti_patterns": ["TCP/IP handshake SYN-ACK", "### FILE:"]
    },
    {
        "id": "reason_01_math_word",
        "category": "Penalaran",
        "language": "English",
        "prompt": "A fruit shop had 120 apples in stock. On Monday morning, it sold 25% of them. On Tuesday, it sold 20 more apples. On Wednesday morning, a new shipment of 50 apples arrived. Exactly how many apples are in the fruit shop at the end of Wednesday? Show your step-by-step calculation.",
        "checks": ["90", "70", "120", "step"],
        "expected_answer": "120",
        "anti_patterns": []
    },
    {
        "id": "reason_02_logic_puzzle",
        "category": "Penalaran",
        "language": "English",
        "prompt": "Three people (Alice, Bob, and Charlie) are sitting next to each other on a bench in a straight line from left to right. We know that:\n1. Alice is not on the far right.\n2. Bob is immediately to the left of Charlie.\nDetermine the exact seating arrangement from left to right (Position 1, Position 2, Position 3) and explain why.",
        "checks": ["Alice", "Bob", "Charlie"],
        "expected_order": ["Alice", "Bob", "Charlie"],
        "anti_patterns": []
    },
    {
        "id": "reason_03_code_diagnosis",
        "category": "Penalaran",
        "language": "Indonesian",
        "prompt": "Perhatikan kode Python berikut:\n```python\ndef tambah_item(item, daftar=[]):\n    daftar.append(item)\n    return daftar\n```\nKetika dipanggil berulang kali, fungsi ini menghasilkan output yang tidak terduga. Jelaskan secara teknis mengapa hal tersebut terjadi dan bagaimana cara memperbaikinya secara idiomatik di Python.",
        "checks": ["mutable", "default", "None", "daftar is None"],
        "anti_patterns": ["By Axiom"]
    },
    {
        "id": "reason_04_algo_tradeoff",
        "category": "Penalaran",
        "language": "Indonesian",
        "prompt": "Bandingkan algoritma Merge Sort dan Quick Sort dalam hal: 1) Worst-case & Average-case time complexity, 2) Space complexity, 3) Stabilitas (stability). Kapan kita harus memilih Merge Sort daripada Quick Sort?",
        "checks": ["O(n log n)", "O(n^2)", "O(n)", "stabil", "memory"],
        "anti_patterns": ["### FILE:"]
    }
]

def evaluate_response(item: Dict[str, Any], text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    
    # 1. Checks matched
    checks = item.get("checks", [])
    matched_checks = [c for c in checks if c.lower() in text_lower]
    check_score = (len(matched_checks) / len(checks) * 100.0) if checks else 100.0
    
    # 2. Anti-pattern detection (Style-bleed, unintended code leaking)
    anti_patterns = item.get("anti_patterns", [])
    found_anti = [ap for ap in anti_patterns if ap.lower() in text_lower]
    leakage_penalty = len(found_anti) * 25.0
    
    # 3. Specific correctness checks
    correct = True
    if item["id"] == "reason_01_math_word":
        # Check if final answer reaches 120
        # 120 - 30 = 90; 90 - 20 = 70; 70 + 50 = 120.
        correct = "120" in text
    elif item["id"] == "reason_02_logic_puzzle":
        # Order must be Alice, Bob, Charlie
        correct = bool(re.search(r"Alice.*Bob.*Charlie", text, re.IGNORECASE))
    
    quality_score = max(0.0, min(100.0, check_score - leakage_penalty))
    if not correct:
        quality_score = min(quality_score, 60.0)
        
    return {
        "id": item["id"],
        "category": item["category"],
        "language": item["language"],
        "quality_score_pct": round(quality_score, 1),
        "checks_matched": matched_checks,
        "anti_patterns_found": found_anti,
        "is_correct_logic": correct,
        "response_length_words": len(text.split()),
        "response_preview": text[:200].replace("\n", " ") + "..."
    }

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  DIAGNOSTIC AUDIT: REASONING & CONVERSATION INTEGRITY TEST")
    print("  Evaluating: Baseline (Frozen 2B) vs. HADL v4.5 Dynamic Recurrent Runtime")
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

    results = {"baseline": {}, "hadl": {}}

    for mode in ["baseline", "hadl"]:
        print(f"\n{'='*40} TESTING MODE: {mode.upper()} {'='*40}")
        if mode == "hadl":
            runtime.attach()
        else:
            runtime.detach()

        scores = []
        for idx, item in enumerate(TEST_PROMPTS, 1):
            if mode == "hadl":
                runtime.reset_state()

            messages = [
                {"role": "system", "content": "You are a helpful, brilliant, and articulate AI assistant."},
                {"role": "user", "content": item["prompt"]}
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(device)

            with torch.no_grad():
                out = base_model.generate(
                    **inputs,
                    max_new_tokens=350,
                    temperature=0.3,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            eval_res = evaluate_response(item, decoded)
            eval_res["full_text"] = decoded
            results[mode][item["id"]] = eval_res
            scores.append(eval_res["quality_score_pct"])

            print(f"[{idx:02d}/8] {item['category']:<10} ({item['language']:<10}) [{item['id']:<22}]: Score={eval_res['quality_score_pct']:5.1f}% | Correct={eval_res['is_correct_logic']} | Leakage={len(eval_res['anti_patterns_found'])}")
            print(f"       Preview: {eval_res['response_preview']}")

        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\n[*] Mean Quality Score ({mode.upper()}): {avg_score:.1f}%")

    if mode == "hadl":
        runtime.detach()

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved conversation & reasoning audit results to: {JSON_OUT}")

if __name__ == "__main__":
    main()
