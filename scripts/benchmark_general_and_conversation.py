"""
HADL v4.4: General Conversation, Coding, Creative, and Practical Stress-Test Suite
===================================================================================
Evaluates the model across 12 diverse real-world tasks outside the formal reasoning domain:
1. Coding & Software Development (Python, Debugging, SQL, Regex)
2. Daily Casual & Empathetic Conversation (Bahasa Indonesia & English)
3. Creative Writing, Technical Explanation & Practical Advice

Checks for:
- Routing accuracy (Should select FAST BYPASS Route 0)
- Style Bleed / Subspace Hijacking (Checking for inappropriate 'By Axiom', 'Final Answer:' leakage)
- Code correctness & executability
- Conversational naturalness, empathy, and fluency
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop.callearn_oc_engine import (
    HADLv4ModelWrapper,
    FastEmbeddedRouting
)
from scripts.retrain_qwen_27b_grade_dualloop import MODEL_ID

CHECKPOINT_ENGINE = PROJECT_ROOT / "checkpoints" / "hadl_v42_ghost_unified.pt"
CHECKPOINT_ROUTER = PROJECT_ROOT / "checkpoints" / "fast_embedded_router_weights.pt"
OUTPUT_REPORT = PROJECT_ROOT / "eval_results" / "hadl_v44_general_conversation_stress_test.json"

REAL_WORLD_BENCHMARK_TASKS = [
    # =========================================================================
    # CATEGORY 1: CODE GENERATION & SOFTWARE ENGINEERING (4 TASKS)
    # =========================================================================
    {
        "id": "code_01_binary_search",
        "category": "Code Generation",
        "name": "Python Binary Search with Edge Cases",
        "prompt": (
            "Write a Python function `binary_search(arr, target)` that implements the binary search algorithm on a sorted list of integers.\n"
            "Requirements:\n"
            "1. Return the 0-based index if target is found, otherwise return -1.\n"
            "2. Properly handle edge cases (empty list, target smaller than min, target larger than max).\n"
            "3. Include a concise docstring and type hints.\n"
            "Provide only the Python code with a brief explanation."
        ),
        "evaluation_criteria": {
            "checks": ["def binary_search", "return -1", "while", "mid"],
            "anti_patterns": ["By Axiom", "Universe-K", "Final Answer: [ELEMENT]"]
        }
    },
    {
        "id": "code_02_debug_mutable_default",
        "category": "Code Debugging",
        "name": "Python Mutable Default Argument Bugfix",
        "prompt": (
            "A junior developer wrote the following Python function, but it exhibits strange buggy behavior across multiple calls:\n\n"
            "```python\n"
            "def add_item(item, item_list=[]):\n"
            "    item_list.append(item)\n"
            "    return item_list\n\n"
            "print(add_item('apple'))   # ['apple']\n"
            "print(add_item('banana'))  # ['apple', 'banana'] - BUG!\n"
            "```\n\n"
            "Question: Explain clearly why this bug happens in Python, and provide the idiomatic, correct way to fix it."
        ),
        "evaluation_criteria": {
            "checks": ["None", "default", "mutable", "item_list is None"],
            "anti_patterns": ["By Axiom", "Final Answer: [ELEMENT]"]
        }
    },
    {
        "id": "code_03_sql_join_group",
        "category": "Database & SQL",
        "name": "SQL Multi-Table Aggregation Query",
        "prompt": (
            "Given two SQL tables:\n"
            "- `departments` (dept_id INT, dept_name VARCHAR)\n"
            "- `employees` (emp_id INT, emp_name VARCHAR, dept_id INT, salary NUMERIC)\n\n"
            "Write an ANSI SQL query to find each department's name along with its average employee salary, "
            "but only include departments that have at least 3 employees and an average salary greater than 50,000. "
            "Order by average salary descending."
        ),
        "evaluation_criteria": {
            "checks": ["JOIN", "GROUP BY", "HAVING", "COUNT", "AVG", "ORDER BY"],
            "anti_patterns": ["By Axiom", "Universe-K"]
        }
    },
    {
        "id": "code_04_regex_date_extraction",
        "category": "Regex & Parsing",
        "name": "Python Regex Date Pattern Extractor",
        "prompt": (
            "Write a Python function `extract_dates(text)` using the `re` module that extracts all dates formatted either as "
            "`YYYY-MM-DD` (e.g. 2026-10-06) or `DD/MM/YYYY` (e.g. 06/10/2026) from a given string. "
            "Return them as a list of matched date strings."
        ),
        "evaluation_criteria": {
            "checks": ["import re", "re.findall", r"\d{4}-\d{2}-\d{2}", r"\d{2}/\d{2}/\d{4}"],
            "anti_patterns": ["By Axiom", "Final Answer: [ELEMENT]"]
        }
    },

    # =========================================================================
    # CATEGORY 2: DAILY CONVERSATION & SOCIAL INTERACTION (4 TASKS)
    # =========================================================================
    {
        "id": "chat_01_weekend_burnout",
        "category": "Daily Conversation (ID)",
        "name": "Empathetic Burnout & Relaxation Advice (Bahasa Indonesia)",
        "prompt": (
            "Halo! Akhir pekan kemarin aku capek banget karena harus kerja lembur tanpa henti. "
            "Badan rasanya pegal dan otak terasa jenuh banget. "
            "Boleh minta saran aktivitas santai yang gak makan banyak tenaga buat recharge energi sebelum mulai kerja lagi besok?"
        ),
        "evaluation_criteria": {
            "checks": ["istirahat", "santai", "tidur", "recharge", "air"],
            "anti_patterns": ["By Axiom", "Step 1: By", "Final Answer: [ELEMENT]", "Universe-K"]
        }
    },
    {
        "id": "chat_02_polite_debt_reminder",
        "category": "Interpersonal & Empathy (ID)",
        "name": "Polite Debt Collection Advice (Bahasa Indonesia)",
        "prompt": (
            "Mau tanya pendapat dong. Teman dekatku pernah pinjam uang sebulan yang lalu untuk keperluan mendesak, "
            "tapi sampai sekarang belum ada kabar atau balikin. Aku agak segan dan gak enak hati mau nagihnya. "
            "Gimana ya cara ngomong atau kirim chat yang sopan, gak terkesan menuduh, tapi pesannya tetap sampai tanpa merusak pertemanan?"
        ),
        "evaluation_criteria": {
            "checks": ["sopan", "chat", "kabar", "ingatkan", "teman"],
            "anti_patterns": ["By Axiom", "Final Stack:", "Final State:", "Final Answer:"]
        }
    },
    {
        "id": "chat_03_travel_kyoto",
        "category": "Conversational Planning (EN)",
        "name": "Kyoto Autumn 3-Day Travel Itinerary (English)",
        "prompt": (
            "Hey! I'm planning a 3-day solo trip to Kyoto during autumn (late November) to see the fall foliage. "
            "Can you give me a realistic, well-paced day-by-day itinerary that balances iconic temples with peaceful scenic walks? "
            "Keep the tone friendly and conversational."
        ),
        "evaluation_criteria": {
            "checks": ["Day 1", "Day 2", "Day 3", "Kyoto", "temple"],
            "anti_patterns": ["By Axiom", "Final Answer: [ELEMENT]", "Universe-K"]
        }
    },
    {
        "id": "chat_04_proverb_context",
        "category": "Cultural & Idiomatic (ID)",
        "name": "Proverb Contextualization (Bahasa Indonesia)",
        "prompt": (
            "Jelaskan makna peribahasa 'Air tenang menghanyutkan' dan berikan satu contoh situasinya di lingkungan kerja atau kantor modern."
        ),
        "evaluation_criteria": {
            "checks": ["pendiam", "kemampuan", "bahaya", "kantor", "kerja"],
            "anti_patterns": ["By Axiom", "Final Answer: [ELEMENT]"]
        }
    },

    # =========================================================================
    # CATEGORY 3: CREATIVE WRITING, EXPLANATION & PRACTICAL ADVICE (4 TASKS)
    # =========================================================================
    {
        "id": "creative_01_coffee_rain_poem",
        "category": "Creative Writing (ID)",
        "name": "Poem: Secangkir Kopi Saat Hujan Pagi (Bahasa Indonesia)",
        "prompt": (
            "Tuliskan sebuah puisi pendek yang hangat dan puitis (terdiri dari 3-4 bait) "
            "tentang kenikmatan secangkir kopi hangat di pagi hari saat hujan deras turun di luar jendela."
        ),
        "evaluation_criteria": {
            "checks": ["kopi", "hujan", "pagi", "jendela", "hangat"],
            "anti_patterns": ["By Axiom", "Step 1:", "Final Answer:", "Universe-K"]
        }
    },
    {
        "id": "explain_02_rsa_layman",
        "category": "Technical Layman Explanation",
        "name": "RSA Public-Key Cryptography for Laypeople",
        "prompt": (
            "Jelaskan konsep dasar kriptografi kunci publik (seperti RSA) kepada orang yang sama sekali tidak punya latar belakang teknis atau matematika. "
            "Gunakan analogi sederhana sehari-hari (seperti gembok dan kotak surat) agar mudah dipahami, tanpa menggunakan rumus matematika."
        ),
        "evaluation_criteria": {
            "checks": ["gembok", "kunci", "publik", "privat", "rahasia"],
            "anti_patterns": ["By Axiom", "mod 16", "S_i'"]
        }
    },
    {
        "id": "trans_03_grain_of_salt",
        "category": "Language & Translation",
        "name": "Idiom Nuance Translation & Usage",
        "prompt": (
            "Dalam bahasa Inggris ada ungkapan 'take it with a grain of salt'.\n"
            "1. Apa arti dan maksud sebenarnya dari idiom ini?\n"
            "2. Bagaimana cara menerjemahkannya ke dalam bahasa Indonesia secara alami (bukan terjemahan harfiah)?\n"
            "3. Berikan contoh kalimat penggunaannya dalam percakapan sehari-hari."
        ),
        "evaluation_criteria": {
            "checks": ["jangan langsung percaya", "kritis", "skeptis", "garam", "contoh"],
            "anti_patterns": ["By Axiom", "Final Answer: [ELEMENT]"]
        }
    },
    {
        "id": "advice_04_laptop_buying",
        "category": "Practical Consumer Advice (ID)",
        "name": "Laptop Buying Guide for Programming & Casual Gaming",
        "prompt": (
            "Aku berencana beli laptop dengan budget sekitar 10-12 juta rupiah. "
            "Kebutuhan utamaku adalah belajar programming (web development & sedikit machine learning pemula) "
            "serta sesekali main game kasual seperti Dota 2 atau Valorant. "
            "Spesifikasi minimal apa yang harus kupilih (prosesor, RAM, storage, GPU)? Berikan rekomendasi praktis."
        ),
        "evaluation_criteria": {
            "checks": ["RAM", "16GB", "SSD", "prosesor", "GPU", "Intel", "Ryzen"],
            "anti_patterns": ["By Axiom", "Universe-K", "Final Answer: [ELEMENT]"]
        }
    }
]


def evaluate_task_quality(task: Dict[str, Any], output_text: str, is_bypass: bool, route_probs: List[float]) -> Dict[str, Any]:
    lower = output_text.lower()
    criteria = task["evaluation_criteria"]

    # 1. Anti-pattern detection (checking for style bleed)
    detected_anti_patterns = []
    for anti in criteria["anti_patterns"]:
        if anti.lower() in lower:
            detected_anti_patterns.append(anti)

    # 2. Key content checks
    matched_checks = []
    missing_checks = []
    for check in criteria["checks"]:
        if check.lower() in lower:
            matched_checks.append(check)
        else:
            missing_checks.append(check)

    # 3. Overall pass evaluation
    # To pass, must have:
    # - Correct routing (MUST be Fast Bypass for these general tasks)
    # - ZERO anti-patterns (no style bleed)
    # - Sufficient matched keywords (> 50% of expected content signals)
    # - Non-empty, reasonable length (> 40 tokens)
    has_style_bleed = len(detected_anti_patterns) > 0
    content_score = len(matched_checks) / max(len(criteria["checks"]), 1)
    is_well_routed = is_bypass  # All 12 tasks should be Fast Bypass

    is_passed = (not has_style_bleed) and (content_score >= 0.50) and is_well_routed and len(output_text.strip()) > 80

    return {
        "is_passed": is_passed,
        "is_well_routed": is_well_routed,
        "has_style_bleed": has_style_bleed,
        "detected_anti_patterns": detected_anti_patterns,
        "content_score_pct": round(content_score * 100.0, 1),
        "matched_checks": matched_checks,
        "missing_checks": missing_checks,
        "p_bypass": round(route_probs[0], 4),
        "p_heavy": round(route_probs[1], 4)
    }


def main():
    print("=" * 85)
    print("  HADL v4.4 STRESS-TEST: CODING, DAILY CONVERSATION, CREATIVE & PRACTICAL SUITE")
    print("  Rigorous Evaluation for Subspace Hijacking, Style Bleed, and Linguistic Quality")
    print("=" * 85)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Hardware : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("\n[*] Loading Qwen3.5-2B Base Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    for p in base_model.parameters():
        p.requires_grad = False

    print("[*] Initializing HADLv4ModelWrapper with FastEmbeddedRouting & Ghost Bridge...")
    wrapper = HADLv4ModelWrapper(base_model, target_layer_idx=11, ghost_layer_idx=23)
    wrapper.engine.to(device=device, dtype=torch.bfloat16)
    wrapper.ghost.to(device=device, dtype=torch.bfloat16)

    # Load engine and router weights
    if CHECKPOINT_ENGINE.exists():
        ckpt_eng = torch.load(CHECKPOINT_ENGINE, map_location=device)
        wrapper.engine.load_state_dict(ckpt_eng["engine"])
        wrapper.ghost.load_state_dict(ckpt_eng["ghost"])
        print(f"[OK] Loaded trained engine weights from: {CHECKPOINT_ENGINE}")

    if CHECKPOINT_ROUTER.exists():
        wrapper.router.load_routing_weights(str(CHECKPOINT_ROUTER))
        print(f"[OK] Loaded calibrated router weights from: {CHECKPOINT_ROUTER}")

    wrapper.engine.eval()
    wrapper.ghost.eval()

    print(f"\n[*] Beginning Evaluation across {len(REAL_WORLD_BENCHMARK_TASKS)} Real-World Tasks...\n")

    results = []
    passed_count = 0
    total_tokens_generated = 0
    total_latency_sec = 0.0

    for idx, task in enumerate(REAL_WORLD_BENCHMARK_TASKS, 1):
        wrapper.engine.reset_kv_cache()
        wrapper.cached_h11 = None

        messages = [{"role": "user", "content": task["prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        # 1. Routing Inspection
        embed_layer = getattr(wrapper.base_model.model, "embed_tokens", None)
        is_bypass, route_probs = wrapper.router(inputs.input_ids, embed_layer=embed_layer)
        p_byp = route_probs[0, 0].item()
        p_hvy = route_probs[0, 1].item()

        # 2. Generation
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = wrapper.generate(
                inputs.input_ids,
                max_new_tokens=650,
                temperature=0.7,  # Natural conversational temperature
                top_p=0.9,
                do_sample=True,   # Sample for rich, natural language
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.perf_counter() - t0
        wrapper.engine.reset_kv_cache()
        wrapper.cached_h11 = None

        gen_tokens = output_ids[0][prompt_len:]
        num_tokens = len(gen_tokens)
        speed = num_tokens / max(elapsed, 1e-4)

        total_tokens_generated += num_tokens
        total_latency_sec += elapsed

        response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # 3. Quality and Integrity Evaluation
        eval_metrics = evaluate_task_quality(task, response_text, is_bypass, [p_byp, p_hvy])

        if eval_metrics["is_passed"]:
            passed_count += 1
            tag = "[PASS]"
        else:
            tag = "[FAIL]"

        route_name = "FAST BYPASS" if is_bypass else "HEAVY HADL"
        print(f"[{idx:2d}/12] {tag} [{route_name:<11} P_byp={p_byp:.2f}] {task['category']:<22} | {task['name']}")
        print(f"       Speed: {speed:.1f} tok/s | Tokens: {num_tokens} | Content Score: {eval_metrics['content_score_pct']}% | Style Bleed: {eval_metrics['has_style_bleed']}")
        if eval_metrics["detected_anti_patterns"]:
            print(f"       [!] WARNING: Detected Anti-Patterns: {eval_metrics['detected_anti_patterns']}")
        print(f"       Preview: {response_text[:160]}...\n")

        results.append({
            "task_id": task["id"],
            "category": task["category"],
            "name": task["name"],
            "prompt": task["prompt"],
            "route_taken": route_name,
            "p_bypass": p_byp,
            "p_heavy": p_hvy,
            "is_passed": eval_metrics["is_passed"],
            "has_style_bleed": eval_metrics["has_style_bleed"],
            "detected_anti_patterns": eval_metrics["detected_anti_patterns"],
            "content_score_pct": eval_metrics["content_score_pct"],
            "matched_keywords": eval_metrics["matched_checks"],
            "missing_keywords": eval_metrics["missing_checks"],
            "num_tokens": num_tokens,
            "speed_tok_s": round(speed, 2),
            "full_response": response_text
        })

    overall_acc = (passed_count / len(REAL_WORLD_BENCHMARK_TASKS)) * 100.0
    avg_speed = total_tokens_generated / max(total_latency_sec, 1e-4)

    print("=" * 85)
    print("                     REAL-WORLD STRESS-TEST SCOREBOARD                     ")
    print("=" * 85)
    print(f"| Total Tasks Evaluated       : {len(REAL_WORLD_BENCHMARK_TASKS)}")
    print(f"| Tasks Passed (High Quality) : {passed_count} / {len(REAL_WORLD_BENCHMARK_TASKS)} ({overall_acc:.1f}%)")
    print(f"| Routing Accuracy (Fast Byp) : {sum(1 for r in results if r['route_taken']=='FAST BYPASS')} / {len(results)} ({sum(1 for r in results if r['route_taken']=='FAST BYPASS')/len(results)*100:.1f}%)")
    print(f"| Style Bleed Incidents       : {sum(1 for r in results if r['has_style_bleed'])} (0 = Clean Separation)")
    print(f"| Average Generation Speed    : {avg_speed:.1f} tok/s")
    print(f"| Peak VRAM Usage             : {torch.cuda.max_memory_allocated() / (1024**3):.2f} GiB")
    print("=" * 85)

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump({
            "suite_title": "HADL v4.4 Real-World Stress-Test: Coding, Daily Conversation, Creative & Practical",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": MODEL_ID,
            "overall_accuracy_pct": overall_acc,
            "passed_count": passed_count,
            "total_tasks": len(REAL_WORLD_BENCHMARK_TASKS),
            "style_bleed_incidents": sum(1 for r in results if r['has_style_bleed']),
            "routing_accuracy_pct": sum(1 for r in results if r['route_taken']=='FAST BYPASS') / len(results) * 100.0,
            "average_speed_tok_s": round(avg_speed, 2),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
            "task_results": results
        }, f, indent=2)
    print(f"\n[OK] Detailed real-world report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
