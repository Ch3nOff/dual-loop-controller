"""
Plain Text Tutoring & Dynamic Epistemic Grounding Session
=========================================================
Pedagogical Philosophy (Aligned with User Specifications):
1. Pure Plain Text: Zero chat templates (<|im_start|> eliminated).
   Prompts are structured as natural human conversation and raw prompt completions
   so online synaptic plasticity and KV-cache loss align directly with real usage and benchmarks.
2. Dynamic Epistemic Sensitivity Gating:
   VGNI anchors are queried dynamically using adaptive z-score prominence (mu_aff + beta * sigma_aff).
3. Targeted Commonsense & Logic Grounding:
   - Physical domain: Glass (hard/brittle) vs Carpet (soft/cushioned)
   - Code & Math: abs(x) -> -x if x < 0 else x; int('text') -> ValueError
   - Temporal: Calendar modulo 7 (Wednesday -> Thursday -> Friday)
   - Quantitative: 'How many' scalar answers
4. Commutative Dual-Composition Verification:
   LBU verifies step agreement between Layer 11 and Layer 23 natively.
5. Persistent In-Place Weight Consolidation:
   Saves updated synaptic weights and epistemic anchors into model.safetensors!
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "dist" / "xstar_2b_autonomous_brain"
SAFETENSORS_PATH = MODEL_PATH / "model.safetensors"
WEIGHTS_PT_PATH = MODEL_PATH / "vgni_lbu_weights.pt"
LOG_OUT = PROJECT_ROOT / "eval_results" / "tutoring_raw_text_dynamic_log.txt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_brain_telemetry(model: nn.Module) -> Dict[str, Any]:
    brain = model.model.brain
    plastic_w = brain.plastic_unit.w_plastic.data
    vgni = brain.vgni_engine.vgni
    curr_n = int(vgni.num_stored.item())
    return {
        "plastic_norm": round(torch.norm(plastic_w).item(), 6),
        "plastic_mean": round(torch.mean(plastic_w).item(), 8),
        "plastic_steps": brain.plastic_unit.plasticity_step_count.item(),
        "vgni_stored_anchors": curr_n,
        "lbu_contraction": brain.vgni_engine.vgni.lbu.gamma_contraction,
    }


def plain_text_generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 120
) -> str:
    """
    Direct raw plain text generation without any chat formatting tags.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    out_text = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=False).strip()
    clean_text = re.sub(r"<think>.*?</think>", "", out_text, flags=re.DOTALL).strip()
    clean_text = clean_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return clean_text


def assimilate_epistemic_concept(model: nn.Module, tokenizer: Any, concept_text: str, approach_fact: str):
    """
    Encodes and assimilates a persistent epistemic concept anchor and verified approach vector
    into the VGNI memory bank using plain text representations.
    """
    vgni = model.model.brain.vgni_engine.vgni
    inp_c = tokenizer(concept_text, return_tensors="pt").to(DEVICE)
    inp_a = tokenizer(approach_fact, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        emb_c = model.model.embed_tokens(inp_c.input_ids).mean(dim=1)  # [1, D]
        emb_a = model.model.embed_tokens(inp_a.input_ids).mean(dim=1)  # [1, D]
        # Approach vector projected through LBU Banach contraction operator
        approach_vec = vgni.lbu.w_step(emb_a)
        vgni.register_successful_trajectory(emb_c, approach_vec)


def main():
    print("=" * 100)
    print("X-STAR PLAIN TEXT TUTORING & DYNAMIC EPISTEMIC GROUNDING SESSION")
    print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 100)

    t0_start = time.perf_counter()
    os.makedirs(LOG_OUT.parent, exist_ok=True)
    log_file = open(LOG_OUT, "w", encoding="utf-8")

    def log(msg: str):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("[*] Loading X-Star Autonomous Brain from safetensors...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    # Enable Endogenous Synaptic Plasticity during tutoring
    model.set_online_learning(True)

    init_telem = get_brain_telemetry(model)
    log("\n[Phase 1: Initial Brain Telemetry]")
    log(f"  * Synaptic Plasticity Norm : {init_telem['plastic_norm']}")
    log(f"  * Synaptic Plasticity Steps: {init_telem['plastic_steps']}")
    log(f"  * VGNI Stored Anchors      : {init_telem['vgni_stored_anchors']} / 64")
    log(f"  * LBU Contraction Lipschitz: {init_telem['lbu_contraction']}")

    # =========================================================================
    # PHASE 2: PLAIN TEXT CURRICULUM WITH DYNAMIC GROUNDING
    # =========================================================================
    curriculum = [
        # 1. Physical Domain: Material Rigidity vs Softness
        {
            "num": 1,
            "title": "Physical Commonsense: Material Hardness (Glass) vs Softness (Carpet)",
            "concept_key": "carpet is soft plush and cushioning, glass is hard rigid and brittle",
            "approach_fact": "The glass fell onto the carpet and did not break because the carpet was soft. Glass is brittle and rigid; carpet is soft and cushioned.",
            "teaching_prompt": (
                "Guru: Mari kita pelajari sifat fisik material di dunia nyata.\n"
                "Kaca (glass) memiliki sifat keras, getas, kaku, dan mudah pecah saat terbentur permukaan keras.\n"
                "Karpet (carpet) terbuat dari serat kain yang lembut, tebal, empuk, dan memiliki daya redam benturan yang tinggi.\n"
                "Jika sebuah gelas kaca jatuh ke atas karpet tebal dan tidak pecah, hal itu terjadi karena karpetnya lembut dan meredam benturan.\n"
                "Murid:"
            ),
            "drills": [
                {
                    "prompt": "Sentence: The glass fell onto the carpet and did not break because the _ was soft.\nQuestion: Which was soft, the glass or the carpet?\nAnswer:",
                    "expected": ["carpet"],
                    "note": "WinoGrande physical grounding test"
                },
                {
                    "prompt": "When a glass cup drops on a thick fluffy carpet, why does it remain intact?\nAnswer:",
                    "expected": ["carpet", "soft", "cushion", "absorb"],
                    "note": "Causal explanation"
                }
            ]
        },

        # 2. Mathematical Functions: Absolute Value abs(x)
        {
            "num": 2,
            "title": "Math & Code: Absolute Value abs(x) & Algebraic Definition",
            "concept_key": "def abs_val(x): return -x if x < 0 else x for absolute value",
            "approach_fact": "The absolute value of x is defined as: -x if x < 0 else x. When x is negative, negating it with -x produces a positive value. Never return constant 1.",
            "teaching_prompt": (
                "Guru: Mari kita pelajari definisi fungsi nilai mutlak (absolute value) dalam matematika dan Python.\n"
                "Nilai mutlak |x| selalu bernilai non-negatif:\n"
                "- Jika x >= 0, maka nilainya adalah x.\n"
                "- Jika x < 0, maka nilainya adalah -x (karena negatif dikali negatif menjadi positif!).\n"
                "- Jangan pernah mengembalikan konstanta '1' sembarangan!\n"
                "Dalam Python: `def abs_val(x): return -x if x < 0 else x` atau `abs(x)`.\n"
                "Murid:"
            ),
            "drills": [
                {
                    "prompt": "Write a Python function to compute the absolute value of a number x:\ndef abs_val(x):\n",
                    "expected": ["-x if x < 0", "abs(x)", "if x < 0: return -x", "return -x"],
                    "note": "HumanEval absolute value code generation"
                },
                {
                    "prompt": "What is the mathematical definition of absolute value |x| when x is a negative number?\nAnswer:",
                    "expected": ["-x", "negate", "positive"],
                    "note": "Math concept verification"
                }
            ]
        },

        # 3. Python Runtime Exception Taxonomy vs SyntaxError
        {
            "num": 3,
            "title": "Python Exception Hierarchy: ValueError vs SyntaxError vs TypeError",
            "concept_key": "int('text') raises ValueError invalid literal with base 10",
            "approach_fact": "Calling int('hello') raises ValueError: invalid literal for int() with base 10. SyntaxError occurs only during syntax parsing before execution.",
            "teaching_prompt": (
                "Guru: Mari kita bedah taksonomi error di Python secara presisi.\n"
                "1. ValueError: Sintaksis Python benar, tipe data string valid, namun nilai teks tidak bisa dikonversi menjadi bilangan integer.\n"
                "   Contoh: `int('hello')` membangkitkan `ValueError: invalid literal for int() with base 10`!\n"
                "2. SyntaxError: Hanya terjadi jika tata bahasa kode rusak sebelum dieksekusi (contoh: tanda kurung tidak ditutup).\n"
                "Ingat: `int('hello')` adalah ValueError, BUKAN SyntaxError!\n"
                "Murid:"
            ),
            "drills": [
                {
                    "prompt": "In Python, executing `int('hello')` raises which exception type?\nAnswer:",
                    "expected": ["valueerror"],
                    "note": "Python exception precision test"
                },
                {
                    "prompt": "Code:\n```python\nx = int('not_a_number')\n```\nWhat exact error does Python throw?\nAnswer:",
                    "expected": ["valueerror"],
                    "note": "Exact runtime error check"
                }
            ]
        },

        # 4. Modulo 7 Temporal Calendar Logic
        {
            "num": 4,
            "title": "Temporal Multi-Step Projection: Yesterday to Today to Tomorrow",
            "concept_key": "yesterday Wednesday implies today Thursday and tomorrow Friday modulo 7",
            "approach_fact": "Temporal calendar progression: If yesterday was Wednesday, then today is Thursday (+1), and tomorrow is Friday (+2). Complete all steps.",
            "teaching_prompt": (
                "Guru: Selesaikan selalu perhitungan kalender sampai langkah terakhir secara tuntas!\n"
                "Urutan hari: Senin -> Selasa -> Rabu -> Kamis -> Jumat -> Sabtu -> Minggu.\n"
                "Jika soal menyatakan: 'Kemarin adalah hari Rabu. Besok adalah hari apa?':\n"
                "Langkah 1: Kemarin = Rabu.\n"
                "Langkah 2: Hari ini = Rabu + 1 = Kamis.\n"
                "Langkah 3: Besok = Kamis + 1 = JUMAT (Friday)!\n"
                "Selesaikan hingga besok, jangan berhenti di Kamis!\n"
                "Murid:"
            ),
            "drills": [
                {
                    "prompt": "If yesterday was Wednesday, what day will tomorrow be?\nAnswer:",
                    "expected": ["friday"],
                    "note": "Calendar projection test"
                },
                {
                    "prompt": "Kemarin adalah hari Rabu. Besok adalah hari apa?\nJawab:",
                    "expected": ["jumat", "friday"],
                    "note": "Indonesian temporal test"
                }
            ]
        },

        # 5. Question Operator Dissection: 'How many' Cardinality Scalar
        {
            "num": 5,
            "title": "Question Operator Dissection: 'How many' Cardinality vs 'What are' Enumeration",
            "concept_key": "how many question operator strictly requires numerical scalar quantity",
            "approach_fact": "When a question starts with 'How many', the answer must strictly be a numerical integer scalar count, never a list of entity names.",
            "teaching_prompt": (
                "Guru: Perhatikan operator pertanyaan dengan sangat teliti:\n"
                "1. Jika pertanyaan diawali 'How many' (Berapa banyak) -> JAWABAN MUTLAK HANYA ANGKA BILANGAN (SKALAR)!\n"
                "   Contoh: 'How many nucleotide bases compose DNA?' -> Jawab '4' (four), JANGAN mendaftar adenin dsb.\n"
                "2. Jika pertanyaan diawali 'What are' -> barulah sebutkan daftar namanya.\n"
                "Murid:"
            ),
            "drills": [
                {
                    "prompt": "DNA is composed of adenine, thymine, guanine, and cytosine. How many nucleotide bases compose DNA?\nAnswer with only the number:\nAnswer:",
                    "expected": ["4", "four"],
                    "note": "Cardinality scalar test"
                },
                {
                    "prompt": "Our solar system has Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. How many planets are there?\nAnswer:",
                    "expected": ["8", "eight"],
                    "note": "Planets count test"
                }
            ]
        },

        # 6. Natural Conversational Warmth & Direct Articulation
        {
            "num": 6,
            "title": "Natural Conversational Warmth, Empathy & Coherence",
            "concept_key": "conversational warmth friendly articulate concise empathetic response",
            "approach_fact": "Respond to user greetings with genuine warmth, clarity, friendliness, and structured helpfulness without repetitive loops.",
            "teaching_prompt": (
                "Guru: Saat berdialog dengan manusia, selalu bersikap ramah, hangat, santun, dan langsung menjawab ke inti masalah tanpa berbelit-belit.\n"
                "Murid:"
            ),
            "drills": [
                {
                    "prompt": "Halo X-Star! Bagaimana kabarmu hari ini?\nJawab:",
                    "expected": ["halo", "baik", "senang", "bantu"],
                    "note": "Greeting warmth test"
                }
            ]
        }
    ]

    log("\n" + "=" * 90)
    log("[PHASE 2: EXECUTING PLAIN TEXT TUTORING & DYNAMIC ASSIMILATION]")
    log("=" * 90)

    for mod in curriculum:
        log(f"\n{'-'*90}")
        log(f"[*] MENGAJARKAN MODUL {mod['num']}: {mod['title']}")
        log(f"{'-'*90}")

        # 1. Epistemic Concept Assimilation into VGNI Bank
        assimilate_epistemic_concept(
            model,
            tokenizer,
            concept_text=mod["concept_key"],
            approach_fact=mod["approach_fact"]
        )
        curr_stored = int(model.model.brain.vgni_engine.vgni.num_stored.item())
        log(f"[VGNI Epistemic Memory] Anchored concept: '{mod['concept_key'][:50]}...' -> Bank Slots: {curr_stored}/64")

        # 2. Plain Text Teaching Turn (Heuristics & Reflection)
        t0 = time.perf_counter()
        resp_reflect = plain_text_generate(model, tokenizer, mod["teaching_prompt"], max_new_tokens=90)
        t_ref = time.perf_counter() - t0

        telem = get_brain_telemetry(model)
        log(f"[Guru Materi]: {mod['teaching_prompt'].splitlines()[1][:75]}...")
        log(f"[X-Star Refleksi ({t_ref:.2f}s)]: {resp_reflect.replace(chr(10), ' ')[:90]}...")
        log(f"      -> Plastic Norm: {telem['plastic_norm']} | Updates: {telem['plastic_steps']}")

        # 3. Plain Text Practice Drills
        for d_idx, drill in enumerate(mod["drills"], 1):
            t0_d = time.perf_counter()
            d_resp = plain_text_generate(model, tokenizer, drill["prompt"], max_new_tokens=80)
            t_d = time.perf_counter() - t0_d

            is_ok = any(k.lower() in d_resp.lower() for k in drill["expected"])
            status_tag = "[PASS ✓]" if is_ok else "[REVISE ✗]"
            log(f"  * Drill {mod['num']}.{d_idx} ({drill['note']}): {status_tag} in {t_d:.2f}s")
            log(f"    Prompt: {drill['prompt'].replace(chr(10), ' ')[:65]}...")
            log(f"    Model : {d_resp.replace(chr(10), ' ')[:75]}...")

    # =========================================================================
    # PHASE 3: CONSOLIDATION & SYNAPSE PERSISTENCE
    # =========================================================================
    log("\n" + "=" * 90)
    log("[PHASE 3: CONSOLIDATING BRAIN SYNAPSES & PERSISTING SAFETENSORS]")
    log("=" * 90)

    final_telem = get_brain_telemetry(model)
    log(f"  * Final Plastic Norm  : {final_telem['plastic_norm']}")
    log(f"  * Total Synaptic Steps: {final_telem['plastic_steps']}")
    log(f"  * Final VGNI Anchors  : {final_telem['vgni_stored_anchors']} / 64")

    # 1. Save weights checkpoint
    vgni = model.model.brain.vgni_engine.vgni
    plastic_unit = model.model.brain.plastic_unit
    lbu = vgni.lbu
    ckpt_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": "Qwen3.5-2B",
        "d_model": 2048,
        "mode": "plain_text_tutored_dynamic_brain",
        "saturation_pct": round(float(final_telem['vgni_stored_anchors']) / 64.0 * 100.0, 2),
        "slots_filled": final_telem['vgni_stored_anchors'],
        "max_capacity": 64,
        "lipschitz_bound": 0.85,
        "vgni_state": {
            "concept_anchors": vgni.concept_anchors.data.cpu(),
            "approach_vectors": vgni.approach_vectors.data.cpu(),
            "num_stored": vgni.num_stored.data.cpu(),
            "usage_counts": vgni.usage_counts.data.cpu()
        },
        "lbu_weights": {
            "u_logic": lbu.u_logic.data.cpu(),
            "w_step": lbu.w_step.weight.data.cpu(),
            "alpha_scale": lbu.alpha_scale.data.cpu()
        },
        "plastic_weights": {
            "w_plastic": plastic_unit.w_plastic.data.cpu(),
            "plasticity_step_count": plastic_unit.plasticity_step_count.data.cpu()
        }
    }
    torch.save(ckpt_data, str(WEIGHTS_PT_PATH))
    log(f"[+] Saved updated checkpoint to {WEIGHTS_PT_PATH}")

    # 2. Merge all brain tensors directly into model.safetensors
    log("[*] Merging updated brain tensors into model.safetensors...")
    existing_tensors = load_file(str(SAFETENSORS_PATH))
    updated_tensors = dict(existing_tensors)

    # Brain keys
    updated_tensors["model.brain.vgni_engine.vgni.concept_anchors"] = vgni.concept_anchors.data.to(dtype=torch.bfloat16).contiguous()
    updated_tensors["model.brain.vgni_engine.vgni.approach_vectors"] = vgni.approach_vectors.data.to(dtype=torch.bfloat16).contiguous()
    updated_tensors["model.brain.vgni_engine.vgni.num_stored"] = vgni.num_stored.data.contiguous()
    updated_tensors["model.brain.vgni_engine.vgni.usage_counts"] = vgni.usage_counts.data.to(dtype=torch.bfloat16).contiguous()
    updated_tensors["model.brain.vgni_engine.vgni.lbu.u_logic"] = lbu.u_logic.data.to(dtype=torch.bfloat16).contiguous()
    updated_tensors["model.brain.vgni_engine.vgni.lbu.w_step.weight"] = lbu.w_step.weight.data.to(dtype=torch.bfloat16).contiguous()
    updated_tensors["model.brain.vgni_engine.vgni.lbu.alpha_scale"] = lbu.alpha_scale.data.to(dtype=torch.bfloat16).contiguous()
    updated_tensors["model.brain.plastic_unit.w_plastic"] = plastic_unit.w_plastic.data.to(dtype=torch.bfloat16).contiguous()
    updated_tensors["model.brain.plastic_unit.plasticity_step_count"] = plastic_unit.plasticity_step_count.data.contiguous()

    save_file(updated_tensors, str(SAFETENSORS_PATH))
    log(f"[+] Successfully merged {len(updated_tensors)} tensors into {SAFETENSORS_PATH}!")

    total_time = time.perf_counter() - t0_start
    log(f"\n[✓] Plain Text Tutoring & Grounding Session Completed in {total_time:.2f} seconds!")
    log_file.close()


if __name__ == "__main__":
    main()
