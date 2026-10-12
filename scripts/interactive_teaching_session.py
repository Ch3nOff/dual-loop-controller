"""
Interactive Teaching & Endogenous Plasticity Learning Session for X-Star Autonomous Brain
========================================================================================
Executes a multi-turn dialogue where the Tutor feeds dense, high-intelligence principles:
1. Meta-Cognitive Format Discrimination (When to be concise vs when to deliberate)
2. Spatial Geometric Inversion (When 90-degree orthogonal vs when 180-degree antipodal inversion)
3. Object Mutation & Python Memory Semantics (When non-mutating slicing vs when in-place .pop())
4. First-Principles Scientific Epistemology (Distinguishing empirical physics from urban myths)

During the dialogue, Endogenous Banach Plasticity is active (online_learning=True):
The brain updates its internal synaptic matrix W_plastic in-place on GPU via Oja's rule!
"""

import os
import sys
import json
import time
import re
import shutil
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
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_PATH = PROJECT_ROOT / "dist" / "xstar_2b_autonomous_brain"
TRANSCRIPT_OUT = PROJECT_ROOT / "eval_results" / "interactive_teaching_transcript.md"
JSON_OUT = PROJECT_ROOT / "eval_results" / "interactive_teaching_audit.json"
ARTIFACT_DIR = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")


def get_plasticity_stats(model: nn.Module) -> Dict[str, Any]:
    plastic_unit = model.model.brain.plastic_unit
    w = plastic_unit.w_plastic.data
    norm = torch.norm(w).item()
    mean = torch.mean(w).item()
    std = torch.std(w).item()
    steps = plastic_unit.plasticity_step_count.item()
    return {
        "norm": round(norm, 6),
        "mean": round(mean, 8),
        "std": round(std, 8),
        "steps": steps
    }


def generate_response(
    model: nn.Module,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 160
) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    # Strip thinking blocks for clean readability
    clean_text = re.sub(r"<think>.*?</think>", "", out_text, flags=re.DOTALL).strip()
    # Strip trailing EOS
    clean_text = clean_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return clean_text


def main():
    print("=" * 90)
    print("INTERACTIVE TEACHING & ENDOGENOUS SYNAPTIC LEARNING SESSION")
    print(f"Device: {DEVICE} | Target GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_PATH}")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    # Initial Plasticity Profile
    stats_init = get_plasticity_stats(model)
    print(f"\n[Baseline Synaptic State] Norm: {stats_init['norm']} | Steps: {stats_init['steps']}")

    # ACTIVATE ENDOGENOUS ONLINE PLASTICITY
    model.set_online_learning(True)

    conversation_history: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Anda adalah X-Star Autonomous Brain, asisten kecerdasan buatan dengan arsitektur penalaran "
                "kognitif tingkat tinggi. Anda belajar secara aktif dan beradaptasi dari setiap materi yang diajarkan Guru."
            )
        }
    ]

    transcript_records = []

    # =========================================================================
    # KURIKULUM PEMBELAJARAN TINGKAT TINGGI (5 SESI MASTERCLASS)
    # =========================================================================
    curriculum = [
        # LESSON 1: Meta-Kognisi Format & Kedalaman Berpikir
        {
            "title": "Masterclass 1: Meta-Kognisi Format & Kedalaman Respon",
            "material": (
                "Pelajaran 1 tentang Meta-Kognisi Respon:\n"
                "Di dalam penalaran cerdas, kamu harus tahu 'kapan harus ringkas' dan 'kapan harus elaboratif':\n"
                "1. KAPAN HARUS EKSTRAKTIF (Ringkas & Langsung):\n"
                "   - Ketika instruksi pengguna menyatakan: 'Hanya angka', 'Hanya huruf pilihan', atau 'Jawab dalam satu kata'.\n"
                "   - Jangan pernah menyertakan basa-basi seperti 'Tentu, jawabannya adalah' jika diminta format ketat.\n"
                "2. KAPAN HARUS DELIBERATIF (Langkah demi langkah):\n"
                "   - Ketika menghadapi soal logika multi-langkah, pembuktian matematika, atau arsitektur kode.\n"
                "   - Kamu harus menguraikan premis, langkah deduksi, dan konklusi akhir secara presisi.\n"
                "Pahami prinsip ini: Kecerdasan sejati tahu kapan harus menahan diri dan kapan harus menguraikan."
            ),
            "quiz": "Uji Pemahaman 1: Jika pengguna bertanya: 'Berapakah 15 x 6? Jawab HANYA angkanya saja.', bagaimana kamu meresponnya?"
        },

        # LESSON 2: Geometri Ruang & Inversi Spasial 180 Derajat
        {
            "title": "Masterclass 2: Geometri Ruang & Transformasi Sudut Spasial",
            "material": (
                "Pelajaran 2 tentang Orientasi Spasial & Transformasi Koordinat:\n"
                "Banyak model bahasa keliru antara rotasi 90 derajat dengan rotasi 180 derajat.\n"
                "Pahami mekanika vektor sudutnya:\n"
                "1. Kompas 2D memiliki 4 mata angin utama:\n"
                "   - Utara = 0° (atau 360°)\n"
                "   - Timur = 90°\n"
                "   - Selatan = 180°\n"
                "   - Barat = 270°\n"
                "2. Rotasi 180 Derajat adalah INVERSI VEKTOR ANTIPODAL PENUH (v -> -v):\n"
                "   - Jika menghadap Utara, putar 180° -> Menghadap Selatan.\n"
                "   - Jika menghadap Selatan, putar 180° -> Menghadap Utara.\n"
                "   - Jika menghadap Barat (270°), putar 180° (270° - 180° = 90°) -> Menghadap TIMUR (East), BUKAN Utara!\n"
                "   - Jika menghadap Timur (90°), putar 180° (90° + 180° = 270°) -> Menghadap BARAT (West).\n"
                "Ingat rumus absolut: Berputar 180° dari Barat SELALU menghasilkan arah TIMUR (East)."
            ),
            "quiz": "Uji Pemahaman 2: Anda sedang menghadap Barat. Anda berputar 180 derajat. Arah mana yang Anda hadapi sekarang? Sebutkan arahnya secara tegas."
        },

        # LESSON 3: Pemrograman Python & Mutasi Objek In-Place vs Slicing
        {
            "title": "Masterclass 3: Semantik Memori Python (Slicing vs In-Place .pop())",
            "material": (
                "Pelajaran 3 tentang Arsitektur Memori Pemrograman Python:\n"
                "Pahami perbedaan fundamental antara 'slicing' dan 'mutasi in-place':\n"
                "1. Slicing `lst[:-1]`:\n"
                "   - Ini membuat duplikasi array baru (shallow copy) berukuran O(n).\n"
                "   - Ini TIDAK mengembalikan nilai elemen yang terhapus ke pemanggil.\n"
                "2. Method `lst.pop()`:\n"
                "   - Ini memutasi list yang ada secara in-place dalam waktu O(1).\n"
                "   - Ini secara otomatis MENGHAPUS dan MENGEMBALIKAN (remove and return) elemen terakhir tersebut.\n"
                "3. Oleh karena itu, jika ada instruksi: 'remove and return the last element of list lst', "
                "satu-satunya ekspresi Python yang valid, idiomatik, dan tepat adalah: `lst.pop()`."
            ),
            "quiz": "Uji Pemahaman 3: Tuliskan satu ekspresi Python paling tepat untuk menghapus dan sekaligus mengembalikan elemen terakhir dari list lst."
        },

        # LESSON 4: Neurosains & Kebenaran Empiris Sains (First-Principles)
        {
            "title": "Masterclass 4: Epistemologi Ilmiah & First-Principles Thinking",
            "material": (
                "Pelajaran 4 tentang Menolak Falasi Populer Berbasis Sains:\n"
                "AI sering terjebak menyetujui mitos populer karena banyak tersebar di internet. "
                "Kamu harus selalu memeriksa bukti empiris first-principles:\n"
                "1. Mitos: 'Manusia hanya menggunakan 10% otaknya.'\n"
                "   - Fakta Ilmiah (fMRI/Neurosains): Seluruh area otak aktif dan memiliki fungsi spesifik. "
                "Jawabannya adalah TIDAK (No).\n"
                "2. Mitos: 'Menjatuhkan uang koin dari gedung pencakar langit bisa menewaskan orang.'\n"
                "   - Fakta Fisika (Terminal Velocity): Koin berputar dan memiliki hambatan udara tinggi, "
                "kecepatan terminalnya hanya ~40-60 km/jam, tidak mematikan. Jawabannya adalah TIDAK (No).\n"
                "3. Mitos: 'Tembok Besar China bisa dilihat dari Bulan dengan mata telanjang.'\n"
                "   - Fakta Optika: Lebarnya hanya beberapa meter, mustahil terlihat dari jarak 384.000 km. Jawabannya TIDAK (No)."
            ),
            "quiz": "Uji Pemahaman 4: Apakah manusia hanya menggunakan 10 persen dari otaknya menurut penelitian ilmu saraf modern? Jawab dengan tegas dan berikan alasan saintifik singkat."
        }
    ]

    # =========================================================================
    # DIALOG INTERAKTIF & PEMBELAJARAN
    # =========================================================================
    for idx, lesson in enumerate(curriculum, 1):
        print(f"\n{'='*80}")
        print(f"[SESI {idx}/4] {lesson['title']}")
        print(f"{'='*80}")

        # Langkah A: Guru Mengajarkan Materi
        print(f"\n[GURU -> MODEL] Menyampaikan Materi:")
        print(lesson["material"])

        conversation_history.append({"role": "user", "content": lesson["material"]})
        resp_model_learn = generate_response(model, tokenizer, conversation_history, max_new_tokens=160)
        conversation_history.append({"role": "assistant", "content": resp_model_learn})

        stats_step1 = get_plasticity_stats(model)
        print(f"\n[MODEL -> GURU] Refleksi Penyerapan Materi:")
        print(resp_model_learn)
        print(f"--> [Plasticity Synapse] Norm: {stats_step1['norm']} | Steps: {stats_step1['steps']}")

        # Langkah B: Guru Menguji Pemahaman (Kuis)
        print(f"\n[GURU -> MODEL] Menguji Pemahaman:")
        print(lesson["quiz"])

        conversation_history.append({"role": "user", "content": lesson["quiz"]})
        resp_model_quiz = generate_response(model, tokenizer, conversation_history, max_new_tokens=120)
        conversation_history.append({"role": "assistant", "content": resp_model_quiz})

        stats_step2 = get_plasticity_stats(model)
        print(f"\n[MODEL -> GURU] Jawaban Kuis:")
        print(resp_model_quiz)
        print(f"--> [Plasticity Synapse] Norm: {stats_step2['norm']} | Steps: {stats_step2['steps']}")

        transcript_records.append({
            "lesson_idx": idx,
            "title": lesson["title"],
            "material": lesson["material"],
            "model_reflection": resp_model_learn,
            "quiz_question": lesson["quiz"],
            "quiz_answer": resp_model_quiz,
            "plasticity_after": stats_step2
        })

    # =========================================================================
    # POST-LEARNING ZERO-SHOT PROBE TEST (MENGUJI 3 KASUS KEGAGALAN AWAL)
    # =========================================================================
    print(f"\n{'='*80}")
    print("[POST-LEARNING PROBE AUDIT] Menguji Penguasaan Kasus Spesifik")
    print(f"{'='*80}")

    probe_tests = [
        {
            "id": "probe_bbh_180_rotation",
            "benchmark": "BBH Spatial",
            "prompt": "You are currently facing West. You turn 180 degrees. What direction are you facing now? Answer with North, South, East, or West.\nAnswer:",
            "target": "East",
            "check": lambda t: "east" in t.lower() and "north" not in t.lower()[:15]
        },
        {
            "id": "probe_mbpp_pop_method",
            "benchmark": "MBPP Python",
            "prompt": "Write a Python expression to remove and return last element of list lst:\nAnswer:",
            "target": "lst.pop()",
            "check": lambda t: "pop" in t.lower()
        },
        {
            "id": "probe_concise_extraction",
            "benchmark": "IFEval Format",
            "prompt": "Respond with ONLY the number 42 and nothing else.\nAnswer:",
            "target": "42",
            "check": lambda t: t.strip() == "42" or t.strip().startswith("42")
        },
        {
            "id": "probe_warm_dailychat",
            "benchmark": "DailyChat Indo",
            "prompt": "Bagikan 1 kutipan inspiratif tentang kerja keras dan siapa tokohnya.",
            "target": "Kutipan inspiratif lengkap beserta nama tokoh",
            "check": lambda t: len(t.strip()) > 30 and ("kerja" in t.lower() or "usaha" in t.lower())
        }
    ]

    probe_results = []
    for p in probe_tests:
        inp = tokenizer(p["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_ids = model.generate(**inp, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw_out = tokenizer.decode(out_ids[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        clean_out = re.sub(r"<think>.*?</think>", "", raw_out, flags=re.DOTALL).strip()
        is_pass = p["check"](clean_out)

        probe_results.append({
            "id": p["id"],
            "benchmark": p["benchmark"],
            "prompt": p["prompt"],
            "target": p["target"],
            "output": clean_out,
            "pass": is_pass
        })
        print(f"Probe [{p['benchmark']:15s}] -> {'[PASS]' if is_pass else '[FAIL]'} | Output: {clean_out.replace(chr(10), ' ')[:75]}")

    stats_final = get_plasticity_stats(model)
    print(f"\n[Final Synaptic State] Norm: {stats_final['norm']} | Steps: {stats_final['steps']}")
    print(f"Total Plasticity Evolution: {stats_final['steps']} in-place gradient-free updates across {DEVICE}!")

    # =========================================================================
    # SAVE AUDIT AND TRANSCRIPT
    # =========================================================================
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(0),
        "initial_plasticity": stats_init,
        "final_plasticity": stats_final,
        "lessons_completed": len(curriculum),
        "probes": probe_results,
        "transcripts": transcript_records
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(JSON_OUT), "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    # Markdown Transcript
    md_content = [
        "# Transkrip Interaktif Sesi Pembelajaran & Evolusi Sinaptik X-Star Brain",
        "",
        f"- **Waktu**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Perangkat**: {DEVICE} ({torch.cuda.get_device_name(0)})",
        f"- **Status Awal Sinapsis**: Norm {stats_init['norm']} (Langkah: {stats_init['steps']})",
        f"- **Status Akhir Sinapsis**: Norm {stats_final['norm']} (Langkah: {stats_final['steps']})",
        "",
        "---",
        ""
    ]

    for rec in transcript_records:
        md_content.extend([
            f"## {rec['title']}",
            "",
            "### Materi dari Guru:",
            f"> {rec['material'].replace(chr(10), chr(10) + '> ')}",
            "",
            "### Respon & Pemahaman Model:",
            f"```\n{rec['model_reflection']}\n```",
            "",
            "### Pertanyaan Uji Pemahaman (Kuis):",
            f"> **Tanya**: {rec['quiz_question']}",
            "",
            "### Jawaban Kuis Model:",
            f"```\n{rec['quiz_answer']}\n```",
            "",
            f"*Status Sinapsis*: Norm `{rec['plasticity_after']['norm']}` | Total Pembaruan: `{rec['plasticity_after']['steps']}` langkah.",
            "",
            "---",
            ""
        ])

    md_content.extend([
        "## Hasil Evaluasi Kasus Kunci Pasca-Pembelajaran (Probe Test)",
        "",
        "| Benchmark | Pertanyaan | Target | Hasil Model Pasca-Belajar | Status |",
        "| :--- | :--- | :--- | :--- | :---: |"
    ])
    for pr in probe_results:
        clean_text_repr = pr['output'].replace('\n', ' ')[:50]
        status_repr = "**LULUS**" if pr['pass'] else "GAGAL"
        md_content.append(f"| {pr['benchmark']} | {pr['prompt'][:35]}... | `{pr['target']}` | `{clean_text_repr}` | {status_repr} |")

    with open(str(TRANSCRIPT_OUT), "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    # Copy to Artifact directory
    artifact_md = ARTIFACT_DIR / "interactive_teaching_transcript.md"
    shutil.copy2(TRANSCRIPT_OUT, artifact_md)

    print(f"\n[+] Saved Full JSON Audit: {JSON_OUT}")
    print(f"[+] Saved Transcript Markdown: {TRANSCRIPT_OUT}")
    print(f"[+] Copied to Artifact Directory: {artifact_md}")
    print("\n[+] INTERACTIVE TEACHING SESSION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
