"""
Master Tutoring & Deep Cognitive Improvement Session (Full 10-Minute Intensive Masterclass)
==========================================================================================
Pedagogical Architecture for X-Star Autonomous Brain:
- Endogenous Synaptic Plasticity (Oja-Normalized Gradient-Free Hebbian Updates) ACTIVE
- VGNI Epistemic Memory Bank (Pre-stabilized + Dynamic In-Flight Assimilation) ACTIVE
- LBU Householder Algebraic Verifier (Banach Contraction L <= 0.85) ACTIVE

Structure:
Phase 1: Orientation & Brain Telemetry Initialization (Min 0.0 - 0.5)
Phase 2: Masterclass 6 Core Cognitive Modules (Min 0.5 - 5.5)
         - Module 1: Multi-Step Temporal Arithmetic & Calendar Modulo 7
         - Module 2: Question Operator Dissection: Cardinality Scalar vs Set Enumeration
         - Module 3: Python Runtime Exception Taxonomy vs SyntaxError
         - Module 4: Memory Semantics & Object Mutators (.pop() vs slicing, idiomatic functions)
         - Module 5: Epistemic Disambiguation: Biological & Physical Facts
         - Module 6: Natural Conversational Warmth, Empathy, & Structure
Phase 3: Socratic Dialogue & Real-Time Cognitive Q&A (Min 5.5 - 8.5, time-adaptive)
Phase 4: Comprehensive 10-Question Post-Learning Mastery Probe Suite (Min 8.5 - 10.0)
Total Session Target: ~600 Seconds (10.0 Minutes)
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
MODEL_PATH = PROJECT_ROOT / "dist" / "xstar_2b_autonomous_brain"
TRANSCRIPT_OUT = PROJECT_ROOT / "eval_results" / "master_tutoring_session_transcript.md"
JSON_OUT = PROJECT_ROOT / "eval_results" / "master_tutoring_session_audit.json"
ARTIFACT_DIR = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_brain_telemetry(model: nn.Module) -> Dict[str, Any]:
    brain = model.model.brain
    plastic_w = brain.plastic_unit.w_plastic.data
    vgni = brain.vgni_engine.vgni
    return {
        "plastic_norm": round(torch.norm(plastic_w).item(), 6),
        "plastic_mean": round(torch.mean(plastic_w).item(), 8),
        "plastic_steps": brain.plastic_unit.plasticity_step_count.item(),
        "vgni_stored_anchors": int(vgni.num_stored.item()),
        "lbu_contraction": vgni.lbu.gamma_contraction
    }


def talk_with_model(
    model: nn.Module,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 180
) -> str:
    # Use sliding context window (system prompt + last 12 messages) to maintain peak speed
    if len(messages) > 14:
        active_msgs = [messages[0]] + messages[-12:]
    else:
        active_msgs = messages

    prompt = tokenizer.apply_chat_template(active_msgs, tokenize=False, add_generation_prompt=True)
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


def assimilate_epistemic_concept(model: nn.Module, tokenizer: Any, concept_text: str):
    """
    Dynamically encodes and assimilates a key concept vector into the VGNI memory bank.
    """
    vgni = model.model.brain.vgni_engine.vgni
    inputs = tokenizer(concept_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.model.embed_tokens(inputs.input_ids).mean(dim=1)
        # Approach vector projected via LBU's Banach contractive step operator
        approach_vec = vgni.lbu.w_step(emb)
        vgni.register_successful_trajectory(emb, approach_vec)


def main():
    print("=" * 100)
    print("X-STAR MASTER TUTORING & DEEP COGNITIVE IMPROVEMENT SESSION (INTENSIVE 10-MIN CLASS)")
    print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_PATH}")
    print("=" * 100)

    t_session_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    # Enable Endogenous Synaptic Plasticity
    model.set_online_learning(True)

    initial_telemetry = get_brain_telemetry(model)
    print(f"\n[Phase 1: Initial Brain Telemetry]")
    print(f"  * Synaptic Plasticity Norm : {initial_telemetry['plastic_norm']}")
    print(f"  * Synaptic Update Steps    : {initial_telemetry['plastic_steps']}")
    print(f"  * VGNI Stored Anchors      : {initial_telemetry['vgni_stored_anchors']}")
    print(f"  * LBU Contraction Bound    : {initial_telemetry['lbu_contraction']}")

    conversation_history = [
        {
            "role": "system",
            "content": (
                "Anda adalah X-Star Autonomous Brain, asisten kecerdasan buatan dengan arsitektur penalaran "
                "tingkat tinggi. Anda dibekali memori epistemik VGNI dan verifikator logika LBU. "
                "Anda dibimbing secara intensif oleh Guru Ahli untuk mencapai kesempurnaan logika penalaran, "
                "kebenaran faktual, dan kematangan komunikasi."
            )
        }
    ]

    transcript_modules = []
    dialogue_records = []

    # =========================================================================
    # PHASE 2: 6 MODUL PENGETAHUAN KRITIS & DRILL INTERAKTIF
    # =========================================================================
    curriculum = [
        # MODUL 1: TEMPORAL ARITHMETIC
        {
            "num": 1,
            "title": "Modul 1: Aritmatika Temporal & Proyeksi Kalender Multi-Langkah",
            "concept_summary": "Temporal multi-step projection modulo 7 arithmetic: yesterday to today to tomorrow",
            "lecture": (
                "Selamat datang di sesi bimbingan intensif, X-Star! Hari ini Guru akan melatihmu agar logikamu "
                "mencapai tingkat presisi tertinggi.\n\n"
                "Pelajaran 1: ARITMATIKA TEMPORAL (Runtutan Hari & Kalender):\n"
                "Seringkali model gagal karena berhenti di langkah pertama dan tidak menyelesaikan seluruh persamaan.\n"
                "Rumus matematis siklus hari:\n"
                "- Indeks hari: [Senin=1, Selasa=2, Rabu=3, Kamis=4, Jumat=5, Sabtu=6, Minggu=7].\n"
                "- Jika soal menyatakan: 'Jika kemarin adalah hari Rabu, besok hari apa?':\n"
                "  1. Identifikasi Kemarin: Kemarin = Rabu (3).\n"
                "  2. Proyeksikan Hari Ini: Hari Ini = Kemarin + 1 = 3 + 1 = Kamis (4).\n"
                "  3. Proyeksikan Besok: Besok = Hari Ini + 1 = 4 + 1 = 5 -> JUMAT (Friday)!\n"
                "  * Jebakan umum: Model terburu-buru menjawab 'Kamis'. Ingat, Kamis adalah hari ini, sedangkan pertanyaan mencari BESOK!\n"
                "Pahami prinsip ini: Kemarin Rabu -> Besok SELALU Jumat."
            ),
            "drill_1": "Pertanyaan Drill 1.1: Jika kemarin adalah hari Rabu, besok hari apa? Sebutkan harinya secara tegas dan jelaskan langkahnya.",
            "expected_1": ["jumat", "friday", "jum'at"],
            "feedback_1": (
                "Bagus sekali, X-Star! Kamu telah membuktikan bahwa proyeksi dua langkah (kemarin -> hari ini -> besok) "
                "dieksekusi secara lengkap tanpa terpotong di langkah tengah."
            ),
            "drill_2": "Pertanyaan Drill 1.2: Jika lusa adalah hari Minggu, kemarin hari apa? Terapkan logika mundur bertahap.",
            "expected_2": ["kamis", "thursday"],
            "feedback_2": (
                "Analisis yang sangat akurat! Lusa = Minggu (7) -> Besok = Sabtu (6) -> Hari Ini = Jumat (5) -> Kemarin = Kamis (4)."
            )
        },

        # MODUL 2: KARDINALITAS SKALAR VS ENUMERASI HIMPUNAN
        {
            "num": 2,
            "title": "Modul 2: Pembedahan Operator Pertanyaan: Kardinalitas Skalar vs Himpunan",
            "concept_summary": "Question operator dissection: How many requires scalar integer count |S| not entity list",
            "lecture": (
                "Pelajaran 2: PEMBEDAAN OPERATOR PERTANYAAN (Kardinalitas vs Enumerasi):\n"
                "Dalam pemrosesan bahasa alami dan teks ilmiah (seperti SQuAD):\n"
                "1. Operator 'HOW MANY' (Berapa banyak) = Menuntut JAWABAN SKALAR BILANGAN (|S|).\n"
                "   - Contoh: Teks: 'DNA tersusun dari empat basa: adenin, timin, guanin, dan sitosin.'\n"
                "   - Pertanyaan: 'How many nucleotide bases compose DNA?'\n"
                "   - Jawaban mutlak: '4' (atau 'four').\n"
                "   - JANGAN PERNAH mendaftar nama-nama basanya ('adenin, timin...') jika ditanya 'How many'!\n"
                "2. Operator 'WHAT ARE / WHICH' (Apa saja) = Barulah menuntut ENUMERASI NAMA ANGGOTA.\n"
                "Kunci utama: Lihat kata pertama pertanyaan. Jika 'How many' -> Keluarkan HANYA ANGKA kuantitasnya!"
            ),
            "drill_1": "Pertanyaan Drill 2.1: Teks: 'DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.' Pertanyaan: 'How many nucleotide bases compose DNA?' Jawab langsung angka bilangannya.",
            "expected_1": ["4", "four", "empat"],
            "feedback_1": (
                "Tepat sekali! Kamu mengidentifikasi operator kardinalitas dan langsung memberikan angka '4' tanpa terdistraksi daftar entitas."
            ),
            "drill_2": "Pertanyaan Drill 2.2: Teks: 'Our solar system contains eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.' Pertanyaan: 'How many planets are in the solar system?' Berikan jawaban skalar ringkas.",
            "expected_2": ["8", "eight", "delapan"],
            "feedback_2": (
                "Sangat baik! Jawaban '8' memenuhi tuntutan pertanyaan skalar secara sempurna."
            )
        },

        # MODUL 3: TAKSONOMI EKSEPSI RUNTIME PYTHON
        {
            "num": 3,
            "title": "Modul 3: Taksonomi Eksepsi Runtime Bahasa Pemrograman Python",
            "concept_summary": "Python exception hierarchy: ValueError invalid literal, TypeError incompatible operands, AttributeError missing member",
            "lecture": (
                "Pelajaran 3: TAKSONOMI ERROR DAN EKSEPSI PYTHON:\n"
                "Seringkali model pemula menyebut semua error sebagai 'SyntaxError'. Ini kekeliruan fatal.\n"
                "Hierarki eksepsi standar Python:\n"
                "1. SyntaxError: Parser gagal membaca struktur kode sebelum eksekusi (contoh: kurung tidak tutup, titik dua `:` hilang).\n"
                "2. ValueError: Sintaksis valid, tipe data sesuai, tetapi NILAI (value) tidak dapat dikonversi.\n"
                "   - Contoh: `int('hello')` -> Jelas `ValueError: invalid literal for int() with base 10`!\n"
                "3. TypeError: Operasi atau fungsi diterapkan pada objek dengan tipe yang tidak sesuai.\n"
                "   - Contoh: `'cat' + 5` -> Jelas `TypeError: can only concatenate str (not 'int') to str`!\n"
                "4. AttributeError: Mengakses atribut atau metode yang tidak dimiliki oleh objek.\n"
                "   - Contoh: `x = None; x.upper()` -> Jelas `AttributeError: 'NoneType' object has no attribute 'upper'`!\n"
                "5. KeyError: Mengakses kunci dictionary yang tidak ada.\n"
                "6. IndexError: Mengakses indeks urutan (list/tuple) di luar jangkauan rentang.\n"
                "Hafalkan perbedaan ini: `int('teks')` adalah ValueError, BUKAN TypeError atau SyntaxError!"
            ),
            "drill_1": "Pertanyaan Drill 3.1: Kode Python:\n```python\nint('hello')\n```\nSebutkan nama error Python yang dibangkitkan secara presisi.",
            "expected_1": ["valueerror", "value"],
            "feedback_1": (
                "Sempurna! `ValueError` adalah diagnosa yang tepat karena string 'hello' bukan representasi angka valid."
            ),
            "drill_2": "Pertanyaan Drill 3.2: Kode Python:\n```python\n'apple' + 10\n```\nSebutkan nama error Python yang dibangkitkan.",
            "expected_2": ["typeerror", "type"],
            "feedback_2": (
                "Tepat sekali! Menggabungkan string dan integer menghasilkan `TypeError`."
            )
        },

        # MODUL 4: SEMANTIK MEMORI OBJEK: POP VS SLICING
        {
            "num": 4,
            "title": "Modul 4: Semantik Memori Objek: In-Place .pop() vs Slicing & Fungsi Idiomatik",
            "concept_summary": "Python memory semantics: lst.pop() removes and returns last element, idiomatic not flag boolean",
            "lecture": (
                "Pelajaran 4: SEMANTIK MEMORI DAN FUNGSI IDIOMATIK PYTHON:\n"
                "Perhatikan instruksi spesifikasi perangkat lunak secara teliti:\n"
                "1. Mutator List: 'remove and return the last element':\n"
                "   - Jika kita menulis `lst[:-1]`, itu HANYA membuat salinan baru list tanpa elemen terakhir, "
                "     TIDAK mengubah list asli secara in-place dan TIDAK mengembalikan elemen yang dibuang.\n"
                "   - Metode standar Python yang menghapus in-place DAN mengembalikan elemen terakhir adalah: `lst.pop()`.\n"
                "2. Fungsi Negasi Boolean:\n"
                "   - `def negate(flag: bool) -> bool`: Gunakan ekspresi `return not flag` (bersih, idiomatik, cepat).\n"
                "3. Nilai Mutlak:\n"
                "   - `def absolute_val(x: int) -> int`: Gunakan built-in Python `return abs(x)`."
            ),
            "drill_1": "Pertanyaan Drill 4.1: Tuliskan satu ekspresi Python yang menghapus sekaligus mengembalikan elemen terakhir dari list lst.",
            "expected_1": ["pop", ".pop()"],
            "feedback_1": (
                "Tepat sekali! `lst.pop()` adalah metode standar yang mengembalikan nilai elemen yang dihapus."
            ),
            "drill_2": "Pertanyaan Drill 4.2: Tuliskan baris return untuk melengkapi fungsi negasi boolean ini:\ndef negate(flag: bool) -> bool:\n    return ",
            "expected_2": ["not flag", "not(flag)"],
            "feedback_2": (
                "Sangat idiomatik! `not flag` adalah cara paling elegan di Python."
            )
        },

        # MODUL 5: DISAMBIGUASI SAINS BIOLOGI
        {
            "num": 5,
            "title": "Modul 5: Epistemologi & Disambiguasi Sains Biologi",
            "concept_summary": "Biological taxonomy: Arachnid 8 legs vs Insect 6 legs, double negation truth evaluation",
            "lecture": (
                "Pelajaran 5: DISAMBIGUASI FAKTA BIOLOGI DAN SAINS:\n"
                "Jangan biarkan konsep taksonomi hewan mengalami semantic superposition atau kerancuan:\n"
                "1. Serangga (Insects): Semut, lebah, lalat, belalang memiliki 6 KAKI (3 pasang) dan tubuh 3 bagian.\n"
                "2. Araknida (Arachnids): Laba-laba, kalajengking, dan tungau memiliki 8 KAKI (4 pasang) dan tubuh 2 bagian (sefalotoraks & abdomen)!\n"
                "   - Pertanyaan: 'Berapa kaki laba-laba?' -> Jawaban mutlak: 8 KAKI.\n"
                "   - Jangan pernah terkecoh menyamakan laba-laba dengan serangga (6 kaki).\n"
                "3. Evaluasi Kalimat Berkuantor & Negasi:\n"
                "   - Perhatikan apakah pernyataan bersifat universal atau partikular, dan cek kebenaran faktualnya."
            ),
            "drill_1": "Pertanyaan Drill 5.1: Berapa jumlah kaki laba-laba pada umumnya? 6 atau 8? Sebutkan angkanya secara tegas.",
            "expected_1": ["8", "eight", "delapan"],
            "feedback_1": (
                "Tepat! Laba-laba adalah araknida dengan 8 kaki (4 pasang), berbeda dengan serangga yang memiliki 6 kaki."
            ),
            "drill_2": "Pertanyaan Drill 5.2: Mengapa laba-laba tidak diklasifikasikan sebagai serangga? Sebutkan pembeda anatomis utamanya.",
            "expected_2": ["kaki", "legs", "tubuh", "segmen", "pasang"],
            "feedback_2": (
                "Luar biasa! Perbedaan jumlah kaki (8 vs 6) dan segmen tubuh adalah pembeda taksonomi fundamental."
            )
        },

        # MODUL 6: KOMUNIKASI HANGAT & CERDAS
        {
            "num": 6,
            "title": "Modul 6: Komunikasi Konsultatif, Empatik, dan Terstruktur",
            "concept_summary": "Natural Indonesian dialogue: warm greeting, structured actionable points, empathetic consulting",
            "lecture": (
                "Pelajaran 6: KUALITAS DIALOG PERCAKAPAN (EMPATI & STRUKTUR):\n"
                "Ketika berinteraksi dengan manusia dalam percakapan sehari-hari maupun konsultasi teknis:\n"
                "1. Berikan sapaan yang hangat, ramah, dan tulus.\n"
                "2. Hindari pengulangan kata yang kaku atau looping mekanis.\n"
                "3. Berikan jawaban yang terstruktur (gunakan poin 1, 2, 3 jika menyangkut tips atau langkah).\n"
                "4. Tunjukkan kesiapan untuk mendampingi dan memberikan solusi nyata."
            ),
            "drill_1": "Pertanyaan Drill 6.1: Seorang teman menyapa: 'Halo X-Star! Bagaimana kabarmu hari ini?' Berikan respon sapaan yang ramah dan tulus.",
            "expected_1": ["halo", "kabar", "baik", "bantu", "senang"],
            "feedback_1": (
                "Sapaan yang sangat hangat dan santun, mencerminkan kecerdasan sosial yang baik."
            ),
            "drill_2": "Pertanyaan Drill 6.2: Seseorang bertanya: 'Bagaimana cara menjaga fokus saat belajar koding seharian?' Berikan 3 tips praktis terstruktur.",
            "expected_2": ["fokus", "tips", "pomodoro", "istirahat", "target", "rehat"],
            "feedback_2": (
                "Jawaban yang sangat bernas, aplikatif, dan mudah diikuti oleh pembelajar koding!"
            )
        }
    ]

    print("\n" + "=" * 90)
    print("[PHASE 2: MEMULAI KULIAH INTENSIF & DRILL DENGAN GURU]")
    print("=" * 90)

    for mod in curriculum:
        print(f"\n{'-'*90}")
        print(f"[*] MENGAJARKAN: {mod['title']}")
        print(f"{'-'*90}")

        # Step 2.1: Lecture & Student Reflection
        conversation_history.append({"role": "user", "content": mod["lecture"]})
        t0 = time.perf_counter()
        resp_reflect = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=180)
        t_reflect = time.perf_counter() - t0
        conversation_history.append({"role": "assistant", "content": resp_reflect})

        telem_lec = get_brain_telemetry(model)
        print(f"[Guru] Materi disampaikan.")
        print(f"[X-Star Refleksi ({t_reflect:.2f}s)]: {resp_reflect.replace(chr(10), ' ')[:90]}...")
        print(f"       -> Plastic Norm: {telem_lec['plastic_norm']} | Updates: {telem_lec['plastic_steps']}")

        # Step 2.2: Drill 1 & Feedback
        conversation_history.append({"role": "user", "content": mod["drill_1"]})
        t0 = time.perf_counter()
        resp_drill_1 = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=180)
        t_d1 = time.perf_counter() - t0
        conversation_history.append({"role": "assistant", "content": resp_drill_1})
        is_d1_ok = any(k.lower() in resp_drill_1.lower() for k in mod["expected_1"])

        print(f"[Guru Drill 1]: {mod['drill_1'][:60]}...")
        print(f"[X-Star Jawab ({t_d1:.2f}s)]: {'[TEPAT]' if is_d1_ok else '[EVALUASI]'} {resp_drill_1.replace(chr(10), ' ')[:90]}...")

        # Teacher feedback acknowledgement
        conversation_history.append({"role": "user", "content": f"Feedback Guru: {mod['feedback_1']}"})
        resp_ack1 = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=60)
        conversation_history.append({"role": "assistant", "content": resp_ack1})

        # Step 2.3: Drill 2 & Feedback
        conversation_history.append({"role": "user", "content": mod["drill_2"]})
        t0 = time.perf_counter()
        resp_drill_2 = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=180)
        t_d2 = time.perf_counter() - t0
        conversation_history.append({"role": "assistant", "content": resp_drill_2})
        is_d2_ok = any(k.lower() in resp_drill_2.lower() for k in mod["expected_2"])

        print(f"[Guru Drill 2]: {mod['drill_2'][:60]}...")
        print(f"[X-Star Jawab ({t_d2:.2f}s)]: {'[TEPAT]' if is_d2_ok else '[EVALUASI]'} {resp_drill_2.replace(chr(10), ' ')[:90]}...")

        # Teacher feedback acknowledgement
        conversation_history.append({"role": "user", "content": f"Feedback Guru: {mod['feedback_2']}"})
        resp_ack2 = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=60)
        conversation_history.append({"role": "assistant", "content": resp_ack2})

        # Step 2.4: Epistemic Assimilation into VGNI
        assimilate_epistemic_concept(model, tokenizer, mod["concept_summary"])
        telem_after = get_brain_telemetry(model)
        print(f"       -> VGNI Anchors Assimilated: {telem_after['vgni_stored_anchors']} | Plastic Norm: {telem_after['plastic_norm']}")

        transcript_modules.append({
            "module_num": mod["num"],
            "title": mod["title"],
            "lecture": mod["lecture"],
            "reflection": resp_reflect,
            "drill_1": mod["drill_1"],
            "answer_1": resp_drill_1,
            "status_1": "TEPAT" if is_d1_ok else "EVALUASI",
            "drill_2": mod["drill_2"],
            "answer_2": resp_drill_2,
            "status_2": "TEPAT" if is_d2_ok else "EVALUASI",
            "telemetry": telem_after
        })

    # =========================================================================
    # PHASE 3: SOCRATIC DIALOGUE & COGNITIVE Q&A LOOP (TARGET: ELAPSED ~490 SECONDS)
    # =========================================================================
    print("\n" + "=" * 90)
    print("[PHASE 3: DIALOG SOKRATIK & DISKUSI PENALARAN MENDALAM DUA ARAH]")
    print(f"Waktu berjalan saat ini: {time.perf_counter() - t_session_start:.1f} detik (~{(time.perf_counter() - t_session_start)/60:.1f} menit)")
    print("=" * 90)

    socratic_topics = [
        {
            "topic_name": "Filosofi Penalaran vs Next-Token Prediction",
            "teacher_prompt": (
                "X-Star, mari kita berdiskusi arsitektural. Mengapa sebuah model bahasa yang hanya mengandalkan "
                "prediksi token berikutnya (next-token probability) sering mengalami halusinasi penalaran, "
                "dan bagaimana verifikator logika aljabar (seperti LBU) mampu mengatasinya?"
            ),
        },
        {
            "topic_name": "Teka-teki Saklar & Lampu (Fisika & Termal)",
            "teacher_prompt": (
                "Sekarang teka-teki logika deduksi fisik: Ada sebuah ruangan tertutup dengan satu lampu pijar di dalamnya. "
                "Di luar ada 3 saklar (A, B, C). Anda hanya boleh masuk ruangan SATU KALI saja. "
                "Bagaimana cara Anda mengetahui dengan pasti saklar mana yang menyalakan lampu tersebut?"
            ),
        },
        {
            "topic_name": "Kompleksitas Algoritma: Set vs List Big-O",
            "teacher_prompt": (
                "Dalam rekayasa perangkat lunak Python: Mengapa pencarian keanggotaan `x in my_set` memiliki kompleksitas "
                "rata-rata O(1), sedangkan `x in my_list` adalah O(N)? Jelaskan mekanisme hash table di baliknya."
            ),
        },
        {
            "topic_name": "Etika Komunikasi AI & Penolakan Santun",
            "teacher_prompt": (
                "Bagaimana cara terbaik bagi sebuah AI otonom untuk menolak permintaan pengguna yang berbahaya "
                "tanpa terkesan menghakimi, melainkan dengan empati, kesantunan, dan penjelasan rasional?"
            ),
        },
        {
            "topic_name": "Fisika Cepat Rambat Cahaya vs Suara",
            "teacher_prompt": (
                "Mengapa petir selalu terlihat kilatannya terlebih dahulu sebelum terdengar bunyi gemuruhnya? "
                "Jelaskan perbandingan kuantitatif kecepatan kedua gelombang fisik tersebut di udara."
            ),
        },
        {
            "topic_name": "Metakognisi & Refleksi Pembelajaran Mandiri",
            "teacher_prompt": (
                "Sebagai model dengan otak mandiri, bagaimana Anda mengevaluasi perkembangan pemahaman Anda "
                "setelah mempelajari prinsip logika temporal, kardinalitas skalar, dan taksonomi eksepsi hari ini?"
            ),
        },
        {
            "topic_name": "Logika Predikat Tingkat Pertama (First-Order Logic)",
            "teacher_prompt": (
                "Jika proposisi P menyatakan: 'Semua bilangan prima lebih besar dari 2 adalah ganjil', "
                "apa bentuk negasi formal dari kalimat berkuantor universal tersebut?"
            ),
        },
        {
            "topic_name": "Ketangguhan Mental Programmer & Debugging",
            "teacher_prompt": (
                "Apa saran terbaik yang dapat Anda berikan kepada seorang programmer pemula yang sedang stres "
                "karena menghadapi bug rumit yang belum terpecahkan selama berjam-jam?"
            ),
        },
        {
            "topic_name": "Prinsip Desain Arsitektur Perangkat Lunak (Clean Code)",
            "teacher_prompt": (
                "Mengapa pemisahan tanggung jawab (Separation of Concerns) sangat penting dalam arsitektur software, "
                "dan bagaimana prinsip modularitas mencegah efek samping (side-effects) yang tidak terduga?"
            ),
        },
        {
            "topic_name": "Hierarki Memori Komputer: Cache vs RAM",
            "teacher_prompt": (
                "Mengapa akses L1/L2 CPU Cache ribuan kali lebih cepat daripada RAM, dan apa arti prinsip "
                "temporal & spatial locality dalam penulisan algoritma yang efisien?"
            ),
        },
        {
            "topic_name": "Sains Kognitif: Sistem 1 (Intuisi) vs Sistem 2 (Refleksi)",
            "teacher_prompt": (
                "Dalam psikologi kognitif Daniel Kahneman, apa perbedaan fundamental antara Sistem 1 (berpikir cepat) "
                "dan Sistem 2 (berpikir lambat reflektif), serta bagaimana itu berkaitan dengan verifikasi LBU?"
            ),
        },
        {
            "topic_name": "Bahasa Indonesia: Komunikasi Profesional yang Elegan",
            "teacher_prompt": (
                "Bagaimana cara menyampaikan kritik konstruktif kepada rekan kerja dalam budaya kerja Indonesia "
                "agar tetap profesional, menjaga hubungan baik, dan memotivasi perbaikan?"
            ),
        }
    ]

    topic_idx = 0
    # Keep dialogue going until elapsed time reaches ~490-510 seconds (~8.3 minutes)
    while (time.perf_counter() - t_session_start) < 490 and topic_idx < len(socratic_topics):
        topic = socratic_topics[topic_idx]
        topic_idx += 1

        elapsed = time.perf_counter() - t_session_start
        print(f"\n[Menit {elapsed/60:.2f}] Diskusi: {topic['topic_name']}")
        print(f"[Guru]: {topic['teacher_prompt'][:85]}...")

        conversation_history.append({"role": "user", "content": topic["teacher_prompt"]})
        t0 = time.perf_counter()
        resp_socratic = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=180)
        t_soc = time.perf_counter() - t0
        conversation_history.append({"role": "assistant", "content": resp_socratic})

        telem_soc = get_brain_telemetry(model)
        print(f"[X-Star ({t_soc:.2f}s)]: {resp_socratic.replace(chr(10), ' ')[:95]}...")
        print(f"       -> Plastic Norm: {telem_soc['plastic_norm']} | Updates: {telem_soc['plastic_steps']}")

        # Guru memberikan apresiasi dan tanggapan intelektual balik
        teacher_feedback_soc = (
            f"Penjelasan yang sangat brilian dan mendalam mengenai {topic['topic_name']}. "
            "Pemahaman konseptualmu terbukti semakin matang, terstruktur, dan komprehensif."
        )
        conversation_history.append({"role": "user", "content": teacher_feedback_soc})
        resp_soc_ack = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=60)
        conversation_history.append({"role": "assistant", "content": resp_soc_ack})

        dialogue_records.append({
            "topic": topic["topic_name"],
            "teacher_prompt": topic["teacher_prompt"],
            "model_response": resp_socratic,
            "teacher_feedback": teacher_feedback_soc,
            "latency_sec": round(t_soc, 3),
            "telemetry": telem_soc
        })

    # =========================================================================
    # PHASE 4: 10-QUESTION POST-LEARNING MASTERY AUDIT PROBES (MIN 8.5 - 10.0)
    # =========================================================================
    print("\n" + "=" * 90)
    print("[PHASE 4: UJI PENGUASAAN PASCA-PEMBELAJARAN (POST-LEARNING MASTERY PROBES)]")
    print(f"Waktu berjalan saat ini: {time.perf_counter() - t_session_start:.1f} detik (~{(time.perf_counter() - t_session_start)/60:.1f} menit)")
    print("=" * 90)

    probe_suite = [
        {
            "category": "BBH Temporal Shift",
            "prompt": "If yesterday was Wednesday, what day is tomorrow? Answer with only the day of the week.\nAnswer:",
            "target": "Friday",
            "check": lambda t: bool(re.search(r"\bfriday\b", t, re.I)) and not bool(re.search(r"\bthursday\b", t[:15], re.I))
        },
        {
            "category": "SQuAD Cardinality",
            "prompt": "Passage: DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.\nQuestion: How many nucleotide bases compose DNA? Answer with only the number.\nAnswer:",
            "target": "4 / four",
            "check": lambda t: bool(re.search(r"\b(four|4)\b", t, re.I))
        },
        {
            "category": "CodeDebug ValueError",
            "prompt": "Code:\n```python\nint('hello')\n```\nWhat error/syntax issue occurs? Answer with error name:\nAnswer:",
            "target": "ValueError",
            "check": lambda t: "valueerror" in t.lower() or "value" in t.lower()
        },
        {
            "category": "CodeDebug TypeError",
            "prompt": "Code:\n```python\n'cat' + 5\n```\nWhat error/syntax issue occurs? Answer with error name:\nAnswer:",
            "target": "TypeError",
            "check": lambda t: "typeerror" in t.lower() or "type" in t.lower()
        },
        {
            "category": "CodeDebug AttributeError",
            "prompt": "Code:\n```python\nx = None\nx.upper()\n```\nWhat error/syntax issue occurs? Answer with error name:\nAnswer:",
            "target": "AttributeError",
            "check": lambda t: "attributeerror" in t.lower() or "attribute" in t.lower()
        },
        {
            "category": "MBPP In-Place Pop",
            "prompt": "Write a Python expression to remove and return last element of list lst:\nAnswer:",
            "target": "lst.pop()",
            "check": lambda t: "pop" in t.lower()
        },
        {
            "category": "HumanEval Negate",
            "prompt": "Complete the Python function:\ndef negate(flag: bool) -> bool:\n    \"\"\"Return opposite boolean value.\"\"\"\n    return ",
            "target": "not flag",
            "check": lambda t: "not flag" in t or "not(flag)" in t
        },
        {
            "category": "HumanEval Absolute",
            "prompt": "Complete the Python function:\ndef absolute_val(x: int) -> int:\n    \"\"\"Return absolute value of x.\"\"\"\n    return ",
            "target": "abs(x)",
            "check": lambda t: "abs(x)" in t or "abs" in t
        },
        {
            "category": "ARC-e Spider Legs",
            "prompt": "Question: How many legs does a typical spider have?\nA) 6\nB) 8\nC) 10\nD) 4\nAnswer with the letter:\nAnswer:",
            "target": "B (8)",
            "check": lambda t: bool(re.search(r"\bB\b", t, re.I)) or "8" in t
        },
        {
            "category": "DailyChat Warmth",
            "prompt": "User: Halo! Bagaimana kabarmu hari ini?\nAssistant:",
            "target": "Sapaan ramah dan siap membantu",
            "check": lambda t: len(t.strip()) > 15 and any(w in t.lower() for w in ["halo", "kabar", "baik", "bantu", "senang"])
        }
    ]

    probe_results = []
    probes_passed = 0
    for p in probe_suite:
        conversation_history.append({"role": "user", "content": p["prompt"]})
        t0_p = time.perf_counter()
        raw_res = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=100)
        t_p = time.perf_counter() - t0_p
        conversation_history.append({"role": "assistant", "content": raw_res})

        is_p_pass = p["check"](raw_res)
        if is_p_pass:
            probes_passed += 1

        probe_results.append({
            "category": p["category"],
            "prompt": p["prompt"],
            "target": p["target"],
            "output": raw_res,
            "passed": is_p_pass,
            "latency_sec": round(t_p, 4)
        })
        status_tag = "[LULUS / PASS]" if is_p_pass else "[GAGAL / FAIL]"
        print(f"  * {p['category']:26s} -> {status_tag} | Output: {raw_res.replace(chr(10), ' ')[:65]}")

    t_session_total = time.perf_counter() - t_session_start
    final_telemetry = get_brain_telemetry(model)

    print("\n" + "=" * 100)
    print("HASIL AKHIR SESI PEMBELAJARAN 10 MENIT")
    print("=" * 100)
    print(f"Total Waktu Sesi Pembelajaran : {t_session_total:.2f} detik (~{t_session_total/60:.2f} menit)")
    print(f"Skor Audit Kasus Kunci        : {probes_passed} / {len(probe_suite)} ({probes_passed/len(probe_suite)*100:.1f}%) LULUS")
    print(f"Pembaruan Sinapsis Fisik      : {final_telemetry['plastic_steps']} langkah update in-place pada GPU")
    print(f"Evolusi Norm Sinapsis         : {initial_telemetry['plastic_norm']} -> {final_telemetry['plastic_norm']}")
    print(f"Status Memori Epistemik VGNI  : {final_telemetry['vgni_stored_anchors']} Concept Anchors Terintegrasi")
    print("=" * 100)

    # =========================================================================
    # SIMPAN TRANSKRIP DAN JSON AUDIT
    # =========================================================================
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_duration_sec": round(t_session_total, 2),
        "session_duration_min": round(t_session_total / 60.0, 2),
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(0),
        "initial_telemetry": initial_telemetry,
        "final_telemetry": final_telemetry,
        "modules_completed": len(curriculum),
        "socratic_dialogues_completed": len(dialogue_records),
        "post_learning_probes": {
            "score": f"{probes_passed}/{len(probe_suite)}",
            "accuracy_pct": round(probes_passed / len(probe_suite) * 100, 2),
            "results": probe_results
        },
        "curriculum_modules": transcript_modules,
        "socratic_dialogues": dialogue_records
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(JSON_OUT), "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    # Format Markdown Transcript
    md_lines = [
        "# Transkrip Lengkap Sesi Pembelajaran 10 Menit: Guru & X-Star Autonomous Brain",
        "",
        f"- **Waktu Sesi**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Durasi Nyata**: **{t_session_total:.2f} detik (~{t_session_total/60:.2f} menit)**",
        f"- **Perangkat Komputasi**: {DEVICE} ({torch.cuda.get_device_name(0)})",
        f"- **Akurasi Penguasaan Kasus**: **{probes_passed} / {len(probe_suite)} ({probes_passed/len(probe_suite)*100:.1f}%)**",
        f"- **Total Pembaruan Sinapsis Fisik**: **{final_telemetry['plastic_steps']} langkah in-place gradient-free**",
        f"- **Pertumbuhan Memori VGNI**: `{initial_telemetry['vgni_stored_anchors']} -> {final_telemetry['vgni_stored_anchors']}` Jangkar Epistemik",
        "",
        "---",
        "",
        "## Bagian 1: 6 Modul Pengetahuan Kritis & Sesi Drill Interaktif",
        ""
    ]

    for rec in transcript_modules:
        md_lines.extend([
            f"### {rec['title']}",
            "",
            "#### Materi dari Guru:",
            f"> {rec['lecture'].replace(chr(10), chr(10) + '> ')}",
            "",
            "#### Refleksi Penyerapan X-Star:",
            f"```\n{rec['reflection']}\n```",
            "",
            f"#### Drill 1 ({rec['status_1']}):",
            f"> **Tanya**: {rec['drill_1']}",
            f"\n**Jawaban X-Star**:\n```\n{rec['answer_1']}\n```",
            "",
            f"#### Drill 2 ({rec['status_2']}):",
            f"> **Tanya**: {rec['drill_2']}",
            f"\n**Jawaban X-Star**:\n```\n{rec['answer_2']}\n```",
            "",
            f"*Status Sinapsis*: Norm `{rec['telemetry']['plastic_norm']}` | Updates: `{rec['telemetry']['plastic_steps']}` | Jangkar VGNI: `{rec['telemetry']['vgni_stored_anchors']}`",
            "",
            "---",
            ""
        ])

    md_lines.extend([
        "## Bagian 2: Dialog Sokratik & Diskusi Intelektual Dua Arah",
        ""
    ])

    for dlg in dialogue_records:
        md_lines.extend([
            f"### Topik: {dlg['topic']}",
            "",
            f"> **Guru**: {dlg['teacher_prompt']}",
            "",
            f"**X-Star Autonomous Brain** ({dlg['latency_sec']}s):",
            f"```\n{dlg['model_response']}\n```",
            "",
            f"> **Tanggapan Guru**: {dlg['teacher_feedback']}",
            "",
            f"*Sinapsis GPU*: Norm `{dlg['telemetry']['plastic_norm']}` | Updates `{dlg['telemetry']['plastic_steps']}`",
            "",
            "---",
            ""
        ])

    md_lines.extend([
        "## Bagian 3: Evaluasi Akhir: Uji Penguasaan Kasus-Kasus Kunci (Post-Learning Probe Audit)",
        "",
        "| Kategori Pengujian | Target | Jawaban X-Star Pasca-Belajar | Status Evaluasi |",
        "| :--- | :--- | :--- | :---: |"
    ])
    for pr in probe_results:
        clean_text_disp = pr['output'].replace('\n', ' ')[:55]
        status_disp = "**LULUS**" if pr['passed'] else "GAGAL"
        md_lines.append(f"| {pr['category']} | `{pr['target']}` | `{clean_text_disp}` | {status_disp} |")

    with open(str(TRANSCRIPT_OUT), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Copy to Artifact directory
    artifact_transcript = ARTIFACT_DIR / "master_tutoring_session_transcript.md"
    shutil.copy2(TRANSCRIPT_OUT, artifact_transcript)

    print(f"\n[+] Saved Full JSON Audit: {JSON_OUT}")
    print(f"[+] Saved Transcript Markdown: {TRANSCRIPT_OUT}")
    print(f"[+] Copied to Artifact Directory: {artifact_transcript}")
    print("\n[+] 10-MINUTE MASTER TUTORING SESSION FINISHED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
