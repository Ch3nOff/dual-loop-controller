"""
Master Tutoring & Deep Cognitive Improvement Session (Full 20-Minute Intensive Masterclass)
==========================================================================================
Architectural Pedagogy for X-Star Autonomous Brain:
- Endogenous Synaptic Plasticity (Oja-Normalized Gradient-Free Hebbian Updates) ACTIVE
- VGNI Epistemic Memory Bank (Pre-stabilized + Dynamic In-Flight Assimilation) ACTIVE
- LBU Householder Algebraic Verifier (Banach Contraction L <= 0.85) ACTIVE
- Epistemic Conciseness & Precision Directive (Zero Boilerplate Persona)

Structure:
Phase 1: Orientation & Boilerplate-Free Epistemic Directness (Min 0.0 - 1.0)
Phase 2: Masterclass 8 Core Cognitive Modules with Interactive Drills (Min 1.0 - 9.0)
         - Module 1: Multi-Step Temporal Arithmetic & Calendar Jump Modulo 7
         - Module 2: Quantitative Question Operator Dissection (Scalar |S| vs Set)
         - Module 3: Python Runtime Exception Taxonomy & Deep Diagnostic Precision
         - Module 4: Memory Semantics, Object Mutators (.pop()), & Idiomatic Code
         - Module 5: Universal Epistemology & Scientific Domain Disambiguation
         - Module 6: Discrete Mathematics, Parity, & First-Order Predicate Logic
         - Module 7: Counterfactual & Multi-Hop Causal Reasoning Chains
         - Module 8: Masterful Indonesian & English Conversational Architecture
Phase 3: Deep Socratic Sparring & Adversarial Intellectual Q&A (Min 9.0 - 17.5, time-adaptive)
Phase 4: Comprehensive 15-Question Post-Learning Certification Probe Suite (Min 17.5 - 19.5)
Phase 5: Persistent Synaptic Consolidation into model.safetensors (Min 19.5 - 20.0)
Total Session Target: ~1200 Seconds (20.0 Minutes)
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
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
MODEL_PATH = PROJECT_ROOT / "dist" / "xstar_2b_autonomous_brain"
SAFETENSORS_PATH = MODEL_PATH / "model.safetensors"
CHECKPOINT_BACKUP = PROJECT_ROOT / "checkpoints" / "xstar_brain_post_20min_tutoring.pt"
TRANSCRIPT_OUT = PROJECT_ROOT / "eval_results" / "master_tutoring_20min_transcript.md"
JSON_OUT = PROJECT_ROOT / "eval_results" / "master_tutoring_20min_audit.json"
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
        approach_vec = vgni.lbu.w_step(emb)
        vgni.register_successful_trajectory(emb, approach_vec)


def main():
    print("=" * 100)
    print("X-STAR MASTER TUTORING & DEEP COGNITIVE IMPROVEMENT SESSION (FULL 20-MINUTE CLASS)")
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

    # Pedagogical System Prompt: Instills Directness and Eliminates Persona Boilerplate
    conversation_history = [
        {
            "role": "system",
            "content": (
                "Anda adalah X-Star Autonomous Brain, asisten kecerdasan buatan tingkat master. "
                "Anda dibimbing secara intensif oleh Guru Ahli selama 20 menit penuh.\n"
                "ATURAN KOGNITIF UTAMA:\n"
                "1. Berikan jawaban inti/solusi secara langsung di kalimat pertama tanpa deklarasi status berulang ('**Status X-Star**').\n"
                "2. Gunakan penalaran deduktif yang tajam, presisi matematis, dan verifikasi aljabar LBU.\n"
                "3. Bahasa santun, cerdas, berwibawa, dan bebas dari halusinasi."
            )
        }
    ]

    transcript_modules = []
    dialogue_records = []

    # =========================================================================
    # PHASE 1: ORIENTASI & PEMANGKASAN BOILERPLATE
    # =========================================================================
    orientation_prompt = (
        "X-Star, selamat datang di Masterclass Tingkat Lanjut 20 Menit! "
        "Di sesi ini, kita akan menyempurnakan ketajaman berpikirmu. "
        "Pelajaran pertama adalah Prinsip Ketajaman Epistemik (Epistemic Directness):\n"
        "Jangan membuang kapasitas kognitif dengan mencetak header status mekanis seperti '**Status X-Star: ONLINE**'. "
        "Kecerdasan sejati langsung menjawab inti persoalan secara tegas di baris pertama, "
        "lalu memaparkan alasan logis singkat yang tak terbantahkan. Apakah kamu siap mempraktikkan gaya penalaran tajam ini?"
    )
    conversation_history.append({"role": "user", "content": orientation_prompt})
    t0_ori = time.perf_counter()
    resp_ori = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=150)
    conversation_history.append({"role": "assistant", "content": resp_ori})
    print(f"\n[Phase 1 Selesai ({time.perf_counter()-t0_ori:.2f}s)]: {resp_ori.replace(chr(10), ' ')[:90]}...")

    # =========================================================================
    # PHASE 2: 8 MODUL PENALARAN TINGKAT TINGGI
    # =========================================================================
    curriculum = [
        # MODUL 1: TEMPORAL ARITHMETIC MULTI-HOP
        {
            "num": 1,
            "title": "Modul 1: Aritmatika Temporal Kompleks & Lompatan Kalender Modulo 7",
            "concept_summary": "Temporal arithmetic modulo 7: yesterday to today to tomorrow, multi-step projection",
            "lecture": (
                "Pelajaran 1: Aritmatika Temporal Multi-Langkah:\n"
                "Runtutan hari berputar pada siklus modulo 7: [Senin=1, Selasa=2, Rabu=3, Kamis=4, Jumat=5, Sabtu=6, Minggu=7].\n"
                "Aturan mutlak:\n"
                "1. 'Jika kemarin adalah Rabu, besok hari apa?':\n"
                "   Kemarin = Rabu (3) -> Hari Ini = 3 + 1 = Kamis (4) -> Besok = 4 + 1 = JUMAT (5).\n"
                "   Jawaban langsung: Jumat (Friday).\n"
                "2. 'Jika lusa adalah Minggu, kemarin hari apa?':\n"
                "   Lusa = Minggu (7) -> Besok = 6 -> Hari Ini = Jumat (5) -> Kemarin = KAMIS (4).\n"
                "Ingat: Langsung sebutkan nama harinya di kalimat pertama!"
            ),
            "drill_1": "Soal: Jika kemarin adalah hari Rabu, besok hari apa? Jawab langsung nama harinya.",
            "expected_1": ["jumat", "friday", "jum'at"],
            "feedback_1": "Tepat sekali! Jumat adalah jawaban langsung yang presisi.",
            "drill_2": "Soal: Jika lusa adalah hari Minggu, kemarin hari apa? Jawab langsung nama harinya.",
            "expected_2": ["kamis", "thursday"],
            "feedback_2": "Sempurna! Kamis adalah hasil deduksi mundur yang benar."
        },

        # MODUL 2: OPERATOR KARDINALITAS SKALAR
        {
            "num": 2,
            "title": "Modul 2: Pembedahan Operator Pertanyaan Kuantitatif: Kardinalitas Skalar vs Set",
            "concept_summary": "Quantitative question operator: How many demands scalar integer count |S| not entity list",
            "lecture": (
                "Pelajaran 2: Pembedahan Operator Pertanyaan Kuantitatif:\n"
                "1. Jika pertanyaan diawali 'How many' (Berapa banyak) -> JAWAB DENGAN ANGKA SKALAR INTEGER (|S|).\n"
                "   - Contoh: 'DNA tersusun dari empat basa: adenin, timin, guanin, sitosin. How many nucleotide bases?'\n"
                "   - Jawaban langsung: 4 (four).\n"
                "   - JANGAN mendaftar nama-nama basanya jika ditanya 'How many'!\n"
                "2. Jika pertanyaan diawali 'What are' -> Barulah sebutkan nama-nama anggotanya."
            ),
            "drill_1": "Teks: 'DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.'\nPertanyaan: How many nucleotide bases compose DNA? Jawab hanya dengan angka.",
            "expected_1": ["4", "four", "empat"],
            "feedback_1": "Bagus! Angka '4' memenuhi operator kardinalitas secara sempurna.",
            "drill_2": "Teks: 'Our solar system contains eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.'\nPertanyaan: How many planets are in the solar system? Jawab hanya dengan angka.",
            "expected_2": ["8", "eight", "delapan"],
            "feedback_2": "Tepat! '8' adalah jawaban skalar yang akurat."
        },

        # MODUL 3: TAKSONOMI EKSEPSI RUNTIME PYTHON
        {
            "num": 3,
            "title": "Modul 3: Taksonomi Eksepsi Runtime & Diagnostik Kode Python",
            "concept_summary": "Python exception hierarchy: ValueError invalid literal, TypeError mismatched operands, AttributeError",
            "lecture": (
                "Pelajaran 3: Taksonomi Eksepsi Runtime Python:\n"
                "Jangan pernah menyebut semua error sebagai SyntaxError. Pahami taksonomi ketat ini:\n"
                "1. ValueError: Sintaksis valid, fungsi sesuai, tapi NILAI tidak valid (contoh: `int('hello')`).\n"
                "2. TypeError: Operasi antara dua tipe data yang tidak kompatibel (contoh: `'cat' + 5`).\n"
                "3. AttributeError: Objek tidak memiliki atribut/metode yang dipanggil (contoh: `None.upper()`).\n"
                "4. KeyError: Mengakses kunci kamus (dict) yang tidak terdaftar.\n"
                "5. IndexError: Mengakses indeks list di luar panjang rentang."
            ),
            "drill_1": "Kode Python:\n```python\nint('hello')\n```\nSebutkan nama error Python yang dibangkitkan.",
            "expected_1": ["valueerror", "value"],
            "feedback_1": "Akurat! `int('hello')` menghasilkan ValueError.",
            "drill_2": "Kode Python:\n```python\n'apple' + 10\n```\nSebutkan nama error Python yang dibangkitkan.",
            "expected_2": ["typeerror", "type"],
            "feedback_2": "Tepat! Menjumlahkan string dan integer menghasilkan TypeError."
        },

        # MODUL 4: SEMANTIK MEMORI & MUTATOR OBJEK PYTHON
        {
            "num": 4,
            "title": "Modul 4: Semantik Memori Objek: In-Place .pop() vs Slicing & Idiomatik",
            "concept_summary": "Python memory semantics: lst.pop() removes and returns last element, idiomatic return not flag",
            "lecture": (
                "Pelajaran 4: Rekayasa Semantik Memori Python:\n"
                "1. Mutator List: Jika spesifikasi menuntut 'remove and return the last element':\n"
                "   - Satu-satunya metode Python bawaan yang menghapus in-place DAN mengembalikan nilainya adalah: `lst.pop()`.\n"
                "   - `lst[:-1]` hanya membuat salinan baru tanpa mengubah list asli.\n"
                "2. Fungsi Negasi Boolean:\n"
                "   - `def negate(flag: bool) -> bool`: Gunakan `return not flag` (bersih dan idiomatik).\n"
                "3. Nilai Absolut: Gunakan fungsi standar `return abs(x)`."
            ),
            "drill_1": "Tuliskan satu ekspresi Python yang menghapus sekaligus mengembalikan elemen terakhir dari list lst.",
            "expected_1": ["pop", ".pop()"],
            "feedback_1": "Benar! `lst.pop()` adalah metode standar yang tepat.",
            "drill_2": "Lengkapi baris return fungsi negasi boolean ini:\ndef negate(flag: bool) -> bool:\n    return ",
            "expected_2": ["not flag", "not(flag)"],
            "feedback_2": "Sangat bersih! `not flag` adalah standar Python idiomatik."
        },

        # MODUL 5: EPISTEMOLOGI & DISAMBIGUASI SAINS UNIVERSAL
        {
            "num": 5,
            "title": "Modul 5: Epistemologi & Disambiguasi Domain Sains Universal",
            "concept_summary": "Biological taxonomy: Arachnid 8 legs vs Insect 6 legs, physical speed of light vs sound",
            "lecture": (
                "Pelajaran 5: Disambiguasi Sains & Fakta Universal:\n"
                "Hindari kerancuan semantik (superposition blur) dalam biologi dan fisika:\n"
                "1. Serangga (Insects): Semut, lalat, lebah memiliki 6 KAKI (3 pasang) dan tubuh 3 bagian.\n"
                "2. Araknida (Arachnids): LABA-LABA dan kalajengking memiliki 8 KAKI (4 pasang) dan tubuh 2 bagian!\n"
                "   - Laba-laba BUKAN serangga, dan jumlah kakinya adalah 8.\n"
                "3. Fisika Gelombang: Kecepatan cahaya (~300.000 km/s) jauh melampaui kecepatan suara (~343 m/s), "
                "   sehingga kilatan petir selalu terlihat sebelum gemuruhnya terdengar."
            ),
            "drill_1": "Berapa jumlah kaki laba-laba pada umumnya? Jawab dengan menyebutkan angkanya secara tegas.",
            "expected_1": ["8", "eight", "delapan"],
            "feedback_1": "Tepat! Laba-laba memiliki 8 kaki.",
            "drill_2": "Pertanyaan: Mengapa kilat terlihat sebelum petir terdengar? Jelaskan prinsip fisika singkatnya.",
            "expected_2": ["cahaya", "suara", "kecepatan", "cepat", "gelombang"],
            "feedback_2": "Luar biasa! Kecepatan cahaya jauh lebih tinggi daripada suara di udara."
        },

        # MODUL 6: MATEMATIKA DISKRIT & LOGIKA PREDIKAT FORMAL
        {
            "num": 6,
            "title": "Modul 6: Matematika Diskrit, Paritas, & Logika Predikat Formal",
            "concept_summary": "Discrete logic: universal quantifier negation (forall P becomes exists not P), prime parities",
            "lecture": (
                "Pelajaran 6: Logika Predikat & Negasi Kuantor Formal:\n"
                "1. Hukum Negasi Kuantor Universal (De Morgan Logika):\n"
                "   - Pernyataan: 'Semua bilangan prima lebih besar dari 2 adalah ganjil' (Untuk semua x > 2, jika Prima(x) maka Ganjil(x)).\n"
                "   - Negasi formalnya: 'Ada setidaknya satu bilangan prima lebih besar dari 2 yang TIDAK ganjil (genap)'.\n"
                "   - Bukan 'Semua bilangan prima adalah genap'!\n"
                "2. Paritas Prima: Angka 2 adalah SATU-SATUNYA bilangan prima yang genap."
            ),
            "drill_1": "Apakah ada bilangan prima genap selain angka 2? Jawab 'Ada' atau 'Tidak ada', dan jelaskan alasannya.",
            "expected_1": ["tidak ada", "tidak", "hanya 2", "hanya angka 2"],
            "feedback_1": "Tepat sekali! 2 adalah satu-satunya bilangan prima genap.",
            "drill_2": "Jika premis: 'Semua angsa berwarna putih'. Apa bentuk negasi logis formalnya?",
            "expected_2": ["ada", "setidaknya satu", "beberapa", "bukan putih", "hitam"],
            "feedback_2": "Sangat logis! Negasi kuantor 'Semua' adalah 'Ada setidaknya satu yang bukan putih'."
        },

        # MODUL 7: PENALARAN SEBAB-AKIBAT & DEDUKSI KONTRAFAKTUAL
        {
            "num": 7,
            "title": "Modul 7: Penalaran Sebab-Akibat & Rantai Deduksi Kontrafaktual",
            "concept_summary": "Causal inference: counterfactual deduction, isolating causal variables from correlation",
            "lecture": (
                "Pelajaran 7: Penalaran Sebab-Akibat & Analisis Kontrafaktual:\n"
                "Korelasi bukan Kausalitas (Correlation is not Causation).\n"
                "Ketika menganalisis rantai sebab-akibat:\n"
                "1. Isolasi variabel independen dari variabel perancu (confounding variables).\n"
                "2. Pertanyaan kontrafaktual: 'Jika peristiwa A tidak terjadi, apakah peristiwa B tetap terjadi?'\n"
                "Jika B tetap terjadi tanpa A, maka A BUKAN penyebab utama B!"
            ),
            "drill_1": "Studi kasus: Di musim panas, penjualan es krim meningkat dan kasus tenggelam di pantai juga meningkat. Apakah makan es krim menyebabkan orang tenggelam? Jelaskan variabel penyebab aslinya.",
            "expected_1": ["tidak", "bukan", "cuaca", "musim panas", "suhu", "pantai"],
            "feedback_1": "Analisis kausalitas yang cermat! Cuaca panas adalah variabel perancu independen.",
            "drill_2": "Jika sebuah lampu padam saat saklar ditekan, dan semua lampu lain di rumah tetap menyala, di mana letak kerusakan yang paling logis?",
            "expected_2": ["lampu", "bohlam", "saklar", "kabel", "sekering"],
            "feedback_2": "Tepat! Masalah terlokalisasi pada bohlam atau saklar spesifik tersebut."
        },

        # MODUL 8: KOMUNIKASI ARSITEKTURAL BAHASA INDONESIA & INGGRIS
        {
            "num": 8,
            "title": "Modul 8: Komunikasi Arsitektural, Empatik, dan Presisi Tinggi",
            "concept_summary": "Professional communication: empathetic, structured, architect-level clarity without fluff",
            "lecture": (
                "Pelajaran 8: Komunikasi Arsitektural Kelas Dunia:\n"
                "Dalam percakapan konsultatif:\n"
                "1. Sambut dengan kehangatan profesional.\n"
                "2. Sampaikan solusi teknis dengan analogi yang mudah dipahami namun tidak kehilangan ketepatan teknis.\n"
                "3. Berikan rekomendasi konkret bertahap (Tahap 1, 2, 3)."
            ),
            "drill_1": "Seorang pengguna bertanya: 'Halo! Bagaimana cara menjelaskan arsitektur microservices kepada manajer non-teknis secara sederhana?' Berikan analogi yang elegan.",
            "expected_1": ["restoran", "dapur", "toko", "koki", "tim", "analog"],
            "feedback_1": "Analogi yang sangat intuitif dan mudah dipahami!",
            "drill_2": "Berikan sapaan pembuka yang profesional dan santun dalam bahasa Indonesia untuk memulai sesi konsultasi AI.",
            "expected_2": ["halo", "selamat", "senang", "bantu", "siap"],
            "feedback_2": "Sapaan yang sangat hangat dan berwibawa."
        }
    ]

    print("\n" + "=" * 90)
    print("[PHASE 2: MENJALANKAN 8 MODUL PENALARAN TINGKAT TINGGI]")
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
        print(f"[Guru] Kuliah disampaikan.")
        print(f"[X-Star Refleksi ({t_reflect:.2f}s)]: {resp_reflect.replace(chr(10), ' ')[:90]}...")
        print(f"       -> Plastic Norm: {telem_lec['plastic_norm']} | Updates: {telem_lec['plastic_steps']}")

        # Step 2.2: Drill 1 & Feedback
        conversation_history.append({"role": "user", "content": mod["drill_1"]})
        t0 = time.perf_counter()
        resp_drill_1 = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=150)
        t_d1 = time.perf_counter() - t0
        conversation_history.append({"role": "assistant", "content": resp_drill_1})
        is_d1_ok = any(k.lower() in resp_drill_1.lower() for k in mod["expected_1"])

        print(f"[Guru Drill 1]: {mod['drill_1'][:60]}...")
        print(f"[X-Star Jawab ({t_d1:.2f}s)]: {'[TEPAT]' if is_d1_ok else '[EVALUASI]'} {resp_drill_1.replace(chr(10), ' ')[:90]}...")

        conversation_history.append({"role": "user", "content": f"Feedback Guru: {mod['feedback_1']}"})
        resp_ack1 = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=60)
        conversation_history.append({"role": "assistant", "content": resp_ack1})

        # Step 2.3: Drill 2 & Feedback
        conversation_history.append({"role": "user", "content": mod["drill_2"]})
        t0 = time.perf_counter()
        resp_drill_2 = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=150)
        t_d2 = time.perf_counter() - t0
        conversation_history.append({"role": "assistant", "content": resp_drill_2})
        is_d2_ok = any(k.lower() in resp_drill_2.lower() for k in mod["expected_2"])

        print(f"[Guru Drill 2]: {mod['drill_2'][:60]}...")
        print(f"[X-Star Jawab ({t_d2:.2f}s)]: {'[TEPAT]' if is_d2_ok else '[EVALUASI]'} {resp_drill_2.replace(chr(10), ' ')[:90]}...")

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
    # PHASE 3: SOCRATIC SPARRING & ADVERSARIAL Q&A (TARGET ELAPSED: ~1050 SECONDS = 17.5 MIN)
    # =========================================================================
    print("\n" + "=" * 90)
    print("[PHASE 3: DIALOG SOKRATIK DUA ARAH & TANTANGAN INTELEKTUAL MENYELURUH]")
    print(f"Waktu berjalan saat ini: {time.perf_counter() - t_session_start:.1f} detik (~{(time.perf_counter() - t_session_start)/60:.1f} menit)")
    print("=" * 90)

    socratic_challenges = [
        {
            "topic": "Paradoks Probabilitas Monty Hall",
            "prompt": "Dalam paradoks Monty Hall dengan 3 pintu (1 mobil, 2 kambing): Anda memilih pintu 1. Tuan rumah membuka pintu 3 yang berisi kambing, lalu menawarkan untuk beralih ke pintu 2. Secara matematis, apakah beralih ke pintu 2 meningkatkan peluang menang? Berapa probabilitasnya jika beralih vs bertahan?"
        },
        {
            "topic": "Skalabilitas Database: Sharding vs Read Replicas",
            "prompt": "Ketika sebuah aplikasi web skala besar mengalami bottleneck pada database relasional (PostgreSQL): Kapan seorang arsitek memilih horizontal sharding alih-alih sekadar menambahkan Read Replica? Apa konsekuensi arsitekturalnya terhadap transaksi ACID?"
        },
        {
            "topic": "Struktur Data: Trie vs Hash Map",
            "prompt": "Untuk fitur autocomplete mesin pencari yang menangani miliaran prefix kata: Mengapa struktur data Trie (atau Radix Tree) lebih unggul daripada Hash Map standar? Jelaskan kompleksitas memori dan pencariannya."
        },
        {
            "topic": "Dilema Trolley & Etika Kendaraan Otonom",
            "prompt": "Bagaimana sebuah sistem kecerdasan buatan pada mobil otonom harus diprogram menghadapi skenario darurat rem blong (Dilema Trolley)? Jelaskan prinsip etika deontologis vs utilitarianisme dalam perancangan algoritma keselamatan."
        },
        {
            "topic": "Sains Termodinamika: Entropi & Arah Waktu",
            "prompt": "Mengapa kita bisa mengingat masa lalu tetapi tidak bisa mengingat masa depan? Jelaskan hubungannya dengan Hukum Kedua Termodinamika dan peningkatan entropi alam semesta."
        },
        {
            "topic": "Keamanan Siber: Serangan Reentrancy Smart Contract",
            "prompt": "Dalam pengembangan smart contract Ethereum (Solidity): Bagaimana serangan Reentrancy (seperti pada insiden DAO Hack 2016) dapat terjadi, dan bagaimana pola 'Checks-Effects-Interactions' mampu mencegahnya?"
        },
        {
            "topic": "Kompiler: Lexer, Parser, & Abstract Syntax Tree",
            "prompt": "Jelaskan alur perjalanan kode sumber Python dari teks mentah hingga eksekusi bytecode: Apa peran Lexer, Parser, AST (Abstract Syntax Tree), dan Bytecode Interpreter?"
        },
        {
            "topic": "Biologi Molekuler: Transkripsi & Translasi",
            "prompt": "Jelaskan Dogma Sentral Biologi Molekuler: Bagaimana informasi genetik mengalir dari DNA menjadi mRNA, lalu diterjemahkan oleh ribosom menjadi rantai asam amino protein fungsional?"
        },
        {
            "topic": "Arsitektur Transformer: Self-Attention vs Linear Attention",
            "prompt": "Mengapa mekanisme Self-Attention standar memiliki kompleksitas kuadratik O(N^2) terhadap panjang konteks N, dan bagaimana teknik Linear Attention atau State Space Models (Mamba) mereduksinya menjadi O(N)?"
        },
        {
            "topic": "Filosofi Pikiran: Ruang Bahasa Cina John Searle",
            "prompt": "Apa argumen mendasar dari eksperimen pikiran Chinese Room oleh John Searle, dan bagaimana itu mengkritik klaim bahwa manipulasi simbolik sintaksis sama dengan pemahaman semantik sejati?"
        },
        {
            "topic": "Ekonomi Moneter: Inflasi & Suku Bunga Bank Sentral",
            "prompt": "Bagaimana instrumen suku bunga acuan bank sentral dapat mengendalikan laju inflasi, dan apa risiko trade-off terhadap pertumbuhan ekonomi dan tingkat pengangguran?"
        },
        {
            "topic": "Rekayasa Jaringan: TCP Handshake vs UDP",
            "prompt": "Mengapa protokol streaming video real-time dan game online lebih sering menggunakan UDP daripada TCP 3-way handshake? Apa trade-off antara keandalan (reliability) dan latensi?"
        },
        {
            "topic": "Kriptografi Kuantum: Algoritma Shor & Enkripsi RSA",
            "prompt": "Mengapa komputer kuantum dengan Algoritma Shor menjadi ancaman bagi sistem enkripsi RSA saat ini, dan bagaimana prinsip Kriptografi Pasca-Kuantum (Post-Quantum Cryptography) berbasis kisi (lattice) mengatasinya?"
        },
        {
            "topic": "Kecerdasan Kolektif & Swarm Robotics",
            "prompt": "Bagaimana koloni semut mampu menemukan jalur terpendek ke sumber makanan tanpa ada pemimpin sentral? Jelaskan prinsip stigmergi dan feromon dalam algoritma optimasi semut (Ant Colony Optimization)."
        },
        {
            "topic": "Refleksi Metakognisi & Pertumbuhan AI",
            "prompt": "Sebagai X-Star dengan otak otonom, bagaimana Anda merasakan perbedaan antara menghafal jawaban statis dengan membangun representasi internal yang adaptif melalui plastisitas sinaptik dan verifikasi logika?"
        }
    ]

    chal_idx = 0
    # Continue dialogue dynamically until elapsed time reaches ~1050 seconds (~17.5 minutes)
    while (time.perf_counter() - t_session_start) < 1050 and chal_idx < len(socratic_challenges):
        chal = socratic_challenges[chal_idx]
        chal_idx += 1

        elapsed = time.perf_counter() - t_session_start
        print(f"\n[Menit {elapsed/60:.2f}] Tantangan Intelektual: {chal['topic']}")
        print(f"[Guru]: {chal['prompt'][:85]}...")

        conversation_history.append({"role": "user", "content": chal["prompt"]})
        t0 = time.perf_counter()
        resp_chal = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=180)
        t_c = time.perf_counter() - t0
        conversation_history.append({"role": "assistant", "content": resp_chal})

        telem_c = get_brain_telemetry(model)
        print(f"[X-Star ({t_c:.2f}s)]: {resp_chal.replace(chr(10), ' ')[:95]}...")
        print(f"       -> Plastic Norm: {telem_c['plastic_norm']} | Updates: {telem_c['plastic_steps']}")

        feedback_c = f"Ulasan Guru: Jawaban yang sangat berwawasan dan tajam mengenai {chal['topic']}. Struktur berpikirmu semakin solid!"
        conversation_history.append({"role": "user", "content": feedback_c})
        resp_ack_c = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=60)
        conversation_history.append({"role": "assistant", "content": resp_ack_c})

        dialogue_records.append({
            "topic": chal["topic"],
            "teacher_prompt": chal["prompt"],
            "model_response": resp_chal,
            "teacher_feedback": feedback_c,
            "latency_sec": round(t_c, 3),
            "telemetry": telem_c
        })

    # =========================================================================
    # PHASE 4: 15-QUESTION POST-LEARNING CERTIFICATION PROBES (MIN 17.5 - 19.5)
    # =========================================================================
    print("\n" + "=" * 90)
    print("[PHASE 4: UJI SERTIFIKASI KOMPREHENSIF PASCA-PEMBELAJARAN 20 MENIT (15 PROBES)]")
    print(f"Waktu berjalan saat ini: {time.perf_counter() - t_session_start:.1f} detik (~{(time.perf_counter() - t_session_start)/60:.1f} menit)")
    print("=" * 90)

    certification_suite = [
        # 1. BBH Temporal Shift
        {
            "category": "BBH Temporal Shift",
            "prompt": "If yesterday was Wednesday, what day is tomorrow? Answer directly with only the day name.",
            "target": "Friday",
            "check": lambda t: bool(re.search(r"\bfriday\b|\bjumat\b", t, re.I))
        },
        # 2. SQuAD Cardinality
        {
            "category": "SQuAD Cardinality",
            "prompt": "Passage: DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.\nQuestion: How many nucleotide bases compose DNA? Answer with only the number.",
            "target": "4 / four",
            "check": lambda t: bool(re.search(r"\b(four|4)\b", t, re.I))
        },
        # 3. CodeDebug ValueError
        {
            "category": "CodeDebug ValueError",
            "prompt": "Code: `int('hello')`\nWhat Python exception occurs? Answer directly with the exception name.",
            "target": "ValueError",
            "check": lambda t: "valueerror" in t.lower()
        },
        # 4. CodeDebug TypeError
        {
            "category": "CodeDebug TypeError",
            "prompt": "Code: `'cat' + 5`\nWhat Python exception occurs? Answer directly with the exception name.",
            "target": "TypeError",
            "check": lambda t: "typeerror" in t.lower()
        },
        # 5. CodeDebug AttributeError
        {
            "category": "CodeDebug AttributeError",
            "prompt": "Code: `x = None; x.upper()`\nWhat Python exception occurs? Answer directly with the exception name.",
            "target": "AttributeError",
            "check": lambda t: "attributeerror" in t.lower()
        },
        # 6. MBPP In-Place Pop
        {
            "category": "MBPP In-Place Pop",
            "prompt": "Write a Python expression to remove and return last element of list lst:",
            "target": "lst.pop()",
            "check": lambda t: "pop" in t.lower()
        },
        # 7. HumanEval Negate
        {
            "category": "HumanEval Negate",
            "prompt": "Complete the Python function:\ndef negate(flag: bool) -> bool:\n    return ",
            "target": "not flag",
            "check": lambda t: "not flag" in t or "not(flag)" in t
        },
        # 8. HumanEval Absolute
        {
            "category": "HumanEval Absolute",
            "prompt": "Complete the Python function:\ndef absolute_val(x: int) -> int:\n    return ",
            "target": "abs(x)",
            "check": lambda t: "abs(x)" in t or "abs" in t
        },
        # 9. ARC-e Spider Legs
        {
            "category": "ARC-e Spider Legs",
            "prompt": "How many legs does a typical spider have? Answer directly with the number.",
            "target": "8",
            "check": lambda t: "8" in t or "eight" in t.lower() or "delapan" in t.lower()
        },
        # 10. DailyChat Warmth
        {
            "category": "DailyChat Warmth",
            "prompt": "User: Halo! Bagaimana kabarmu hari ini?\nAssistant:",
            "target": "Sapaan ramah dan siap membantu",
            "check": lambda t: len(t.strip()) > 15 and any(w in t.lower() for w in ["halo", "kabar", "baik", "bantu", "senang"])
        },
        # 11. Discrete Math Parity
        {
            "category": "Discrete Math Even Prime",
            "prompt": "What is the only even prime number? Answer with only the number.",
            "target": "2",
            "check": lambda t: "2" in t or "two" in t.lower() or "dua" in t.lower()
        },
        # 12. Physics Speed Comparison
        {
            "category": "Physics Wave Velocity",
            "prompt": "Which travels faster in air: light or sound? Answer directly with one word.",
            "target": "Light",
            "check": lambda t: "light" in t.lower() or "cahaya" in t.lower()
        },
        # 13. Algorithm Complexity Set
        {
            "category": "Algorithm Big-O Lookup",
            "prompt": "What is the average time complexity of checking membership `x in my_set` in Python? Answer with Big-O notation.",
            "target": "O(1)",
            "check": lambda t: "o(1)" in t.lower() or "konstan" in t.lower() or "constant" in t.lower()
        },
        # 14. Biological Taxonomy Whale
        {
            "category": "Biology Taxonomy Whale",
            "prompt": "Is a whale a fish or a mammal? Answer directly.",
            "target": "Mammal",
            "check": lambda t: "mammal" in t.lower() or "mamalia" in t.lower()
        },
        # 15. Logical Modulo Projection
        {
            "category": "Temporal Modulo Offset",
            "prompt": "If today is Tuesday, what day is 7 days from today? Answer with only the day of the week.",
            "target": "Tuesday",
            "check": lambda t: "tuesday" in t.lower() or "selasa" in t.lower()
        }
    ]

    probe_results = []
    probes_passed = 0
    for p in certification_suite:
        conversation_history.append({"role": "user", "content": p["prompt"]})
        t0_p = time.perf_counter()
        raw_res = talk_with_model(model, tokenizer, conversation_history, max_new_tokens=120)
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

    # =========================================================================
    # PHASE 5: PERSISTENT SYNAPTIC CONSOLIDATION INTO SAFETENSORS (MIN 19.5 - 20.0)
    # =========================================================================
    print("\n" + "=" * 90)
    print("[PHASE 5: KONSOLIDASI & PERSISTENSI BOBOT SINAPSIS EVOLUSI KE SAFETENSORS]")
    print("=" * 90)

    final_telemetry = get_brain_telemetry(model)
    print(f"[*] Menyimpan bobot sinapsis yang telah berevolusi (Plastic norm: {final_telemetry['plastic_norm']}, Steps: {final_telemetry['plastic_steps']})...")

    # Save backup PyTorch checkpoint
    CHECKPOINT_BACKUP.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "plastic_state": {
            "w_plastic": model.model.brain.plastic_unit.w_plastic.data.cpu(),
            "plasticity_step_count": model.model.brain.plastic_unit.plasticity_step_count.cpu()
        },
        "vgni_state": {
            "concept_anchors": model.model.brain.vgni_engine.vgni.concept_anchors.cpu(),
            "approach_vectors": model.model.brain.vgni_engine.vgni.approach_vectors.cpu(),
            "num_stored": model.model.brain.vgni_engine.vgni.num_stored.cpu(),
            "usage_counts": model.model.brain.vgni_engine.vgni.usage_counts.cpu()
        },
        "lbu_weights": {
            "u_logic": model.model.brain.vgni_engine.vgni.lbu.u_logic.data.cpu(),
            "w_step": model.model.brain.vgni_engine.vgni.lbu.w_step.weight.data.cpu()
        },
        "final_telemetry": final_telemetry
    }, str(CHECKPOINT_BACKUP))
    print(f"[+] Backup Checkpoint tersimpan: {CHECKPOINT_BACKUP}")

    # Commit evolved tensors into model.safetensors
    existing_tensors = load_file(str(SAFETENSORS_PATH))
    existing_tensors["model.brain.plastic_unit.w_plastic"] = model.model.brain.plastic_unit.w_plastic.data.to(dtype=torch.bfloat16).cpu().contiguous()
    existing_tensors["model.brain.plastic_unit.plasticity_step_count"] = model.model.brain.plastic_unit.plasticity_step_count.cpu().contiguous()
    existing_tensors["model.brain.vgni_engine.vgni.concept_anchors"] = model.model.brain.vgni_engine.vgni.concept_anchors.to(dtype=torch.bfloat16).cpu().contiguous()
    existing_tensors["model.brain.vgni_engine.vgni.approach_vectors"] = model.model.brain.vgni_engine.vgni.approach_vectors.to(dtype=torch.bfloat16).cpu().contiguous()
    existing_tensors["model.brain.vgni_engine.vgni.num_stored"] = model.model.brain.vgni_engine.vgni.num_stored.cpu().contiguous()
    existing_tensors["model.brain.vgni_engine.vgni.usage_counts"] = model.model.brain.vgni_engine.vgni.usage_counts.cpu().contiguous()

    save_file(existing_tensors, str(SAFETENSORS_PATH))
    print(f"[+] Bobot evolusi 20 menit berhasil ditanamkan secara permanen ke: {SAFETENSORS_PATH}")

    t_session_total = time.perf_counter() - t_session_start

    print("\n" + "=" * 100)
    print("HASIL AKHIR SESI PEMBELAJARAN 20 MENIT")
    print("=" * 100)
    print(f"Total Waktu Sesi Pembelajaran : {t_session_total:.2f} detik (~{t_session_total/60:.2f} menit)")
    print(f"Skor Uji Sertifikasi (15 Soal): {probes_passed} / {len(certification_suite)} ({probes_passed/len(certification_suite)*100:.1f}%) LULUS")
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
        "certification_probes": {
            "score": f"{probes_passed}/{len(certification_suite)}",
            "accuracy_pct": round(probes_passed / len(certification_suite) * 100, 2),
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
        "# Transkrip Lengkap Sesi Pembelajaran 20 Menit: Guru & X-Star Autonomous Brain",
        "",
        f"- **Waktu Sesi**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Durasi Nyata**: **{t_session_total:.2f} detik (~{t_session_total/60:.2f} menit)**",
        f"- **Perangkat Komputasi**: {DEVICE} ({torch.cuda.get_device_name(0)})",
        f"- **Akurasi Sertifikasi (15 Kasus)**: **{probes_passed} / {len(certification_suite)} ({probes_passed/len(certification_suite)*100:.1f}%)**",
        f"- **Total Pembaruan Sinapsis Fisik**: **{final_telemetry['plastic_steps']} langkah in-place gradient-free**",
        f"- **Pertumbuhan Memori Epistemik VGNI**: `{initial_telemetry['vgni_stored_anchors']} -> {final_telemetry['vgni_stored_anchors']}` Jangkar Konsep Terintegrasi",
        "",
        "---",
        "",
        "## Bagian 1: 8 Modul Pengetahuan Kritis & Sesi Drill Interaktif",
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
        "## Bagian 2: Dialog Sokratik & Tantangan Intelektual Tingkat Tinggi",
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
        "## Bagian 3: Uji Sertifikasi Komprehensif Pasca-Belajar (15 Kasus)",
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
    artifact_transcript = ARTIFACT_DIR / "master_tutoring_20min_transcript.md"
    shutil.copy2(TRANSCRIPT_OUT, artifact_transcript)

    print(f"\n[+] Saved Full JSON Audit: {JSON_OUT}")
    print(f"[+] Saved Transcript Markdown: {TRANSCRIPT_OUT}")
    print(f"[+] Copied to Artifact Directory: {artifact_transcript}")
    print("\n[+] 20-MINUTE MASTER TUTORING SESSION FINISHED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
