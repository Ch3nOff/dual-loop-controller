"""
Empirical Verification of "Jalur Tengah" (Hybrid Identity-Anchored Vector Evolution)
Comparing 3 Regimes:
1. Pure Heavy HADL (Left Brain / Deliberative Adapter)
2. Pure Fast Bypass (Right Brain / Native Base Qwen3.5-2B)
3. Jalur Tengah (Identity Anchor I + Tanh-Bounded Vector Projection)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dual_loop.callearn_oc_engine import (
    HADLv4ModelWrapper,
    HADLv4UnifiedEngine,
    GhostVerificationLayer
)

def test_jalur_tengah():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} ({torch.cuda.get_device_name(0)})")

    model_id = "Qwen/Qwen3.5-2B"
    print(f"[Model] Loading {model_id} in BF16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    base_model.eval()

    # Load HADL v4.2 Unified Engine & Ghost
    wrapper = HADLv4ModelWrapper(base_model, target_layer_idx=11, ghost_layer_idx=23)
    ckpt_path = Path("checkpoints/hadl_v42_ghost_unified.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        if "engine" in state:
            wrapper.engine.load_state_dict(state["engine"], strict=False)
        if "ghost" in state:
            wrapper.ghost.load_state_dict(state["ghost"], strict=False)
        print(f"[Checkpoint] Loaded HADL engine and ghost from {ckpt_path}")
    wrapper.engine.to(device=device, dtype=torch.bfloat16)
    wrapper.ghost.to(device=device, dtype=torch.bfloat16)
    wrapper.engine.eval()
    wrapper.ghost.eval()

    # 3 Hybrid Test Scenarios (Story Word Math & Analytical Essay)
    scenarios = [
        {
            "id": "story_math_id",
            "name": "Soal Cerita Aritmatika Toko Kelontong (Bahasa Indonesia)",
            "prompt": "Pak Budi memiliki sebuah toko kelontong. Di awal minggu, ia memiliki stok 120 kotak susu. Pada hari Senin, terjual 25 kotak. Pada hari Selasa, ia menerima pasokan baru sebanyak 50 kotak. Pada hari Rabu, terjual 40 kotak. Pada hari Kamis, ia membagikan 15 kotak susu yang mendekati tanggal kedaluwarsa kepada tetangganya secara gratis. Berapa sisa kotak susu di toko Pak Budi pada akhir hari Kamis? Jelaskan langkah perhitungannya dengan bahasa Indonesia yang santai, jelas, dan rapi.",
            "ground_truth_num": "90"
        },
        {
            "id": "story_math_en",
            "name": "Sarah's Multi-Step Bakery Story Problem (English)",
            "prompt": "Sarah operates an artisan bakery. She starts the day with 140 chocolate croissants. In the morning, she sells 45 croissants to early customers. At noon, she bakes a fresh batch of 60 croissants. In the afternoon, a corporate event purchases 75 croissants. Right before closing, 8 damaged croissants are discarded. How many croissants are left at closing? Walk through the step-by-step calculation with clear, friendly, and cohesive prose.",
            "ground_truth_num": "72"
        },
        {
            "id": "analytical_essay_id",
            "name": "Esai Analitis Etika AI Medis (Bahasa Indonesia)",
            "prompt": "Tuliskan sebuah esai analisis singkat (2 paragraf) tentang dilema etis penggunaan AI dalam menentukan prioritas pasien gawat darurat (triage) di rumah sakit. Paragraf pertama harus memaparkan argumen efisiensi algoritmik, dan paragraf kedua memaparkan argumen empati kemanusiaan serta bahaya bias data. Gunakan gaya bahasa formal namun mengalir.",
            "ground_truth_num": None
        }
    ]

    results = []

    # Implement Custom Hook Controller that supports 3 modes:
    # 0: Pure Fast Bypass (no modification)
    # 1: Pure Heavy HADL (full adapter replacement)
    # 2: Jalur Tengah (Identity Anchor + Tanh-bounded Projected Evolution)

    layers = base_model.model.layers
    layer_11 = layers[11]
    layer_23 = layers[23]

    current_mode = "bypass" # "bypass", "heavy", "jalur_tengah"
    cached_h11 = None
    alpha_mid = 0.25 # blend strength for Jalur Tengah

    def hook_l11(module, args, output):
        nonlocal cached_h11
        if current_mode == "bypass":
            return output

        h = output[0] if isinstance(output, tuple) else output
        h_mod, telem = wrapper.engine(h)

        if current_mode == "heavy":
            cached_h11 = h_mod
            return (h_mod,) + output[1:] if isinstance(output, tuple) else h_mod

        elif current_mode == "jalur_tengah":
            # Jalur Tengah: Identity Matrix Anchor I + Tanh-bounded Vector Delta
            # h_mid = I * h + alpha * tanh(||delta|| / ||h||) * delta_dir * ||h||
            delta = h_mod - h
            norm_h = torch.norm(h, p=2, dim=-1, keepdim=True) + 1e-6
            norm_delta = torch.norm(delta, p=2, dim=-1, keepdim=True) + 1e-6
            dir_delta = delta / norm_delta
            # Bound relative expansion to prevent subspace hijacking
            mag_scale = torch.tanh(norm_delta / norm_h)
            h_evolved = h + alpha_mid * mag_scale * dir_delta * norm_h
            cached_h11 = h_evolved
            return (h_evolved,) + output[1:] if isinstance(output, tuple) else h_evolved

        return output

    def hook_l23(module, args, output):
        nonlocal cached_h11
        if current_mode == "bypass" or cached_h11 is None:
            return output

        h_final = output[0] if isinstance(output, tuple) else output
        h_ver, telem = wrapper.ghost(h_final, cached_h11)

        if current_mode == "heavy":
            return (h_ver,) + output[1:] if isinstance(output, tuple) else h_ver

        elif current_mode == "jalur_tengah":
            # Jalur Tengah at Ghost Layer: Identity Anchor I + Tanh Bound
            delta = h_ver - h_final
            norm_h = torch.norm(h_final, p=2, dim=-1, keepdim=True) + 1e-6
            norm_delta = torch.norm(delta, p=2, dim=-1, keepdim=True) + 1e-6
            dir_delta = delta / norm_delta
            mag_scale = torch.tanh(norm_delta / norm_h)
            h_evolved = h_final + alpha_mid * mag_scale * dir_delta * norm_h
            return (h_evolved,) + output[1:] if isinstance(output, tuple) else h_evolved

        return output

    # Register hooks
    # First disable wrapper's default hooks to avoid conflict
    wrapper.enabled = False
    handle_11 = layer_11.register_forward_hook(hook_l11)
    handle_23 = layer_23.register_forward_hook(hook_l23)

    modes = [
        ("Pure Fast Bypass", "bypass"),
        ("Pure Heavy HADL", "heavy"),
        ("Jalur Tengah (Identity + Proj)", "jalur_tengah")
    ]

    for sc in scenarios:
        print(f"\n=======================================================")
        print(f"Scenario: {sc['name']}")
        print(f"Prompt: {sc['prompt'][:90]}...")
        print(f"=======================================================")

        scenario_res = {"scenario": sc["name"], "modes": {}}

        for mode_title, m_key in modes:
            current_mode = m_key
            cached_h11 = None
            wrapper.engine.reset_kv_cache()

            inputs = tokenizer(sc["prompt"], return_tensors="pt").to(device)
            t0 = time.time()
            with torch.no_grad():
                out_tokens = base_model.generate(
                    **inputs,
                    max_new_tokens=350,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            dt = time.time() - t0
            resp = tokenizer.decode(out_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            # Check for style bleed (axioms, loops)
            has_axiom = bool("axiom" in resp.lower() or "step 1: by axiom" in resp.lower())
            has_loop = bool("= 6 = 6" in resp or "= 2 = 2" in resp or "30+30=60" in resp)
            
            # Check correctness if ground truth is present
            ans_correct = None
            if sc["ground_truth_num"]:
                ans_correct = bool(sc["ground_truth_num"] in resp)

            print(f"\n--- Mode: {mode_title} ---")
            print(f"Tokens: {len(out_tokens[0]) - inputs['input_ids'].shape[1]} | Time: {dt:.2f}s")
            print(f"Correctness: {ans_correct} | Has Axiom Bleed: {has_axiom} | Has Loop: {has_loop}")
            print(f"Snippet:\n{resp[:250]}...\n")

            scenario_res["modes"][m_key] = {
                "title": mode_title,
                "resp": resp,
                "ans_correct": ans_correct,
                "has_axiom": has_axiom,
                "has_loop": has_loop,
                "time_s": dt
            }

        results.append(scenario_res)

    handle_11.remove()
    handle_23.remove()
    wrapper.enabled = True

    # Save results to json
    import json
    out_path = Path("eval_results/jalur_tengah_empirical_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    test_jalur_tengah()
