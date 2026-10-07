"""
End-to-End Verification of Mandatory Polynomial-Bounded Embedded Router (P-MVR)
& Native Jalur Tengah Integration in HADL v4.4
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from dual_loop import attach_dual_loop_v4, HADLv4ModelWrapper, PolynomialEmbeddedRouting

def run_verification():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} ({torch.cuda.get_device_name(0)})")

    model_id = "Qwen/Qwen3.5-2B"
    print(f"[Model] Loading {model_id} in BF16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    base_model.eval()

    # Mandatory Attachment: attach_dual_loop_v4
    print("\n[Attach] Calling attach_dual_loop_v4(base_model)...")
    wrapped = attach_dual_loop_v4(base_model)
    print(f"[Attach] Router instance: {type(wrapped.router).__name__}")
    print(f"[Attach] Router num_routes: {wrapped.router.num_routes}")
    print(f"[Attach] Router weights loaded: {bool(wrapped.router.m_v is not None)}")

    # Load HADL v4.2 checkpoint for engine and ghost
    ckpt_path = Path("checkpoints/hadl_v42_ghost_unified.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        if "engine" in state:
            wrapped.engine.load_state_dict(state["engine"], strict=False)
        if "ghost" in state:
            wrapped.ghost.load_state_dict(state["ghost"], strict=False)
        print(f"[Checkpoint] Loaded HADL engine and ghost from {ckpt_path}")
    wrapped.engine.to(device=device, dtype=torch.bfloat16)
    wrapped.ghost.to(device=device, dtype=torch.bfloat16)

    # Test cases representing the 3 natural regimes
    test_cases = [
        {
            "category": "Chat & Code (Expected: Fast Bypass)",
            "name": "Python Binary Search",
            "prompt": "Write a clean Python function binary_search(arr, target) with type hints and docstring."
        },
        {
            "category": "Chat & Code (Expected: Fast Bypass)",
            "name": "Daily Relaxation Advice (ID)",
            "prompt": "Halo! Akhir pekan kemarin aku capek banget karena lembur. Ada saran aktivitas santai buat recharge energi?"
        },
        {
            "category": "Hybrid Reasoning (Expected: Jalur Tengah)",
            "name": "Story Math Pak Budi (ID)",
            "prompt": "Pak Budi memiliki sebuah toko kelontong. Di awal minggu, ia memiliki stok 120 kotak susu. Pada hari Senin, terjual 25 kotak. Pada hari Selasa, ia menerima pasokan baru sebanyak 50 kotak. Pada hari Rabu, terjual 40 kotak. Pada hari Kamis, ia membagikan 15 kotak susu yang mendekati tanggal kedaluwarsa kepada tetangganya secara gratis. Berapa sisa kotak susu di toko Pak Budi pada akhir hari Kamis? Jelaskan langkah perhitungannya dengan bahasa Indonesia yang santai, jelas, dan rapi."
        },
        {
            "category": "Hybrid Reasoning (Expected: Jalur Tengah)",
            "name": "Sarah Bakery Word Problem (EN)",
            "prompt": "Sarah operates an artisan bakery. She starts the day with 140 chocolate croissants. In the morning, she sells 45 croissants to early customers. At noon, she bakes a fresh batch of 60 croissants. In the afternoon, a corporate event purchases 75 croissants. Right before closing, 8 damaged croissants are discarded. How many croissants are left at closing? Walk through the step-by-step calculation with clear, friendly, and cohesive prose."
        },
        {
            "category": "Deep Deliberation (Expected: Heavy HADL)",
            "name": "Synthetic Cryptographic X-Hash",
            "prompt": "[CRYPTO_DIFFUSION_v1] Apply deterministic bitwise permutations to block [4, 15, 6, 9]. State round keys and diffusion table. Output final 4-byte cipher block."
        },
        {
            "category": "Deep Deliberation (Expected: Heavy HADL)",
            "name": "Counterfactual Inverted Gravity",
            "prompt": "In Counterfactual Universe-K, gravity is inverted such that g = -9.8 m/s^2. A ball is dropped from h=100m. What is its velocity after 2 seconds? State the inverted invariant."
        }
    ]

    print("\n" + "=" * 80)
    print("TESTING AUTONOMOUS POLYNOMIAL ROUTING & GENERATION (NO MANUAL FORCING)")
    print("=" * 80)

    for tc in test_cases:
        inputs = tokenizer(tc["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = wrapped.generate(
                inputs["input_ids"],
                max_new_tokens=220,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        resp = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        telem = getattr(wrapped, "last_routing_scores", wrapped.engine.last_telemetry)

        p_byp = telem.get("p_bypass", 0.0)
        p_mid = telem.get("p_mid", 0.0)
        p_hvy = telem.get("p_heavy", 0.0)
        mode = telem.get("route_mode", "unknown")
        alpha = telem.get("alpha_eff", 0.0)

        print(f"\n[{tc['category']}] {tc['name']}")
        print(f"  Polynomial Probs: P=[Bypass:{p_byp:.3f}, Mid:{p_mid:.3f}, Heavy:{p_hvy:.3f}]")
        print(f"  Autonomously Chosen Route: {mode.upper()} (alpha_eff={alpha:.3f})")
        print(f"  Snippet: {resp[:180].replace(chr(10), ' ')}...")

    print("\n[Done] All cases evaluated smoothly without forcing.")

if __name__ == "__main__":
    run_verification()
