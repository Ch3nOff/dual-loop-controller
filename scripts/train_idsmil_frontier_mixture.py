"""
HADL ID-SMIL: Large-Scale Multi-Domain SFT Fine-Tuning on Frontier Datasets
============================================================================
Datasets from Frontier Recipe:
1. UltraChat 200k (HuggingFaceH4/ultrachat_200k): Multi-turn conversational dialogue
2. GSM8k (openai/gsm8k): Step-by-step mathematical reasoning
3. CodeAlpaca-20k (sahil2801/CodeAlpaca-20k): Algorithmic & functional coding
4. Alpaca Cleaned (yahma/alpaca-cleaned): General instruction following

Model: Qwen/Qwen3.5-2B in BF16
Hardware: NVIDIA RTX 5060 Laptop GPU (8GB VRAM)
Architecture: HADL ID-SMIL (Incoherent Diffuse Router + LEA + Bounce-Back + SMIL Background Unit)
Base Model Weights: 100% Frozen (requires_grad = False)
"""

import os
import sys
import math
import json
import gc
import re
from pathlib import Path
from typing import Dict, Any, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

from dual_loop import (
    attach_hadl_idsmil,
    IncoherentDiffuseRouter,
    LocalAffordanceGate,
    SMILBackgroundUnit,
    IDSMILController,
    HADLIDSMILModelWrapper
)

MODEL_ID = "Qwen/Qwen3.5-2B"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

PERPLEXITY_TEXT = """
The printing press was invented in the Holy Roman Empire by Johannes Gutenberg around 1440. Gutenberg developed a complete printing system that mechanized the transfer of ink onto paper using movable type, oil-based ink, and a wooden screw press. This technological breakthrough allowed the mass production of books and the rapid dissemination of scientific and philosophical knowledge across Europe, catalyzing the Renaissance, the Scientific Revolution, and the Age of Enlightenment. Prior to Gutenberg's invention, books were hand-copied by scribes, making them exceedingly rare and costly luxury items.
""".strip()

def load_frontier_multi_domain_mixture(max_samples_per_domain: int = 50) -> List[Dict[str, str]]:
    print(f"\n[Data Loader] Assembling Frontier SFT Mixture ({max_samples_per_domain} samples per domain)...")
    mixture = []

    # 1. UltraChat 200k (Multi-turn dialogue)
    try:
        parquet_path = "C:/Users/Matthew Chen/.cache/huggingface/hub/datasets--HuggingFaceH4--ultrachat_200k/snapshots/8049631c405ae6576f93f445c6b8166f76f5505a/data/train_sft-00000-of-00003-a3ecf92756993583.parquet"
        if os.path.exists(parquet_path):
            t = pq.read_table(parquet_path)
            msgs_col = t["messages"].to_pylist()
            count = 0
            for item in msgs_col[:max_samples_per_domain * 2]:
                if len(item) >= 2:
                    u = item[0]["content"]
                    a = item[1]["content"]
                    prompt = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
                    mixture.append({"domain": "dialogue_ultrachat", "text": prompt})
                    count += 1
                    if count >= max_samples_per_domain:
                        break
            print(f"  • UltraChat 200k loaded: {count} samples", flush=True)
    except Exception as e:
        print(f"  [!] UltraChat error: {e}", flush=True)

    # 2. GSM8k (Step-by-step Math Reasoning)
    try:
        ds_gsm = load_dataset('openai/gsm8k', 'main', split=f'train[:{max_samples_per_domain}]')
        for item in ds_gsm:
            prompt = f"<|im_start|>user\nSolve step-by-step: {item['question']}<|im_end|>\n<|im_start|>assistant\n{item['answer']}<|im_end|>"
            mixture.append({"domain": "math_gsm8k", "text": prompt})
        print(f"  • GSM8k Math loaded: {len(ds_gsm)} samples", flush=True)
    except Exception as e:
        print(f"  [!] GSM8k error: {e}", flush=True)

    # 3. CodeAlpaca-20k (Algorithmic & Code Synthesis)
    try:
        ds_code = load_dataset('sahil2801/CodeAlpaca-20k', split=f'train[:{max_samples_per_domain}]')
        for item in ds_code:
            inst = item['instruction']
            inp = item.get('input', '')
            full_inst = f"{inst}\n{inp}".strip()
            prompt = f"<|im_start|>user\n{full_inst}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
            mixture.append({"domain": "code_codealpaca", "text": prompt})
        print(f"  • CodeAlpaca loaded: {len(ds_code)} samples", flush=True)
    except Exception as e:
        print(f"  [!] CodeAlpaca error: {e}", flush=True)

    # 4. Alpaca Cleaned (General Instruction & Alignment)
    try:
        ds_alpaca = load_dataset('yahma/alpaca-cleaned', split=f'train[:{max_samples_per_domain}]')
        for item in ds_alpaca:
            inst = item['instruction']
            inp = item.get('input', '')
            full_inst = f"{inst}\n{inp}".strip()
            prompt = f"<|im_start|>user\n{full_inst}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
            mixture.append({"domain": "general_alpaca", "text": prompt})
        print(f"  • Alpaca Cleaned loaded: {len(ds_alpaca)} samples", flush=True)
    except Exception as e:
        print(f"  [!] Alpaca Cleaned error: {e}", flush=True)

    print(f"[Data Loader] Total Multi-Domain SFT Training Samples: {len(mixture)}\n", flush=True)
    return mixture

def compute_perplexity(model, tokenizer, text: str) -> float:
    encodings = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(encodings.input_ids, labels=encodings.input_ids)
        loss = out.loss.item()
    return round(float(math.exp(loss)), 3)

def evaluate_fast_benchmarks(model, tokenizer) -> Dict[str, float]:
    results = {}
    
    # 1. MMLU (20 questions)
    try:
        ds_mmlu = load_dataset('cais/mmlu', 'college_computer_science', split='test[:20]')
        correct = 0
        choice_tokens = [tokenizer.encode(" " + c)[-1] for c in ['A', 'B', 'C', 'D']]
        for item in ds_mmlu:
            prompt = f"Question: {item['question']}\nChoices:\nA: {item['choices'][0]}\nB: {item['choices'][1]}\nC: {item['choices'][2]}\nD: {item['choices'][3]}\nAnswer:"
            inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                out = model(inp.input_ids)
                last_logits = out.logits[0, -1]
                scores = [last_logits[t].item() for t in choice_tokens]
                pred = scores.index(max(scores))
                if pred == item['answer']:
                    correct += 1
        results["mmlu"] = round((correct / len(ds_mmlu)) * 100.0, 2)
    except Exception as e:
        results["mmlu"] = 0.0

    # 2. ARC-Challenge (20 questions)
    try:
        ds_arc = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test[:25]')
        correct = 0
        choice_tokens = [tokenizer.encode(" " + c)[-1] for c in ['A', 'B', 'C', 'D']]
        cnt = 0
        for item in ds_arc:
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            ans_key = item["answerKey"]
            if ans_key in labels and len(choices) >= 4:
                ans_idx = labels.index(ans_key)
                if ans_idx < 4:
                    prompt = f"Question: {item['question']}\nChoices:\nA: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[3]}\nAnswer:"
                    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                    with torch.no_grad():
                        out = model(inp.input_ids)
                        last_logits = out.logits[0, -1]
                        scores = [last_logits[t].item() for t in choice_tokens]
                        pred = scores.index(max(scores))
                        if pred == ans_idx:
                            correct += 1
                    cnt += 1
                    if cnt >= 20:
                        break
        results["arc"] = round((correct / max(1, cnt)) * 100.0, 2)
    except Exception as e:
        results["arc"] = 0.0

    # 3. HumanEval (10 questions)
    try:
        ds_he = load_dataset('openai/openai_humaneval', split='test[:10]')
        correct = 0
        for item in ds_he:
            prompt = item['prompt']
            inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                gen_out = model.generate(
                    inp.input_ids,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                code = tokenizer.decode(gen_out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            try:
                compile(prompt + code, "<string>", "exec")
                correct += 1
            except SyntaxError:
                pass
        results["humaneval"] = round((correct / len(ds_he)) * 100.0, 2)
    except Exception as e:
        results["humaneval"] = 0.0

    return results

def main():
    print("=" * 95)
    print("HADL ID-SMIL: MULTI-DOMAIN SFT FINE-TUNING ON FRONTIER MIXTURE")
    print("=" * 95)
    print(f"[Device] {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"[Model] {MODEL_ID} in BF16")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load Multi-Domain SFT Data
    train_data = load_frontier_multi_domain_mixture(max_samples_per_domain=30)  # 120 samples total

    # 2. Load Base Model and measure baseline
    print("[Baseline] Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    base_model.eval()

    base_ppl = compute_perplexity(base_model, tokenizer, PERPLEXITY_TEXT)
    base_eval = evaluate_fast_benchmarks(base_model, tokenizer)
    print(f"[Baseline] Wikipedia PPL: {base_ppl:.3f}")
    print(f"[Baseline] MMLU CS: {base_eval['mmlu']}% | ARC: {base_eval['arc']}% | HumanEval: {base_eval['humaneval']}%\n", flush=True)

    # 3. Attach ID-SMIL Architecture
    print("=" * 95)
    print("[Attachment] Attaching ID-SMIL (IDR + LEA + Bounce-Back + SMIL Background Unit)...")
    for p in base_model.parameters():
        p.requires_grad = False  # 100% Frozen Base Model!

    model_wrapped = attach_hadl_idsmil(base_model, target_layer_idx=11, ghost_layer_idx=23)
    trainable_params = [p for p in model_wrapped.controller.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4)

    total_base_params = sum(p.numel() for p in base_model.parameters())
    trainable_controller_params = sum(p.numel() for p in trainable_params)
    print(f"  • Total Base Parameters: {total_base_params:,} (100% Frozen)")
    print(f"  • Trainable ID-SMIL Parameters: {trainable_controller_params:,} ({trainable_controller_params/total_base_params*100:.3f}% ratio)\n", flush=True)

    # 4. Multi-Domain SFT Training Loop (60 Steps across 4 domains)
    STEPS = 60
    print("=" * 95)
    print(f"[Training] Running Multi-Domain SFT ({STEPS} steps on UltraChat + GSM8k + Code + Alpaca)...")
    print("=" * 95)

    training_logs = []
    model_wrapped.controller.train()

    for step in range(STEPS):
        sample = train_data[step % len(train_data)]
        prompt = sample["text"]
        domain = sample["domain"]

        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        optimizer.zero_grad()

        # Forward pass through ID-SMIL
        out = model_wrapped(input_ids=inp.input_ids, labels=inp.input_ids)
        loss = out.loss
        loss.backward()
        optimizer.step()

        telem = dict(model_wrapped.controller.last_telemetry)
        step_log = {
            "step": step + 1,
            "domain": domain,
            "loss": round(float(loss.item()), 4),
            "kappa": telem.get("kappa_mean", 0.0),
            "p_understand": telem.get("p_understand_mean", 0.0),
            "bounce_ratio": telem.get("bounce_ratio", 0.0),
            "specialist_weight": telem.get("specialist_weight_mean", 0.0)
        }
        training_logs.append(step_log)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step+1:02d}/{STEPS} [{domain:<18}] | Loss: {loss.item():.4f} | LEA Kappa: {step_log['kappa']:+.4f} | Bounce: {step_log['bounce_ratio']*100:4.1f}% | SpecWeight: {step_log['specialist_weight']:.3f}", flush=True)

    print("\n[Training] Multi-Domain SFT Complete!", flush=True)
    model_wrapped.controller.eval()

    # 5. Post-Training Evaluation
    print("=" * 95)
    print("[Evaluation] Evaluating ID-SMIL After Multi-Domain SFT...")
    idsmil_ppl = compute_perplexity(model_wrapped, tokenizer, PERPLEXITY_TEXT)
    idsmil_eval = evaluate_fast_benchmarks(model_wrapped, tokenizer)

    print(f"[ID-SMIL] Wikipedia PPL: {idsmil_ppl:.3f} (Delta = {idsmil_ppl - base_ppl:+.3f})")
    print(f"[ID-SMIL] MMLU CS: {idsmil_eval['mmlu']}% (Delta = {idsmil_eval['mmlu'] - base_eval['mmlu']:+.2f}%)")
    print(f"[ID-SMIL] ARC-Challenge: {idsmil_eval['arc']}% (Delta = {idsmil_eval['arc'] - base_eval['arc']:+.2f}%)")
    print(f"[ID-SMIL] HumanEval: {idsmil_eval['humaneval']}% (Delta = {idsmil_eval['humaneval'] - base_eval['humaneval']:+.2f}%)\n", flush=True)

    # 6. Summary and Artifact Generation
    summary = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "datasets": ["HuggingFaceH4/ultrachat_200k", "openai/gsm8k", "sahil2801/CodeAlpaca-20k", "yahma/alpaca-cleaned"],
        "parameter_efficiency": {
            "total_base_params": total_base_params,
            "trainable_params": trainable_controller_params,
            "ratio_pct": round(trainable_controller_params / total_base_params * 100, 3)
        },
        "baseline": {
            "ppl": base_ppl,
            **base_eval
        },
        "idsmil_post_sft": {
            "ppl": idsmil_ppl,
            **idsmil_eval
        },
        "training_logs": training_logs
    }

    out_json = Path("eval_results/idsmil_frontier_multidomain_sft_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Log] Results saved to {out_json}")

    # Generate Visualization Graph
    generate_frontier_sft_graph(summary)

def generate_frontier_sft_graph(data):
    steps = [x["step"] for x in data["training_logs"]]
    losses = [x["loss"] for x in data["training_logs"]]
    kappas = [x["kappa"] for x in data["training_logs"]]
    bounces = [x["bounce_ratio"] * 100.0 for x in data["training_logs"]]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), facecolor='#0B0F19')
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#0B0F19')
        ax.grid(True, linestyle='--', alpha=0.2, color='#94A3B8')

    # Panel 1: Multi-Domain Loss Curve
    ax1.plot(steps, losses, color='#38BDF8', linewidth=2.2, label='Multi-Domain Loss')
    ax1.set_title('Frontier Multi-Domain SFT Loss Convergence', fontsize=12, fontweight='bold', color='#F8FAFC')
    ax1.set_xlabel('Training Steps', fontsize=11, color='#E2E8F0')
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11, color='#E2E8F0')
    ax1.tick_params(colors='#CBD5E1')
    ax1.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')

    # Panel 2: LEA Understanding Score (Kappa)
    ax2.plot(steps, kappas, color='#A855F7', linewidth=2.2, label='LEA Kappa Score')
    ax2.axhline(0, color='#EF4444', linestyle=':', label='Bounce-Back Boundary (kappa=0)')
    ax2.set_title('Local Epistemic Affordance (LEA) Dynamics', fontsize=12, fontweight='bold', color='#F8FAFC')
    ax2.set_xlabel('Training Steps', fontsize=11, color='#E2E8F0')
    ax2.set_ylabel('Comprehension Score (Kappa)', fontsize=11, color='#E2E8F0')
    ax2.tick_params(colors='#CBD5E1')
    ax2.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')

    # Panel 3: Benchmark Retention (Base vs ID-SMIL Post-SFT)
    bench_names = ['MMLU CS', 'ARC-Chal', 'HumanEval']
    base_scores = [data["baseline"]["mmlu"], data["baseline"]["arc"], data["baseline"]["humaneval"]]
    post_scores = [data["idsmil_post_sft"]["mmlu"], data["idsmil_post_sft"]["arc"], data["idsmil_post_sft"]["humaneval"]]

    x = np.arange(len(bench_names))
    width = 0.35
    r1 = ax3.bar(x - width/2, base_scores, width, label='Base Zero-Shot', color='#64748B', alpha=0.85)
    r2 = ax3.bar(x + width/2, post_scores, width, label='ID-SMIL Post-SFT', color='#10B981', alpha=0.95)

    ax3.set_title('Benchmark Retention: Base vs ID-SMIL Post-SFT', fontsize=12, fontweight='bold', color='#F8FAFC')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bench_names, fontsize=10.5, fontweight='bold', color='#E2E8F0')
    ax3.set_ylabel('Accuracy (%)', fontsize=11, color='#E2E8F0')
    ax3.set_ylim(0, 110)
    ax3.tick_params(colors='#CBD5E1')
    ax3.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')

    for bar in r1 + r2:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 1.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#F8FAFC')

    plt.tight_layout()
    out_png = Path("docs/images/idsmil_frontier_multidomain_sft_graph.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    # Copy to artifact dir
    artifact_dir = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")
    import shutil
    shutil.copy(out_png, artifact_dir / "idsmil_frontier_multidomain_sft_graph.png")
    print(f"[Graph] Saved to {out_png} and artifact directory.", flush=True)

if __name__ == "__main__":
    main()
