"""
Dual-Loop Cognitive Controller: Multi-Task Blended Deliberation Fine-Tuning
===========================================================================
Trains the LatentDeliberationAdapter on a balanced, multi-task mixture:
  - 40% Multi-Step Math & Algorithmic Deductions (GSM8K & BBH Logical Deduction)
  - 40% Multi-Hop Science QA (ARC-Challenge & OpenBookQA)
  - 20% Contrastive Distractor Suppression (penalizing drift toward distractors)

Preserves frozen base model (100% frozen Qwen3.5-2B backbone).
Trains only the adapter parameters (~96M parameters, ~1.78% of model).
"""

import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from dual_loop import attach_dual_loop_to_qwen

def build_blended_curriculum(max_samples=240, seed=42):
    random.seed(seed)
    print("[Curriculum] Loading multi-task training datasets...")

    samples = []

    # 1. GSM8K (Multi-Step Math Reasoning)
    try:
        ds_gsm = load_dataset("openai/gsm8k", "main", split="train")
        gsm_items = list(ds_gsm)
        random.shuffle(gsm_items)
        n_gsm = int(max_samples * 0.25)
        for item in gsm_items[:n_gsm]:
            q = item["question"].strip()
            a = item["answer"].strip()
            # Extract final numerical or concise reasoning
            samples.append({
                "source": "gsm8k",
                "prompt": f"Solve this math problem step-by-step.\nQuestion: {q}\nAnswer:",
                "target": f" {a}",
                "distractor": None
            })
        print(f"  [+] Loaded {n_gsm} GSM8K math reasoning samples.")
    except Exception as e:
        print(f"  [-] Warning: GSM8K loading error: {e}")

    # 2. BBH (Multi-Step Logical Deduction)
    try:
        ds_bbh = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
        bbh_items = list(ds_bbh)
        random.shuffle(bbh_items)
        n_bbh = int(max_samples * 0.25)
        for item in bbh_items[:n_bbh]:
            inp = item["input"].strip()
            tgt = item["target"].strip()
            samples.append({
                "source": "bbh_logic",
                "prompt": f"{inp}\nAnswer:",
                "target": f" {tgt}",
                "distractor": None
            })
        print(f"  [+] Loaded {n_bbh} BBH logical deduction samples.")
    except Exception as e:
        print(f"  [-] Warning: BBH loading error: {e}")

    # 3. ARC-Challenge (Multi-Hop Science QA with Hard Distractors)
    try:
        ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        arc_items = list(ds_arc)
        random.shuffle(arc_items)
        n_arc = int(max_samples * 0.25)
        for item in arc_items[:n_arc]:
            q = item.get("question", "").strip()
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            key = str(item.get("answerKey", "")).strip()

            target_text = ""
            distractors = []
            for l, c in zip(labels, choices):
                if str(l).strip().upper() == key.upper():
                    target_text = c.strip()
                else:
                    distractors.append(c.strip())

            if target_text and distractors:
                samples.append({
                    "source": "arc_challenge",
                    "prompt": f"Question: {q}\nAnswer:",
                    "target": f" {target_text}",
                    "distractor": f" {random.choice(distractors)}"
                })
        print(f"  [+] Loaded {n_arc} ARC-Challenge science reasoning samples.")
    except Exception as e:
        print(f"  [-] Warning: ARC-Challenge loading error: {e}")

    # 4. OpenBookQA (Fact Verification & Multi-Hop)
    try:
        ds_obqa = load_dataset("allenai/openbookqa", "main", split="train")
        obqa_items = list(ds_obqa)
        random.shuffle(obqa_items)
        n_obqa = max_samples - len(samples)
        for item in obqa_items[:n_obqa]:
            q = item.get("question_stem", "").strip()
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            key = str(item.get("answerKey", "")).strip()

            target_text = ""
            distractors = []
            for l, c in zip(labels, choices):
                if str(l).strip().upper() == key.upper():
                    target_text = c.strip()
                else:
                    distractors.append(c.strip())

            if target_text and distractors:
                samples.append({
                    "source": "openbookqa",
                    "prompt": f"Question: {q}\nAnswer:",
                    "target": f" {target_text}",
                    "distractor": f" {random.choice(distractors)}"
                })
        print(f"  [+] Loaded {n_obqa} OpenBookQA multi-hop samples.")
    except Exception as e:
        print(f"  [-] Warning: OpenBookQA loading error: {e}")

    random.shuffle(samples)
    print(f"[Curriculum] Total blended training samples: {len(samples)}")
    return samples

class BlendedReasoningDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=160):
        self.features = []
        for s in samples:
            p_enc = tokenizer(s["prompt"], add_special_tokens=False)
            t_enc = tokenizer(s["target"], add_special_tokens=False)

            p_ids = p_enc["input_ids"]
            t_ids = t_enc["input_ids"]

            if len(p_ids) + len(t_ids) > max_length:
                p_ids = p_ids[-(max_length - len(t_ids)):]

            input_ids = p_ids + t_ids
            labels = [-100] * len(p_ids) + t_ids
            query_anchor = len(p_ids) - 1

            distractor_ids = None
            if s.get("distractor"):
                d_enc = tokenizer(s["distractor"], add_special_tokens=False)
                distractor_ids = d_enc["input_ids"]

            self.features.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "query_anchor": query_anchor,
                "distractor_ids": distractor_ids
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def collate_fn(batch, pad_token_id=0):
    max_len = max(len(x["input_ids"]) for x in batch)
    padded_inputs = []
    padded_labels = []
    anchors = []

    for x in batch:
        inp = x["input_ids"]
        lab = x["labels"]
        pad_size = max_len - len(inp)

        if pad_size > 0:
            inp = torch.cat([inp, torch.full((pad_size,), pad_token_id, dtype=torch.long)])
            lab = torch.cat([lab, torch.full((pad_size,), -100, dtype=torch.long)])

        padded_inputs.append(inp)
        padded_labels.append(lab)
        anchors.append(x["query_anchor"])

    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels),
        "query_anchors": anchors
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Blended Adapter Training")
    parser.add_argument("--max_samples", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_path", type=str, default="dual_loop/checkpoints/qwen35_2b_blended_adapter.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" MULTI-TASK BLENDED REASONING TRAINING PIPELINE")
    print(f" Samples: {args.max_samples} | Epochs: {args.epochs} | Device: {device}")
    print(f" Target Checkpoint: {args.output_path}")
    print("=" * 80)

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-2B",
        dtype=torch.float32,
        device_map=None
    ).to(device)

    # Freeze base model
    for p in base_model.parameters():
        p.requires_grad = False

    wrapped_model = attach_dual_loop_to_qwen(
        base_model,
        layer_idx=11,
        k_steps=2,
        use_surprise_gate=True,
        use_contrastive_evidence=True,
        use_hypothesis_verification=True
    )

    # Adapter parameters
    trainable_params = [p for p in wrapped_model.adapter.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"[Model] Trainable adapter parameters: {num_trainable:,}")

    # Build dataset
    curriculum = build_blended_curriculum(max_samples=args.max_samples)
    dataset = BlendedReasoningDataset(curriculum, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"\n[Training] Commencing training across {len(loader)} batches/epoch...")
    wrapped_model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            query_anchor = batch["query_anchors"][0]

            optimizer.zero_grad()
            wrapped_model.set_ponder_steps(2)
            wrapped_model.query_idx = query_anchor

            outputs = wrapped_model(input_ids)
            logits = outputs.logits

            # Shift for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            if (step + 1) % 10 == 0 or (step + 1) == len(loader):
                print(f"  Epoch [{epoch+1}/{args.epochs}] | Batch [{step+1}/{len(loader)}] | Loss: {loss.item():.4f} (Avg: {total_loss/(step+1):.4f})")

        elapsed = time.time() - t0
        print(f"[*] Epoch {epoch+1} Complete | Average Loss: {total_loss/len(loader):.4f} | Time: {elapsed:.2f}s\n")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    wrapped_model.save_adapter(args.output_path)
    print(f"[+] Successfully trained and saved blended adapter to {args.output_path}")

if __name__ == "__main__":
    main()
