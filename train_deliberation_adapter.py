"""
Dual-Loop Cognitive Controller: PEFT Deliberation Fine-Tuning Pipeline
=====================================================================
Trains the Dual-Loop LatentDeliberationAdapter (attached to Layer 11 Full-Attention)
on scientific reasoning training splits to learn deliberative error-correction.
- Freezes 100% of Qwen3.5-2B base model (2.37B params).
- Trains ONLY the 96M adapter parameters.
- Targets multiple-choice scientific deduction.
"""

import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from dual_loop import attach_dual_loop_to_qwen

def build_training_samples(max_samples=120, seed=42):
    random.seed(seed)
    print(f"[Data] Loading multi-hop science reasoning training splits from allenai/ai2_arc & allenai/openbookqa...")
    ds_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    ds_chal = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    ds_obqa = load_dataset("allenai/openbookqa", "main", split="train")
    
    # Mix ARC and OpenBookQA equally
    arc_items = list(ds_easy) + list(ds_chal)
    obqa_items = list(ds_obqa)
    random.shuffle(arc_items)
    random.shuffle(obqa_items)
    
    n_arc = max_samples // 2
    n_obqa = max_samples - n_arc
    selected = arc_items[:n_arc] + obqa_items[:n_obqa]
    random.shuffle(selected)
    
    formatted = []
    for item in selected:
        q = item.get("question", "")
        if not q and "question_stem" in item:
            q = item["question_stem"]
        q = str(q).strip()
        
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        key = str(item.get("answerKey", "")).strip()
        
        target_text = None
        for l, c in zip(labels, choices):
            if l.strip().upper() == key.upper() or (key.isdigit() and str(l).strip() == key):
                target_text = c.strip()
                break
        if target_text is None and len(choices) > 0:
            target_text = choices[0].strip()
            
        formatted.append({
            "prompt": f"Question: {q}\nAnswer:",
            "target": f" {target_text}"
        })
        
    print(f"[Data] Successfully built {len(formatted)} balanced multi-hop reasoning training samples ({n_arc} ARC + {n_obqa} OpenBookQA).")
    return formatted

class ReasoningSFTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.features = []
        for s in samples:
            prompt_enc = tokenizer(s["prompt"], add_special_tokens=False)
            target_enc = tokenizer(s["target"], add_special_tokens=False)
            
            p_ids = prompt_enc["input_ids"]
            t_ids = target_enc["input_ids"]
            
            if len(p_ids) + len(t_ids) > max_length:
                p_ids = p_ids[-(max_length - len(t_ids)):]
                
            input_ids = p_ids + t_ids
            # Query anchor is the last token of the prompt (where question ends and answer begins)
            query_anchor_pos = len(p_ids) - 1
            labels = [-100] * len(p_ids) + t_ids
            
            self.features.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "query_anchor_pos": query_anchor_pos
            })
            
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx]

def collate_fn(batch, pad_token_id=0):
    max_len = max(len(x["input_ids"]) for x in batch)
    padded_inputs = []
    padded_labels = []
    padded_masks = []
    anchors = []
    
    for item in batch:
        inp = item["input_ids"]
        lbl = item["labels"]
        pad_len = max_len - len(inp)
        
        padded_inputs.append(torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)]))
        padded_masks.append(torch.cat([torch.ones(len(inp), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
        anchors.append(item["query_anchor_pos"])
        
    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(padded_masks),
        "query_anchor_pos": anchors
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--init_adapter", type=str, default="dual_loop/checkpoints/qwen35_2b_deliberation_adapter.safetensors")
    parser.add_argument("--save_path", type=str, default="dual_loop/checkpoints/qwen35_2b_deliberation_adapter.pt")
    parser.add_argument("--layer_idx", type=int, default=11)
    parser.add_argument("--k_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--trust_remote_code", action="store_true", default=False, help="Allow executing remote code from Hugging Face Hub")
    parser.add_argument("--revision", type=str, default="15852e8c16360a2fea060d615a32b45270f8a8fc", help="Pinned commit SHA for supply-chain security (SEC-02)")
    args = parser.parse_args()

    print("=" * 80)
    print(" DUAL-LOOP QWEN3.5-2B: PEFT DELIBERATION FINE-TUNING")
    print("=" * 80)
    print(f"Base Model:    {args.model} (FROZEN 100%)")
    print(f"Revision:      {args.revision}")
    print(f"Interception:  Layer {args.layer_idx} (Full Attention)")
    print(f"Deliberation:  K={args.k_steps} recurrent steps")
    print(f"Samples:       {args.max_samples} from ARC + OpenBookQA train splits")
    print(f"Save Path:     {args.save_path}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("[Model] Loading Qwen3.5-2B into CPU memory...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        device_map="cpu"
    )

    print("[Model] Attaching Dual-Loop Adapter...")
    model = attach_dual_loop_to_qwen(base_model, layer_idx=args.layer_idx, k_steps=args.k_steps)
    if os.path.exists(args.init_adapter):
        print(f"[Model] Initializing adapter weights from {args.init_adapter}...")
        model.load_adapter(args.init_adapter)

    # Freeze base model weights (PEFT)
    model.freeze_backbone()
    summary = model.get_parameter_summary()
    print(f"[PEFT] Total parameters:     {summary['total_parameters']:,}")
    print(f"[PEFT] Trainable parameters: {summary['trainable_parameters']:,} ({summary['trainable_ratio_pct']}%)")

    train_samples = build_training_samples(max_samples=args.max_samples)
    dataset = ReasoningSFTDataset(train_samples, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )

    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=0.01)
    
    print("\n[Training] Starting Deliberation Tuning loop...")
    t0_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            anchors = batch["query_anchor_pos"]
            
            # Anchor deliberation per-sample onto each sequence's question boundary (ARCH-03)
            model.query_idx = torch.tensor(anchors, dtype=torch.long, device=input_ids.device) if len(anchors) > 0 else -1
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            if (step + 1) % 10 == 0 or (step + 1) == len(loader):
                print(f"  Epoch {epoch}/{args.epochs} | Step {step+1:02d}/{len(loader)} | Loss: {loss.item():.4f} | Gate Scale: {float(torch.tanh(model.adapter.gate_alpha).item()):.4f}")
                sys.stdout.flush()

        avg_loss = total_loss / batches if batches > 0 else 0.0
        print(f"[Epoch {epoch} Complete] Average Loss: {avg_loss:.4f} in {time.time() - t0_train:.1f}s")
        sys.stdout.flush()

    # Save trained checkpoint (both pt and safetensors)
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    saved_file = model.save_adapter(args.save_path, format="pt")
    safetensors_path = os.path.splitext(args.save_path)[0] + ".safetensors"
    model.save_adapter(safetensors_path, format="safetensors")
    print(f"\n[+] Fine-Tuning Complete! Trained adapter saved to: {saved_file} and {safetensors_path}")
    print(f"    Final Learned ReZero Gate: {float(torch.tanh(model.adapter.gate_alpha).item()):.4f}")

if __name__ == "__main__":
    main()
