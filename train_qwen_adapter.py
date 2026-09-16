"""
Dual-Loop Cognitive Controller: Qwen PEFT Adapter Training Pipeline
===================================================================
Trains the LatentDeliberationAdapter on top of a frozen Qwen base model (Qwen2.5-0.5B,
Qwen2.5-2B, or custom HF checkpoints).

Highlights:
- Freezes 100% of base model weights; only trains the ~1-3% adapter weights.
- Targets complex reasoning tasks: Relational Multi-Hop Chains (AA-LCR) and
  Multi-Step Arithmetic Deduction (PolyMATH).
- Injects deliberated thoughts at the prompt/query transition anchor.
- Employs label masking (-100 on prompt tokens) for pure answer cross-entropy.
- Saves a lightweight adapter checkpoint (.pt) loadable by DualLoopQwenModel.
"""

import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
    get_cosine_schedule_with_warmup
)
from dual_loop import DualLoopQwenModel, attach_dual_loop_to_qwen
from dual_loop.benchmarks.benchmark_qwen_reasoning import (
    generate_relational_sample,
    generate_deduction_sample
)


class QwenReasoningDataset(Dataset):
    """
    Supervised Fine-Tuning dataset for relational and deduction reasoning.
    Prepares input_ids, attention_mask, and masked labels (-100 for prompt).
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = []

        for item in samples:
            prompt = item["prompt_text"]
            target = " " + item["target_text"].strip()
            
            prompt_enc = tokenizer(prompt, add_special_tokens=False)
            target_enc = tokenizer(target, add_special_tokens=False)
            
            p_ids = prompt_enc["input_ids"]
            t_ids = target_enc["input_ids"]
            
            input_ids = p_ids + t_ids
            # Query anchor index: last token of prompt (before answer begins)
            query_anchor_pos = len(p_ids) - 1
            
            # Mask prompt tokens with -100 so loss is computed ONLY on the target answer
            labels = [-100] * len(p_ids) + t_ids
            
            # Truncate if exceeding max_length
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
                
            self.features.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "query_anchor_pos": query_anchor_pos
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pads variable-length sequences dynamically."""
    max_len = max(len(item["input_ids"]) for item in batch)
    
    padded_inputs = []
    padded_labels = []
    padded_masks = []
    anchors = []
    
    for item in batch:
        inp = item["input_ids"]
        lbl = item["labels"]
        seq_len = len(inp)
        pad_len = max_len - seq_len
        
        # Right-pad inputs and masks
        padded_inp = torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        mask = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        
        padded_inputs.append(padded_inp)
        padded_labels.append(padded_lbl)
        padded_masks.append(mask)
        anchors.append(item["query_anchor_pos"])
        
    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(padded_masks),
        "query_anchor_pos": anchors
    }


def create_synthetic_data(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Creates a balanced mixture of AA-LCR relational chains and PolyMATH deductions."""
    samples = []
    half = num_samples // 2
    for i in range(half):
        samples.append(generate_relational_sample(hops=random.choice([2, 3]), seed=seed + i))
    for i in range(half, num_samples):
        samples.append(generate_deduction_sample(steps=random.choice([2, 3]), seed=seed + i))
    random.Random(seed).shuffle(samples)
    return samples


def train(args):
    print("\n" + "=" * 70)
    print(" DUAL-LOOP QWEN ADAPTER TRAINING (PEFT)")
    print("=" * 70)
    print(f"Base Model:     {args.model}")
    print(f"Device:         {args.device}")
    print(f"Ponder Steps:   K={args.k_steps}")
    print(f"Epochs:         {args.epochs}")
    print(f"Learning Rate:  {args.lr}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Output Checkpt: {args.save_path}")
    print("-" * 70)

    # 1. Load Model & Tokenizer
    if args.use_mock:
        print("[Setup] Using lightweight mock Qwen model for rapid testing...")
        cfg = Qwen2Config(
            vocab_size=5000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            pad_token_id=0
        )
        base_model = Qwen2ForCausalLM(cfg).to(args.device)
        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 1
            def __call__(self, text, add_special_tokens=False):
                tokens = [abs(hash(w)) % 4000 + 2 for w in text.split()]
                return {"input_ids": tokens or [2]}
        tokenizer = MockTokenizer()
    else:
        print(f"[Setup] Loading {args.model} tokenizer and causal LM...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16,
            device_map=args.device if args.device == "cuda" else None
        )
        if args.device == "cpu":
            base_model = base_model.to("cpu")

    # 2. Attach Dual-Loop Adapter
    model = attach_dual_loop_to_qwen(base_model, k_steps=args.k_steps, adapter_mode="residual")
    
    # 3. Freeze base weights (PEFT)
    model.freeze_backbone()
    summary = model.get_parameter_summary()
    print(f"[PEFT] Base weights frozen!")
    print(f"[PEFT] Total parameters:     {summary['total_parameters']:,}")
    print(f"[PEFT] Trainable parameters: {summary['trainable_parameters']:,} ({summary['trainable_ratio_pct']}%)")
    print("-" * 70)

    # 4. Prepare Dataset & DataLoader
    print("[Data] Generating synthetic training and validation corpora...")
    train_samples = create_synthetic_data(num_samples=args.train_samples, seed=42)
    val_samples = create_synthetic_data(num_samples=args.val_samples, seed=9999)

    pad_id = tokenizer.pad_token_id or 0
    train_dataset = QwenReasoningDataset(train_samples, tokenizer)
    val_dataset = QwenReasoningDataset(val_samples, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id)
    )

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(2, int(total_steps * 0.1)),
        num_training_steps=total_steps
    )

    # 6. Training Loop
    best_val_loss = float("inf")
    start_train_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        step_count = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            
            # Target the last prompt token position for deliberation per-sample (ARCH-03)
            anchors = batch["query_anchor_pos"]
            model.query_idx = torch.tensor(anchors, dtype=torch.long, device=input_ids.device) if len(anchors) > 0 else -1

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            step_count += 1

        avg_train_loss = train_loss / step_count if step_count > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(args.device)
                labels = batch["labels"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                anchors = batch["query_anchor_pos"]
                model.query_idx = torch.tensor(anchors, dtype=torch.long, device=input_ids.device) if len(anchors) > 0 else -1

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        lr_current = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {lr_current:.2e}")

        # Save best adapter
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_adapter(args.save_path)
            print(f"  --> Checkpoint saved: {args.save_path} (Val Loss: {avg_val_loss:.4f})")

    elapsed_time = time.perf_counter() - start_train_time
    print("-" * 70)
    print(f"[Done] Training completed in {elapsed_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")
    print(f"[Done] Final Adapter Checkpoint: {os.path.abspath(args.save_path)}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Loop Adapter on Qwen Models")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-2B-Instruct",
                        help="HuggingFace model ID or path (e.g. Qwen/Qwen2.5-2B-Instruct, Qwen/Qwen3.5-2B)")
    parser.add_argument("--k_steps", type=int, default=2, help="Number of latent deliberation steps")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for adapter")
    parser.add_argument("--train_samples", type=int, default=80, help="Number of synthetic training samples")
    parser.add_argument("--val_samples", type=int, default=20, help="Number of validation samples")
    parser.add_argument("--save_path", type=str, default="dual_loop/checkpoints/qwen_dualloop_adapter.pt",
                        help="Save path for trained adapter weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_mock", action="store_true", help="Use mock model for quick sanity checking")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
