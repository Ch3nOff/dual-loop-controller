"""
Dual-Loop Cognitive Controller: GLM-4-9B PEFT Adapter Training Pipeline
========================================================================
Supports:
  - Model: zai-org/glm-4-9b-chat / THUDM/glm-4-9b-chat (D=4096, 40 layers)
  - Target Hook: Layer 20 (midpoint latent residual stream)
  - Memory Modes:
      * 16-bit Full Precision (Requires >= 24GB VRAM: A10G, RTX 3090/4090, A100)
      * 4-bit Quantization (bitsandbytes NF4, Fits in ~8-12GB VRAM)
  - Freezes 100% of GLM-4-9B backbone weights; only trains the ~6% adapter parameters.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_cosine_schedule_with_warmup
)
from dual_loop import attach_dual_loop

class GLMReasoningDataset(Dataset):
    """Curated dataset for relational, deductive, and multi-hop reasoning."""
    def __init__(self, samples: List[Dict[str, Any]], tokenizer: Any, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = []

        for item in samples:
            prompt = item["prompt_text"]
            target = " " + item["target_text"].strip()
            
            p_enc = tokenizer(prompt, add_special_tokens=False)
            t_enc = tokenizer(target, add_special_tokens=False)
            
            p_ids = p_enc["input_ids"]
            t_ids = t_enc["input_ids"]
            
            input_ids = p_ids + t_ids
            labels = [-100] * len(p_ids) + t_ids
            
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                
            query_pos = len(p_ids) - 1
            
            self.features.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "query_idx": min(query_pos, len(input_ids) - 1)
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def collate_fn(batch):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    b_size = len(batch)
    
    padded_inputs = torch.zeros(b_size, max_len, dtype=torch.long)
    padded_labels = torch.full((b_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(b_size, max_len, dtype=torch.long)
    query_indices = []

    for i, item in enumerate(batch):
        l = item["input_ids"].shape[0]
        padded_inputs[i, :l] = item["input_ids"]
        padded_labels[i, :l] = item["labels"]
        attention_mask[i, :l] = 1
        query_indices.append(item["query_idx"])

    return {
        "input_ids": padded_inputs,
        "attention_mask": attention_mask,
        "labels": padded_labels,
        "query_indices": query_indices
    }

def get_demo_samples():
    return [
        {
            "prompt_text": "Question: In inverted physics, denser objects float on top of lighter liquids. A lead cube (density 11.3 g/cm3) is dropped into liquid water (density 1.0 g/cm3). Does the lead float or sink?\nAnswer:",
            "target_text": "The lead floats."
        },
        {
            "prompt_text": "Question: All flurbs are snarks. No snarks are vorpal. Is a flurb vorpal?\nAnswer:",
            "target_text": "No, a flurb cannot be vorpal."
        },
        {
            "prompt_text": "Question: Alice is older than Bob. Charlie is younger than Bob. Who is the oldest?\nAnswer:",
            "target_text": "Alice is the oldest."
        },
        {
            "prompt_text": "Question: Evaluate the nested expression: True AND (NOT (False OR True)).\nAnswer:",
            "target_text": "False."
        }
    ] * 20  # 80 training samples

def train_glm4():
    parser = argparse.ArgumentParser(description="Train Dual-Loop Controller on GLM-4-9B")
    parser.add_argument("--model_id", type=str, default="zai-org/glm-4-9b-chat")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use bitsandbytes 4-bit quantization")
    parser.add_argument("--layer_idx", type=int, default=20, help="Hook layer (midpoint for 40-layer GLM-4)")
    parser.add_argument("--bottleneck_dim", type=int, default=1024, help="Bottleneck latent dimension (e.g. 1024 for 8GB VRAM, 0 to disable)")
    parser.add_argument("--cpu_offload", action="store_true", default=True, help="Force CPU offload/execution (recommended for constrained VRAM)")
    parser.add_argument("--from_config", action="store_true", help="Instantiate native GLM-4 architecture from config (skips 18GB download for instant testing)")
    parser.add_argument("--num_layers", type=int, default=0, help="Override layer count for rapid prototyping (e.g. 4 or 8)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="checkpoints/glm4_adapter")
    args = parser.parse_args()

    print("=" * 80)
    print(f"DUAL-LOOP COGNITIVE CONTROLLER: TRAINING PIPELINE FOR {args.model_id}")
    print("=" * 80)

    device = "cpu" if args.cpu_offload or not torch.cuda.is_available() else "cuda"
    print(f"[*] Compute Device: {device} (CPU Offload: {args.cpu_offload})")

    # Quantization Config (only if on CUDA with bitsandbytes)
    quant_kwargs = {}
    if args.load_in_4bit and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            print("[+] Enabled BitsAndBytes 4-bit NF4 Quantization (Low-VRAM mode)")
        except ImportError:
            print("[!] Warning: bitsandbytes not installed. Loading in standard precision.")

    print(f"[*] Loading Tokenizer & Config: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    # Patch modern transformers config compatibility
    if not hasattr(config, "max_length"):
        config.max_length = getattr(config, "seq_length", 8192)
    if not hasattr(config, "use_cache"):
        config.use_cache = False

    if args.num_layers > 0:
        config.num_layers = args.num_layers
        if args.layer_idx >= args.num_layers:
            args.layer_idx = args.num_layers // 2

    if args.from_config:
        print(f"[*] Instantiating {config.model_type} architecture directly from config ({config.num_layers} layers, D={config.hidden_size})...")
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            empty_init=False
        ).to(dtype=torch.float32 if device == "cpu" else torch.bfloat16)
    else:
        print(f"[*] Loading Pretrained Model Weights for {args.model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else "cpu",
            **quant_kwargs
        )

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
    print("[+] Successfully froze 100% of base GLM-4 weights.")

    # Attach Dual-Loop Cognitive Controller at Layer 20
    b_dim = args.bottleneck_dim if args.bottleneck_dim > 0 else None
    if b_dim is not None:
        print(f"[*] Attaching Bottleneck Dual-Loop Adapter at Layer {args.layer_idx} (D=4096 -> d={b_dim} -> D=4096)...")
    else:
        print(f"[*] Attaching Full-Width Dual-Loop Adapter at Layer {args.layer_idx} (D=4096)...")

    wrapped_model = attach_dual_loop(
        model,
        layer_idx=args.layer_idx,
        k_steps=2,
        bottleneck_dim=b_dim,
        enable_plasticity=True,
        use_evidential_gate=True
    )

    summary = wrapped_model.get_parameter_summary()
    adapter_params = [p for p in wrapped_model.adapter.parameters() if p.requires_grad]
    total_adapter_p = sum(p.numel() for p in adapter_params)
    print(f"[+] Trainable Adapter Parameters: {total_adapter_p:,} ({total_adapter_p/1e6:.2f}M)")
    if b_dim is not None:
        p_full = 568_009_125
        saved_pct = (1.0 - total_adapter_p / p_full) * 100.0
        print(f"[+] Bottleneck Reduction: -{saved_pct:.1f}% parameters vs full-width ({p_full/1e6:.1f}M -> {total_adapter_p/1e6:.2f}M)")
        print(f"[+] Memory Profile: ~{total_adapter_p*8/(1024**2):.1f}MB AdamW state (fits comfortably in 8GB VRAM alongside 4-bit base model)")
    print(f"[+] Trainable Parameter Ratio: {summary['trainable_ratio_pct']}% of total model")

    # DataLoader
    dataset = GLMReasoningDataset(get_demo_samples(), tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(adapter_params, lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=total_steps)

    print(f"\n[*] Starting Training: {args.epochs} Epochs, {len(dataset)} Samples...")
    wrapped_model.adapter.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        t_start = time.time()
        for step, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            query_indices = batch["query_indices"]

            wrapped_model.set_query_index(query_indices[0])
            wrapped_model.set_ponder_steps(2)
            optimizer.zero_grad()

            outputs = wrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 10 == 0 or step == len(loader):
                print(f"  Epoch {epoch}/{args.epochs} | Step {step}/{len(loader)} | Loss: {loss.item():.4f}")

        elapsed = time.time() - t_start
        print(f"[+] Epoch {epoch} Complete | Avg Loss: {total_loss/len(loader):.4f} | Time: {elapsed:.2f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "glm4_adapter.safetensors")
    from safetensors.torch import save_file
    state_dict = {f"adapter.{k}": v.cpu() for k, v in wrapped_model.adapter.state_dict().items()}
    save_file(state_dict, save_path)
    print(f"\n[+] Training Complete! Adapter weights saved to: {save_path}")

if __name__ == "__main__":
    train_glm4()
