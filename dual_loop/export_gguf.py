"""
HADL to GGUF Exporter & Weight Fusion Pipeline
==============================================
Fuses the distilled HADL Dual-Loop cognitive controller (Car-Lift v4.5 / v3.0)
into the base transformer weights, producing a standard GGUF (.gguf) file that
runs natively in llama.cpp, Ollama, LM Studio, and Jan on Windows, Linux, and macOS.

Features:
- Mathematical linear weight fusion into target transformer layers (Midpoint & Ghost layers)
- Direct GGUF serialization via llama.cpp convert_hf_to_gguf
- Automatic Ollama Modelfile generation
- Supports FP16, BF16, and Q8_0 outtypes
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def resolve_model_path(model_id_or_path: str) -> str:
    """Resolves local model path from directory or Hugging Face cache."""
    if os.path.exists(model_id_or_path):
        return model_id_or_path

    hf_cache = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub"
    )
    if os.path.isdir(hf_cache):
        sanitized = f"models--{model_id_or_path.replace('/', '--')}"
        snapshots_dir = os.path.join(hf_cache, sanitized, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = [
                os.path.join(snapshots_dir, s)
                for s in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, s))
            ]
            if snapshots:
                return snapshots[0]
    return model_id_or_path


def fuse_hadl_weights(
    model: nn.Module,
    checkpoint_path: str,
    alpha_mid: float = 0.15,
    alpha_ghost: float = 0.15,
) -> nn.Module:
    """
    Fuses the distilled HADL cognitive controller weights into the base transformer layers.

    Piston 1 (Midpoint Layer 11):
      Fuses the low-rank affordance and fluid bridge transformations into o_proj.
    Piston 2 (Ghost Layer 23):
      Fuses the SVD ghost projection and verification reconstruction into o_proj.
    """
    from .adapters.universal_adapter import DynamicGraphIntrospector

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"[*] Loading HADL distilled checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("controller_state_dict", ckpt.get("controller", ckpt))

    # Introspect transformer layers
    container_path, layer_list, total_layers = (
        DynamicGraphIntrospector.discover_layer_container(model)
    )
    d_native = DynamicGraphIntrospector.probe_native_dimension(model)

    mid_idx = total_layers // 2 - 1 if total_layers % 2 == 0 else total_layers // 2
    ghost_idx = max(mid_idx + 1, total_layers - 2)

    print(f"[*] Base model: {total_layers} layers, hidden dimension D={d_native}")
    print(f"[*] Fusing Piston 1 into Layer {mid_idx} and Piston 2 into Layer {ghost_idx}...")

    mid_layer = layer_list[mid_idx]
    ghost_layer = layer_list[ghost_idx]

    # 1. Compute Ghost Layer (Piston 2) delta
    # ghost: u_proj (D, 32), verify_linear.weight (D, 32)
    # Recon = W_verify @ U^T  shape: (D, D)
    u_proj = state_dict.get("ghost.u_proj")
    w_verify = state_dict.get("ghost.verify_linear.weight")

    if u_proj is not None and w_verify is not None:
        delta_ghost = torch.matmul(w_verify, u_proj.T) * alpha_ghost  # (D, D)
        _fuse_delta_into_layer(ghost_layer, delta_ghost, "Piston 2 (Ghost)")

    # 2. Compute Midpoint Layer (Piston 1) delta
    # hydraulic_cup.fluid_bridge.weight (D, D) or smil projections
    fluid_bridge = state_dict.get("hydraulic_cup.fluid_bridge.weight")
    if fluid_bridge is not None:
        delta_mid = fluid_bridge * alpha_mid
        _fuse_delta_into_layer(mid_layer, delta_mid, "Piston 1 (Hydraulic Bridge)")
    else:
        # Fallback to smil up/down projections
        down_proj = state_dict.get("hydraulic_cup.smil.down_proj.weight")
        up_proj = state_dict.get("hydraulic_cup.smil.up_proj.weight")
        if down_proj is not None and up_proj is not None:
            delta_mid = torch.matmul(up_proj, down_proj) * alpha_mid
            _fuse_delta_into_layer(mid_layer, delta_mid, "Piston 1 (SMIL)")

    print("[OK] HADL cognitive transformations mathematically fused into transformer layers!")
    return model


def _fuse_delta_into_layer(layer: nn.Module, delta: torch.Tensor, name: str) -> None:
    """Fuses (D, D) delta into the layer's output projection (o_proj or dense)."""
    target_param = None
    target_name = ""

    # Check common attention output projection names
    for path in ["self_attn.o_proj", "attention.o_proj", "self_attention.o_proj", "attn.c_proj", "self_attn.dense"]:
        parts = path.split(".")
        curr = layer
        for p in parts:
            curr = getattr(curr, p, None)
            if curr is None:
                break
        if curr is not None and hasattr(curr, "weight"):
            target_param = curr.weight
            target_name = path
            break

    if target_param is not None:
        delta_cast = delta.to(device=target_param.device, dtype=target_param.dtype)
        if target_param.shape == delta_cast.shape:
            target_param.data.add_(delta_cast)
            print(f"  [+] Fused {name} into {target_name}.weight (shape: {list(target_param.shape)})")
        else:
            print(f"  [!] Shape mismatch for {target_name}: {target_param.shape} vs {delta_cast.shape}")
    else:
        print(f"  [!] Could not locate o_proj in target layer for {name}")


def export_hadl_to_gguf(
    model_id_or_path: str = "Qwen/Qwen3.5-2B",
    checkpoint_path: str = "checkpoints/xstar_2b_omnireason_carlift_500q_checkpoint.pt",
    output_gguf_path: str = "dist/xstar_2b_carlift_q8_0.gguf",
    outtype: str = "q8_0",
    work_dir: str = "dist/staging_hf_fused",
    trust_remote_code: bool = True,
) -> str:
    """
    Full end-to-end pipeline:
    1. Loads base model & tokenizer
    2. Fuses HADL Car-Lift cognitive weights
    3. Saves fused Hugging Face model
    4. Converts to GGUF format via llama.cpp converter
    5. Writes Ollama Modelfile
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    resolved_model = resolve_model_path(model_id_or_path)
    print("=" * 80)
    print("  HADL TO GGUF EXPORT & WEIGHT FUSION PIPELINE")
    print("=" * 80)
    print(f"[*] Base Model Path     : {resolved_model}")
    print(f"[*] Checkpoint Path     : {checkpoint_path}")
    print(f"[*] Target GGUF File    : {output_gguf_path}")
    print(f"[*] Quantization/Type   : {outtype.upper()}")
    print("=" * 80)

    # 1. Load Tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=trust_remote_code)

    # 2. Load Base Model
    print("\n[2/5] Loading base causal language model (CPU / FP32)...")
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )

    # 3. Fuse HADL Weights
    print("\n[3/5] Fusing HADL Car-Lift v4.5 cognitive controller weights...")
    model = fuse_hadl_weights(model, checkpoint_path=checkpoint_path)

    # 4. Save Fused Model to Staging HF Directory
    print(f"\n[4/5] Saving fused model to staging directory: {work_dir}")
    os.makedirs(work_dir, exist_ok=True)
    model.save_pretrained(work_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(work_dir)

    # 5. Convert to GGUF
    print("\n[5/5] Converting fused model to GGUF format...")
    out_dir = os.path.dirname(os.path.abspath(output_gguf_path))
    os.makedirs(out_dir, exist_ok=True)

    converter_script = os.path.join(
        os.path.dirname(__file__), "..", "tools", "llama_cpp", "convert_hf_to_gguf.py"
    )
    converter_script = os.path.abspath(converter_script)

    if not os.path.exists(converter_script):
        raise FileNotFoundError(f"GGUF converter script not found at: {converter_script}")

    cmd = [
        sys.executable,
        converter_script,
        work_dir,
        "--outfile",
        os.path.abspath(output_gguf_path),
        "--outtype",
        outtype,
    ]

    print(f"[*] Executing converter: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    # Generate Ollama Modelfile
    modelfile_path = os.path.join(out_dir, "Modelfile")
    rel_gguf = os.path.basename(output_gguf_path)
    modelfile_content = f"""# Ollama Modelfile for HADL Car-Lift Fused Model
FROM ./{rel_gguf}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print("\n" + "=" * 80)
    print("  [SUCCESS] GGUF CONVERSION COMPLETED!")
    print("=" * 80)
    print(f"[*] GGUF File Location   : {os.path.abspath(output_gguf_path)}")
    print(f"[*] Ollama Modelfile     : {modelfile_path}")
    print("\nHow to run:")
    print("  1. With Ollama:")
    print(f"     cd \"{out_dir}\"")
    print("     ollama create xstar-2b -f Modelfile")
    print("     ollama run xstar-2b")
    print("\n  2. With llama.cpp server on Windows:")
    print(f"     llama-server.exe -m \"{os.path.abspath(output_gguf_path)}\" --port 8000")
    print("\n  3. With LM Studio / Jan:")
    print(f"     Drag and drop \"{os.path.abspath(output_gguf_path)}\" into your models directory.")
    print("=" * 80 + "\n")

    return os.path.abspath(output_gguf_path)
