#!/usr/bin/env python3
"""
Standalone CLI script to export HADL Dual-Loop fine-tuned models to GGUF format for llama.cpp & Ollama.

Usage:
    python scripts/export_hadl_to_gguf.py \
        --model Qwen/Qwen3.5-2B \
        --checkpoint checkpoints/xstar_2b_omnireason_carlift_500q_checkpoint.pt \
        --output dist/xstar_2b_carlift_q8_0.gguf \
        --outtype q8_0
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_loop.export_gguf import export_hadl_to_gguf


def main():
    parser = argparse.ArgumentParser(
        description="Fuse HADL Car-Lift cognitive controller weights into base model and export as GGUF"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="Hugging Face model ID or local directory (default: Qwen/Qwen3.5-2B)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/xstar_2b_omnireason_carlift_500q_checkpoint.pt",
        help="Path to distilled HADL checkpoint file (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dist/xstar_2b_carlift_q8_0.gguf",
        help="Target GGUF output file path (default: dist/xstar_2b_carlift_q8_0.gguf)",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        default="q8_0",
        choices=["f32", "f16", "bf16", "q8_0", "auto"],
        help="GGUF precision/quantization type (default: q8_0)",
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default="dist/staging_hf_fused",
        help="Directory to stage fused Hugging Face model files before GGUF conversion",
    )

    args = parser.parse_args()

    export_hadl_to_gguf(
        model_id_or_path=args.model,
        checkpoint_path=args.checkpoint,
        output_gguf_path=args.output,
        outtype=args.outtype,
        work_dir=args.staging_dir,
    )


if __name__ == "__main__":
    main()
