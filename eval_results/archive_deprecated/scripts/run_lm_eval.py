"""
Dual-Loop Cognitive Controller: LM-Evaluation-Harness Runner & Visualizer
========================================================================
Runs standard academic benchmarks (MMLU, MMLU-Pro, C-Eval, GPQA, IFEval, etc.)
on Dual-Loop Augmented Qwen models using lm-evaluation-harness (lm_eval),
exports results to ./eval_results/, and automatically renders the dark-mode scorecard.
"""

import os
import sys
import json
import argparse
import time
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop import DualLoopQwenModel, attach_dual_loop_to_qwen


def render_scorecard(results_dict: Dict[str, float], output_image_path: str = "benchmark_barchart.png", model_name: str = "Qwen3.5-2B + Dual-Loop"):
    """
    Renders dark-mode barchart matching the exact llm-stats.com scorecard aesthetics.
    """
    sorted_data = dict(sorted(results_dict.items(), key=lambda item: item[1], reverse=True))
    tasks = list(sorted_data.keys())
    scores = list(sorted_data.values())

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(tasks, scores, color="#5B4DF6", width=0.55)

    # Add numeric score labels on top of each bar
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.2,
            f"{score:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#E0E0E0",
            fontweight="bold"
        )

    ax.set_ylabel("Score (%)", fontsize=11, color="#A0A0A0")
    ax.set_title(
        f"{model_name} Performance Across Datasets (llm-stats.com style)",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#404040")
    ax.spines["bottom"].set_color("#404040")
    ax.grid(axis="y", linestyle="--", alpha=0.15)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"\n[Visualization] Scorecard barchart saved to: {os.path.abspath(output_image_path)}")


def main():
    parser = argparse.ArgumentParser(description="Run lm_eval on Dual-Loop Qwen and visualize scorecard")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B",
                        help="Base HuggingFace model identifier")
    parser.add_argument("--adapter_path", type=str, default="dual_loop/checkpoints/qwen35_2b_adapter.pt",
                        help="Path to trained Dual-Loop adapter weights")
    parser.add_argument("--tasks", type=str, default="mmlu,mmlu_pro,ceval-valid,gpqa,ifeval",
                        help="Comma-separated benchmark tasks")
    parser.add_argument("--batch_size", type=str, default="auto",
                        help="Batch size or 'auto'")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")
    parser.add_argument("--output_path", type=str, default="./eval_results/",
                        help="Output directory for evaluation results JSON")
    parser.add_argument("--k_steps", type=int, default=2,
                        help="Pondering steps for Dual-Loop deliberation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of samples per task (for testing)")
    parser.add_argument("--trust_remote_code", action="store_true", default=False,
                        help="Allow executing remote code from Hugging Face Hub")
    parser.add_argument("--revision", type=str, default="15852e8c16360a2fea060d615a32b45270f8a8fc",
                        help="Pinned commit SHA for supply-chain security (SEC-02)")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    print("=" * 80)
    print(" DUAL-LOOP QWEN: LM-EVALUATION-HARNESS BENCHMARK RUNNER")
    print("=" * 80)
    print(f"Base Model:     {args.model}")
    print(f"Revision:       {args.revision}")
    print(f"Adapter:        {args.adapter_path}")
    print(f"Tasks:          {args.tasks}")
    print(f"Device:         {args.device}")
    print(f"Ponder Steps:   K={args.k_steps}")
    print(f"Output Path:    {args.output_path}")
    print("-" * 80)

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.evaluator import simple_evaluate
    except ImportError:
        print("[Error] lm-eval is not installed. Please run: pip install lm-eval")
        sys.exit(1)

    print(f"[1/4] Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[2/4] Loading base model and attaching Dual-Loop Controller...")
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        device_map=args.device if "cuda" in args.device else None
    )
    if args.device == "cpu":
        base_model = base_model.to("cpu")

    model = attach_dual_loop_to_qwen(base_model, k_steps=args.k_steps)
    if os.path.exists(args.adapter_path):
        print(f"[2/4] Loading trained adapter weights from {args.adapter_path}...")
        model.load_adapter(args.adapter_path)
    else:
        print(f"[Warning] Adapter path {args.adapter_path} not found. Running with initialized adapter.")

    print(f"[3/4] Initializing lm_eval HFLM wrapper...")
    bs = 1 if args.batch_size == "auto" and args.device == "cpu" else (args.batch_size if args.batch_size != "auto" else "auto")
    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=bs,
        device=args.device
    )

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[4/4] Executing evaluation across tasks: {task_list}...")
    t0 = time.time()
    results = simple_evaluate(
        model=hflm,
        tasks=task_list,
        limit=args.limit
    )
    elapsed = time.time() - t0
    print(f"\n[Evaluation Complete] Elapsed time: {elapsed:.1f}s")

    # Save output JSON
    timestamp = int(time.time())
    output_json_path = os.path.join(args.output_path, f"eval_results_{timestamp}.json")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Results] Full evaluation results written to: {os.path.abspath(output_json_path)}")

    # Extract task accuracies for scorecard visualization
    extracted_scores = {}
    if "results" in results:
        for task_name, task_metrics in results["results"].items():
            for metric_key in ["acc,none", "acc_norm,none", "exact_match,none", "acc"]:
                if metric_key in task_metrics:
                    extracted_scores[task_name] = round(float(task_metrics[metric_key]) * 100.0, 1)
                    break

    # Render visualization only from genuinely evaluated tasks
    if extracted_scores:
        render_scorecard(extracted_scores, output_image_path="benchmark_barchart.png", model_name="Qwen3.5-2B + Dual-Loop")
    else:
        print("[Notice] No scalar accuracy metrics extracted from results to render.")


if __name__ == "__main__":
    main()
