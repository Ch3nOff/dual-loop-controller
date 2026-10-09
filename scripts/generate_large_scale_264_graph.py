#!/usr/bin/env python3
"""
Stand-alone plot generator for the Large-Scale 264 Canonical Tasks Benchmark.
Reads from eval_results/large_scale_264_benchmark.json and writes high-res graphics to
docs/images/hadl_vs_baseline_large_scale_264_benchmark.png.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "eval_results" / "large_scale_264_benchmark.json"
OUT_IMG = BASE_DIR / "docs" / "images" / "hadl_vs_baseline_large_scale_264_benchmark.png"

def main():
    if not JSON_PATH.exists():
        print(f"[-] Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_he = data["baseline"]["humaneval"]
    hadl_he = data["hadl_v45"]["humaneval"]
    base_gsm = data["baseline"]["gsm8k"]
    hadl_gsm = data["hadl_v45"]["gsm8k"]

    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.suptitle(
        "Empirical Benchmark Suite: Frozen Base (2B) vs HADL v4.5 (2B)\n"
        "Large-Scale Rigorous Evaluation: 164 HumanEval Tasks + 100 GSM8K Problems (Total N=264)",
        fontsize=16, fontweight="bold", y=0.98
    )

    # 1. Macro Accuracy Comparison Bar Chart
    ax1 = axes[0, 0]
    labels = ["OpenAI HumanEval\n(164 Tasks - 100% Full)", "OpenAI GSM8K\n(100 Tasks - Test Split)"]
    base_scores = [base_he["pass_rate_pct"], base_gsm["accuracy_pct"]]
    hadl_scores = [hadl_he["pass_rate_pct"], hadl_gsm["accuracy_pct"]]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax1.bar(x - w/2, base_scores, w, label="Baseline (Frozen Qwen3.5-2B)", color="#1f77b4", alpha=0.9)
    bars2 = ax1.bar(x + w/2, hadl_scores, w, label="HADL v4.5 (Dual-Loop 2B)", color="#d62728", alpha=0.9)

    ax1.set_ylabel("Accuracy / Pass Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Canonical Task Accuracy & Pass Rates (N=264)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 60)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=11)

    for bar in bars1:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + 1.2, f"{y:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1c3d5a")
    for bar in bars2:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + 1.2, f"{y:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#7b1113")

    # Annotate deltas
    ax1.annotate("Defensive Bias Mismatch\nΔ = -3.05%", xy=(0 + w/2, hadl_scores[0]), xytext=(0.15, 32),
                 arrowprops=dict(arrowstyle="->", color="#7b1113", lw=1.5),
                 fontsize=10, fontweight="bold", color="#7b1113", bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#ef5350"))
    ax1.annotate("Step-Wise Latent Gain\nΔ = +26.00% (2.625×)", xy=(1 + w/2, hadl_scores[1]), xytext=(1.05, 50),
                 arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.5),
                 fontsize=10, fontweight="bold", color="#2e7d32", bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#66bb6a"))

    # 2. Cumulative Pass Rate on HumanEval (1..164)
    ax2 = axes[0, 1]
    base_he_cum = []
    c = 0
    for idx, (tid, data_t) in enumerate(base_he["tasks"].items(), 1):
        if data_t["passed"]: c += 1
        base_he_cum.append((c / idx) * 100.0)

    hadl_he_cum = []
    c = 0
    for idx, (tid, data_t) in enumerate(hadl_he["tasks"].items(), 1):
        if data_t["passed"]: c += 1
        hadl_he_cum.append((c / idx) * 100.0)

    ax2.plot(range(1, 165), base_he_cum, label="Baseline (Frozen 2B)", color="#4A90E2", lw=2)
    ax2.plot(range(1, 165), hadl_he_cum, label="HADL v4.5 (2B)", color="#D9381E", lw=2.5)
    ax2.set_xlabel("HumanEval Task Index (1 to 164)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Pass@1 (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Cumulative Pass Rate Trajectory on HumanEval (164 Tasks)", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=11)

    # 3. Cumulative Accuracy on GSM8K (1..100)
    ax3 = axes[1, 0]
    base_gsm_cum = []
    c = 0
    for idx, (tid, data_t) in enumerate(base_gsm["tasks"].items(), 1):
        if data_t["correct"]: c += 1
        base_gsm_cum.append((c / idx) * 100.0)

    hadl_gsm_cum = []
    c = 0
    for idx, (tid, data_t) in enumerate(hadl_gsm["tasks"].items(), 1):
        if data_t["correct"]: c += 1
        hadl_gsm_cum.append((c / idx) * 100.0)

    ax3.plot(range(1, 101), base_gsm_cum, label="Baseline (Frozen 2B)", color="#4A90E2", lw=2)
    ax3.plot(range(1, 101), hadl_gsm_cum, label="HADL v4.5 (2B)", color="#D9381E", lw=2.5)
    ax3.set_xlabel("GSM8K Problem Index (1 to 100)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Cumulative Accuracy (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Cumulative Accuracy Trajectory on GSM8K (100 Problems)", fontsize=13, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="lower right", fontsize=11)

    # 4. Token Generation Throughput (TPS) & Hardware Metrics
    ax4 = axes[1, 1]
    metric_labels = ["HumanEval TPS\n(Tokens/s)", "GSM8K TPS\n(Tokens/s)"]
    base_tps = [base_he["throughput_tps"], base_gsm["throughput_tps"]]
    hadl_tps = [hadl_he["throughput_tps"], hadl_gsm["throughput_tps"]]
    x2 = np.arange(len(metric_labels))

    b1 = ax4.bar(x2 - w/2, base_tps, w, label="Baseline", color="#6BAED6", alpha=0.85)
    b2 = ax4.bar(x2 + w/2, hadl_tps, w, label="HADL v4.5", color="#E6550D", alpha=0.9)
    ax4.set_ylabel("Throughput (Tokens/s)", fontsize=11, fontweight="bold")
    ax4.set_title(f"Inference Throughput & Latency ({data.get('gpu_name', 'GPU')})", fontsize=13, fontweight="bold")
    ax4.set_xticks(x2)
    ax4.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax4.set_ylim(0, 35)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", fontsize=11)

    for bar in b1:
        y = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in b2:
        y = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, y + 0.5, f"{y:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(OUT_IMG), dpi=300)
    plt.close()
    print(f"[+] Successfully generated high-resolution chart at: {OUT_IMG}")

if __name__ == "__main__":
    main()
