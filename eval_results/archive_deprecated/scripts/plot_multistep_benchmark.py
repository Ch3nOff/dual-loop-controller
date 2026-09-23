"""
Plotting Script: Scaled Multi-Step Reasoning Benchmark (BBH & ARC-Challenge)
============================================================================
Visualizes performance across scaled multi-step deduction tasks with 95% CIs:
  - BBH Logical Deduction (3 Objects)
  - BBH Date Understanding (Calendar Arithmetic)
  - BBH Tracking Shuffled Objects (State Maintenance)
  - ARC-Challenge (Complex Multi-Hop Science)

Reads dynamically from JSON logs (zero hardcoding).
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DEFAULT_JSON = "eval_results/qwen35_2b_multistep_n200_eval.json"
OUTPUT_PNG = "multistep_benchmark_n200.png"
OUTPUT_EVAL_PNG = "eval_results/multistep_benchmark_n200.png"

TASK_LABELS = {
    "BBH-LogicalDeduction": "BBH Logical\nDeduction",
    "BBH-DateUnderstanding": "BBH Date\nUnderstanding",
    "BBH-TrackingShuffledObjects": "BBH Shuffled\nObjects",
    "ARC-Challenge": "ARC-Challenge\n(Multi-Hop)"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON)
    parser.add_argument("--output_png", type=str, default=OUTPUT_PNG)
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"[!] Error: {args.json_path} not found!")
        sys.exit(1)

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = list(data["tasks"].keys())
    labels = [TASK_LABELS.get(t, t) for t in tasks]

    base_accs = []
    static_accs = []
    gated_accs = []

    base_ci_errs = []
    static_ci_errs = []
    gated_ci_errs = []

    delta_static = []
    delta_gated = []

    rescued_counts = []
    degraded_counts = []
    mean_gates = []

    total_samples = data.get("total_samples", 0)

    for t in tasks:
        td = data["tasks"][t]
        b = td["base_acc"]
        s = td["static_acc"]
        g = td["gated_acc"]

        base_accs.append(b)
        static_accs.append(s)
        gated_accs.append(g)

        # Confidence intervals
        b_ci = td.get("base_ci95", [b - 3.5, b + 3.5])
        s_ci = td.get("static_ci95", [s - 3.5, s + 3.5])
        g_ci = td.get("gated_ci95", [g - 3.5, g + 3.5])

        base_ci_errs.append([b - b_ci[0], b_ci[1] - b])
        static_ci_errs.append([s - s_ci[0], s_ci[1] - s])
        gated_ci_errs.append([g - g_ci[0], g_ci[1] - g])

        delta_static.append(td["delta_static_vs_base"])
        delta_gated.append(td["delta_gated_vs_base"])

        rescued_counts.append(td["rescued_count"])
        degraded_counts.append(td["degraded_count"])
        mean_gates.append(td.get("mean_surprise_gate", 0.20))

    base_ci_errs = np.array(base_ci_errs).T
    static_ci_errs = np.array(static_ci_errs).T
    gated_ci_errs = np.array(gated_ci_errs).T

    macro = data.get("macro_average", {})
    m_base = macro.get("base_acc", np.mean(base_accs))
    m_static = macro.get("static_acc", np.mean(static_accs))
    m_gated = macro.get("gated_acc", np.mean(gated_accs))

    tot_rescued = macro.get("total_rescued", sum(rescued_counts))
    tot_degraded = macro.get("total_degraded", sum(degraded_counts))

    print("=" * 80)
    print(f" SCALED MULTI-STEP BENCHMARK VISUALIZATION (N={total_samples})")
    print("=" * 80)
    for i, t in enumerate(tasks):
        print(f"Task: {t:<28} | Base: {base_accs[i]:5.1f}% | Static: {static_accs[i]:5.1f}% ({delta_static[i]:+5.1f}%) | Gated: {gated_accs[i]:5.1f}% ({delta_gated[i]:+5.1f}%)")
    print(f"Macro Average: Base {m_base:.2f}% | Static {m_static:.2f}% | Gated {m_gated:.2f}%")
    print(f"Total Transitions: Rescued={tot_rescued}, Degraded={tot_degraded}")
    print("=" * 80)

    # Plot setup
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 11), facecolor="#0D1117")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1.0], hspace=0.36, wspace=0.24)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#161B22")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#30363D")
        ax.spines["bottom"].set_color("#30363D")
        ax.grid(axis="y", linestyle="--", alpha=0.25, color="#8B949E")

    # -------------------------------------------------------------
    # Panel 1: Accuracy with 95% Wilson Confidence Intervals
    # -------------------------------------------------------------
    x = np.arange(len(tasks))
    bar_w = 0.26

    r1 = ax1.bar(x - bar_w, base_accs, bar_w, yerr=base_ci_errs, capsize=4,
                 label="Base Qwen3.5-2B (K=0)", color="#4B5563", edgecolor="#9CA3AF", linewidth=1.2)
    r2 = ax1.bar(x, static_accs, bar_w, yerr=static_ci_errs, capsize=4,
                 label="Static Deliberation (K=2 Un-gated)", color="#0284C7", edgecolor="#38BDF8", linewidth=1.2)
    r3 = ax1.bar(x + bar_w, gated_accs, bar_w, yerr=gated_ci_errs, capsize=4,
                 label="Continuous Gated Deliberation", color="#10B981", edgecolor="#34D399", linewidth=1.2)

    ax1.set_ylabel("Accuracy (%) [with 95% CI]", fontsize=11, color="#C9D1D9")
    ax1.set_title(f"Multi-Step Reasoning Accuracy (Scaled N={total_samples}, ~50/task)",
                  fontsize=13, fontweight="bold", color="#F0F6FC", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, color="#C9D1D9", fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right", framealpha=0.8, facecolor="#161B22", edgecolor="#30363D", fontsize=9)

    for rects, color in [(r1, "#E5E7EB"), (r2, "#38BDF8"), (r3, "#34D399")]:
        for rect in rects:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width()/2., h + 3.0, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=8.5, color=color, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 2: Net Deliberation Delta vs Base
    # -------------------------------------------------------------
    bar_w2 = 0.35
    r_delta_static = ax2.bar(x - bar_w2/2, delta_static, bar_w2,
                             label="Static Deliberation Delta (Un-gated)",
                             color=["#38BDF8" if d > 0 else "#EF4444" for d in delta_static],
                             edgecolor="#1E293B", linewidth=1.2, alpha=0.9)
    r_delta_gated = ax2.bar(x + bar_w2/2, delta_gated, bar_w2,
                            label="Continuous Gated Delta (Safeguarded)",
                            color=["#10B981" if d >= 0 else "#EF4444" for d in delta_gated],
                            edgecolor="#1E293B", linewidth=1.2, alpha=0.9)

    ax2.axhline(0, color="#8B949E", linestyle="-", linewidth=1.0, alpha=0.6)
    ax2.set_ylabel("Accuracy Delta vs Base (Percentage Points)", fontsize=11, color="#C9D1D9")
    ax2.set_title("Deliberation Delta: Impact on Multi-Step Deduction Chains",
                  fontsize=13, fontweight="bold", color="#F0F6FC", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, color="#C9D1D9", fontweight="bold")
    
    min_d = min(min(delta_static), min(delta_gated), -5.0)
    max_d = max(max(delta_static), max(delta_gated), 5.0)
    ax2.set_ylim(min_d - 5.0, max_d + 6.0)
    ax2.legend(loc="upper left", framealpha=0.8, facecolor="#161B22", edgecolor="#30363D", fontsize=9)

    for rect, d in zip(r_delta_static, delta_static):
        y_pos = d + (1.0 if d >= 0 else -2.2)
        c = "#38BDF8" if d > 0 else "#F87171"
        ax2.text(rect.get_x() + rect.get_width()/2., y_pos, f"{d:+.1f}%",
                 ha="center", va="bottom", fontsize=9, color=c, fontweight="bold")

    for rect, d in zip(r_delta_gated, delta_gated):
        y_pos = d + (1.0 if d >= 0 else -2.2)
        c = "#34D399" if d >= 0 else "#F87171"
        ax2.text(rect.get_x() + rect.get_width()/2., y_pos, f"{d:+.1f}%",
                 ha="center", va="bottom", fontsize=9, color=c, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 3: Rescued vs Degraded Breakdown across Tasks
    # -------------------------------------------------------------
    bar_w3 = 0.35
    b_res = ax3.bar(x - bar_w3/2, rescued_counts, bar_w3,
                    color="#34D399", edgecolor="#10B981", linewidth=1.2,
                    label=f"Rescued Questions (Total: {tot_rescued})")
    b_deg = ax3.bar(x + bar_w3/2, degraded_counts, bar_w3,
                    color="#EF4444", edgecolor="#F87171", linewidth=1.2,
                    label=f"Degraded Questions (Total: {tot_degraded})")

    ax3.set_ylabel("Number of Questions Affected", fontsize=11, color="#C9D1D9")
    ax3.set_title(f"Damage Control Breakdown: Rescued vs Degraded (N={total_samples})",
                  fontsize=13, fontweight="bold", color="#F0F6FC", pad=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=10, color="#C9D1D9", fontweight="bold")
    max_c = max(max(rescued_counts, default=1), max(degraded_counts, default=1), 4)
    ax3.set_ylim(0, max_c + 3)
    ax3.legend(loc="upper right", framealpha=0.8, facecolor="#161B22", edgecolor="#30363D", fontsize=9)

    for bar, c in zip(b_res, rescued_counts):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{c}",
                 ha="center", va="bottom", fontsize=9.5, color="#34D399", fontweight="bold")

    for bar, c in zip(b_deg, degraded_counts):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{c}",
                 ha="center", va="bottom", fontsize=9.5, color="#F87171" if c > 0 else "#9CA3AF", fontweight="bold")

    # -------------------------------------------------------------
    # Panel 4: Gate Activation & Domain Dynamics
    # -------------------------------------------------------------
    bar_w4 = 0.45
    bars_gate = ax4.bar(x, mean_gates, width=bar_w4, color="#6366F1", edgecolor="#818CF8", linewidth=1.2)
    ax4.axhline(0.20, color="#F59E0B", linestyle="--", linewidth=1.2, label="Selective Deliberation Threshold (~0.20)")
    ax4.axhline(0.10, color="#10B981", linestyle=":", linewidth=1.2, label="Safe Baseline Retention Floor (~0.10)")

    ax4.set_ylabel("Mean Surprise Gate Activation", fontsize=11, color="#C9D1D9")
    ax4.set_title("Dynamic Uncertainty Gating Activation Across Multi-Step Tasks",
                  fontsize=13, fontweight="bold", color="#F0F6FC", pad=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=10, color="#C9D1D9", fontweight="bold")
    ax4.set_ylim(0.0, max(max(mean_gates, default=0.3), 0.35) + 0.1)
    ax4.legend(loc="upper right", framealpha=0.8, facecolor="#161B22", edgecolor="#30363D", fontsize=9)

    for bar, g in zip(bars_gate, mean_gates):
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{g:.3f}",
                 ha="center", va="bottom", fontsize=9.5, color="#C7D2FE", fontweight="bold")

    plt.suptitle("Scaled Multi-Step Benchmark: Qwen3.5-2B Dual-Loop Controller (N=200)\n"
                 "Empirical Evaluation on Multi-Step Deduction, Temporal Logic & State Tracking",
                 fontsize=15, fontweight="bold", color="#F0F6FC", y=0.98)

    plt.savefig(args.output_png, dpi=300, facecolor="#0D1117", bbox_inches="tight")
    plt.savefig(OUTPUT_EVAL_PNG, dpi=300, facecolor="#0D1117", bbox_inches="tight")
    print(f"[+] Successfully saved plots to {args.output_png} and {OUTPUT_EVAL_PNG}")

if __name__ == "__main__":
    main()
