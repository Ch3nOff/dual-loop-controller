import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    json_path = os.path.join(os.path.dirname(__file__), "eval_results", "qwen35_k_ablation_and_latency.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plt.style.use("dark_background")
    bg_color = "#0D1117"
    card_color = "#161B22"
    text_color = "#F0F6FC"
    border_color = "#30363D"

    # =========================================================================
    # FIGURE 1: K-Step Compute Ablation & Marginal Gain (Diminishing Returns)
    # =========================================================================
    fig_ablation, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=bg_color)

    ablation = data["k_step_ablation"]
    steps = ablation["steps"]
    bm = ablation["benchmarks"]

    # Panel 1: Trajectory
    ax1.set_facecolor(card_color)
    ax1.grid(True, linestyle="--", alpha=0.25, color="#8B949E")

    palette = {
        "AA-LCR": ("#58A6FF", "o", "-", 2.8, "Relational Chaining"),
        "PolyMATH": ("#BC8CFF", "s", "-", 2.8, "Math Deduction"),
        "SuperGPQA": ("#3FB950", "^", "-", 2.0, "Deep STEM"),
        "LongBench v2": ("#FFA657", "D", "-", 2.0, "Long Context"),
        "MMLU-Redux": ("#8B949E", "x", "--", 1.8, "Core Knowledge (System 1)")
    }

    for name, (color, marker, ls, lw, domain) in palette.items():
        scores = bm[name]["scores"]
        ax1.plot(steps, scores, label=f"{name} ({domain})",
                 color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=8)

    # Shaded regime boundaries
    ax1.axvspan(-0.2, 2.0, color="#58A6FF", alpha=0.08)
    ax1.axvspan(2.0, 3.0, color="#3FB950", alpha=0.12)
    ax1.axvspan(3.0, 5.2, color="#F85149", alpha=0.06)

    # Place regime headers near top banner
    ax1.text(0.9, 91, "Steep Scaling\n(K = 0 -> 2)", color="#58A6FF", fontsize=9.5, fontweight="bold", ha="center")
    ax1.text(2.5, 91, "Optimal Peak\n(K = 2 -> 3)", color="#3FB950", fontsize=9.5, fontweight="bold", ha="center")
    ax1.text(4.1, 91, "Saturation / Drift\n(K >= 4)", color="#F85149", fontsize=9.5, fontweight="bold", ha="center")

    # Annotate peak values
    ax1.annotate("Peak: 46.2%\n(+20.2%)", xy=(3, 46.2), xytext=(3.2, 52.0),
                 arrowprops=dict(facecolor="#58A6FF", shrink=0.08, width=1.5, headwidth=5),
                 color="#58A6FF", fontweight="bold", fontsize=9.5)

    ax1.annotate("Peak: 41.5%\n(+14.7%)", xy=(3, 41.5), xytext=(3.2, 33.0),
                 arrowprops=dict(facecolor="#BC8CFF", shrink=0.08, width=1.5, headwidth=5),
                 color="#BC8CFF", fontweight="bold", fontsize=9.5)

    ax1.set_xlabel("Outer Loop Deliberation Steps (K)", fontsize=12, fontweight="bold", color=text_color, labelpad=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold", color=text_color)
    ax1.set_title("(A) Reasoning Performance vs Deliberation Depth (K)", fontsize=13, fontweight="bold", color=text_color, pad=12)
    ax1.set_xticks(steps)
    ax1.set_xlim(-0.2, 5.2)
    ax1.set_ylim(20, 98)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.02, 0.48), frameon=True, facecolor=card_color, edgecolor=border_color, fontsize=8.5)

    # Panel 2: Marginal Gain (Derivative dAcc / dK)
    ax2.set_facecolor(card_color)
    ax2.grid(True, linestyle="--", alpha=0.25, color="#8B949E")

    k_eval = [1, 2, 3, 4, 5]
    aa_gain = [bm["AA-LCR"]["marginal_step_gain"][i] for i in k_eval]
    poly_gain = [bm["PolyMATH"]["marginal_step_gain"][i] for i in k_eval]

    x = np.arange(len(k_eval))
    width = 0.35

    bars1 = ax2.bar(x - width/2, aa_gain, width, label="AA-LCR (Relational)", color="#58A6FF", alpha=0.9)
    bars2 = ax2.bar(x + width/2, poly_gain, width, label="PolyMATH (Math)", color="#BC8CFF", alpha=0.9)

    ax2.axhline(0, color="#8B949E", linestyle="-", linewidth=1)

    for bar in bars1:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.3 if h >= 0 else -0.7
        ax2.text(bar.get_x() + bar.get_width()/2, h + offset, f"{h:+.1f}%",
                 ha="center", va=va, color="#58A6FF", fontsize=9, fontweight="bold")

    for bar in bars2:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.3 if h >= 0 else -0.7
        ax2.text(bar.get_x() + bar.get_width()/2, h + offset, f"{h:+.1f}%",
                 ha="center", va=va, color="#BC8CFF", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Ponder Step Transition", fontsize=12, fontweight="bold", color=text_color, labelpad=10)
    ax2.set_ylabel("Marginal Accuracy Gain Δ (%)", fontsize=12, fontweight="bold", color=text_color)
    ax2.set_title("(B) Marginal Gain per Step (Proof of Diminishing Returns)", fontsize=13, fontweight="bold", color=text_color, pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["K=0->1", "K=1->2", "K=2->3", "K=3->4", "K=4->5"])
    ax2.set_ylim(-2, 14.5)
    ax2.legend(loc="upper right", frameon=True, facecolor=card_color, edgecolor=border_color, fontsize=9.5)

    plt.suptitle("Qwen3.5-2B Dual-Loop Deliberation: K-Ablation & Saturation Analysis", fontsize=15, fontweight="bold", color=text_color, y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig_ablation.text(
        0.5, 0.015,
        "Benchmark Setup: NVIDIA RTX 4090 (24GB) | PyTorch 2.4 | bfloat16 | Qwen3.5-2B (2.31B params) + 96.5M Latent Adapter (Layer 12)",
        ha="center", fontsize=8.5, color="#8B949E"
    )
    fig_ablation.savefig("figure_k_ablation.png", dpi=300, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig_ablation)
    print("[OK] figure_k_ablation.png saved!")

    # =========================================================================
    # FIGURE 2: Inference Latency, FLOPS & Pareto Frontier
    # =========================================================================
    fig_pareto, (p1, p2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=bg_color)

    methods = data["latency_and_compute_overhead"]["methods"]

    # Extract series for Dual-Loop vs CoT
    dl_methods = [m for m in methods if "Dual-Loop" in m["name"] or "Base" in m["name"]]
    cot_methods = [m for m in methods if "Explicit CoT" in m["name"] or "Base" in m["name"]]

    # Panel 1: Accuracy vs Added Latency (Pareto Frontier)
    p1.set_facecolor(card_color)
    p1.grid(True, linestyle="--", alpha=0.25, color="#8B949E")

    # Plot CoT curve
    cot_overhead = [m["deliberation_overhead_ms"] for m in cot_methods]
    cot_acc = [m["aa_lcr_acc"] for m in cot_methods]
    p1.plot(cot_overhead, cot_acc, color="#F85149", linestyle="--", marker="s", linewidth=2.2, markersize=8, label="Explicit CoT (Tokens)")

    # Plot Dual-Loop curve
    dl_overhead = [m["deliberation_overhead_ms"] for m in dl_methods]
    dl_acc = [m["aa_lcr_acc"] for m in dl_methods]
    p1.plot(dl_overhead, dl_acc, color="#3FB950", linestyle="-", marker="o", linewidth=2.8, markersize=9, label="Dual-Loop Controller (Latent)")

    # Annotate points
    for m in dl_methods:
        lbl = m["name"].replace("Model ", "")
        p1.annotate(f"{lbl}\n({m['aa_lcr_acc']}%, +{m['deliberation_overhead_ms']}ms)",
                    xy=(m["deliberation_overhead_ms"], m["aa_lcr_acc"]),
                    xytext=(m["deliberation_overhead_ms"] + 60, m["aa_lcr_acc"] - 1.5),
                    color="#3FB950", fontsize=8.5, fontweight="bold")

    for m in cot_methods[1:]:
        lbl = m["name"].replace("Explicit CoT ", "CoT ")
        p1.annotate(f"{lbl}\n(+{m['deliberation_overhead_ms']:.0f}ms)",
                    xy=(m["deliberation_overhead_ms"], m["aa_lcr_acc"]),
                    xytext=(m["deliberation_overhead_ms"] - 250, m["aa_lcr_acc"] + 1.2),
                    color="#F85149", fontsize=8.5, fontweight="bold")

    p1.set_xlabel("Added Inference Latency (ms)", fontsize=12, fontweight="bold", color=text_color, labelpad=10)
    p1.set_ylabel("AA-LCR Relational Accuracy (%)", fontsize=12, fontweight="bold", color=text_color)
    p1.set_title("(A) Accuracy vs Added Deliberation Latency (Pareto Frontier)", fontsize=13, fontweight="bold", color=text_color, pad=12)
    p1.set_xlim(-100, 3800)
    p1.set_ylim(23, 50)
    p1.legend(loc="lower right", frameon=True, facecolor=card_color, edgecolor=border_color, fontsize=10)

    # Inset box highlighting the 99% latency advantage
    p1.text(600, 27, "Dual-Loop Advantage:\n• +20.2% accuracy in only +3.8 ms\n• 99.89% faster than 300-token CoT\n• Zero output token inflation",
            fontsize=9.5, color="#F0F6FC", bbox=dict(boxstyle="round,pad=0.6", facecolor="#21262D", edgecolor="#3FB950", lw=1.5))

    # Panel 2: Added Compute (GFLOPs) & Effective Generation Throughput
    p2.set_facecolor(card_color)
    p2.grid(True, linestyle="--", alpha=0.25, color="#8B949E")

    names = [m["name"].replace("Model ", "").replace("Explicit ", "") for m in methods]
    gflops = [m["added_inference_gflops"] for m in methods]
    throughputs = [m["decode_throughput_tok_sec"] for m in methods]

    y_pos = np.arange(len(names))

    # Horizontal Bar Chart for FLOPs
    colors = ["#8B949E"] + ["#3FB950"]*3 + ["#F85149"]*3
    bars = p2.barh(y_pos, gflops, color=colors, alpha=0.85, height=0.55)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        txt = f"{w:.2f} GFLOPs" if w < 10 else f"{w:.0f} GFLOPs"
        p2.text(w + 20, bar.get_y() + bar.get_height()/2, txt,
                va="center", color=text_color, fontsize=8.5, fontweight="bold")

    p2.set_yticks(y_pos)
    p2.set_yticklabels(names, fontsize=9.5, color=text_color)
    p2.set_xlabel("Added Inference Compute per Sample (GFLOPs)", fontsize=12, fontweight="bold", color=text_color, labelpad=10)
    p2.set_title("(B) Compute Overhead: Dual-Loop vs Discrete CoT", fontsize=13, fontweight="bold", color=text_color, pad=12)
    p2.set_xlim(0, 1600)

    p2.text(450, 2.0, "Dual-Loop Outer Loop (K=3):\nOnly +0.40 GFLOPs total\n(< 0.04% of prompt prefill)",
            fontsize=9.5, color="#3FB950", bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262D", edgecolor="#3FB950", lw=1.2))

    plt.suptitle("Inference Efficiency & Pareto Frontier: Dual-Loop vs Autoregressive Chain-of-Thought",
                 fontsize=15, fontweight="bold", color=text_color, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig_pareto.text(
        0.5, 0.015,
        "Benchmark Hardware: Single NVIDIA RTX 4090 (24GB) | PyTorch 2.4 | bfloat16 | Batch Size = 1 | Prompt Length N = 256\n"
        "*Latent FLOPs Definition: +0.40 GFLOPs accounts strictly for recurrent controller updates (L_thought=8, M=16 CWM slots), not a full 24-layer backbone pass.",
        ha="center", fontsize=8.5, color="#8B949E", linespacing=1.35
    )
    fig_pareto.savefig("figure_pareto_latency.png", dpi=300, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig_pareto)
    print("[OK] figure_pareto_latency.png saved!")

    # =========================================================================
    # MASTER FIGURE 3: Combined 4-Panel Publication Chart
    # =========================================================================
    fig_master, axes = plt.subplots(2, 2, figsize=(18, 12.5), facecolor=bg_color)

    # Copy Panel 1: Trajectory
    ax_t = axes[0, 0]
    ax_t.set_facecolor(card_color)
    ax_t.grid(True, linestyle="--", alpha=0.25, color="#8B949E")
    for name, (color, marker, ls, lw, domain) in palette.items():
        scores = bm[name]["scores"]
        ax_t.plot(steps, scores, label=f"{name}", color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=7)
    ax_t.axvspan(-0.2, 2.0, color="#58A6FF", alpha=0.08)
    ax_t.axvspan(2.0, 3.0, color="#3FB950", alpha=0.12)
    ax_t.axvspan(3.0, 5.2, color="#F85149", alpha=0.06)
    ax_t.text(0.9, 91, "Steep Scaling\n(K = 0 -> 2)", color="#58A6FF", fontsize=8.5, fontweight="bold", ha="center")
    ax_t.text(2.5, 91, "Optimal Peak\n(K = 2 -> 3)", color="#3FB950", fontsize=8.5, fontweight="bold", ha="center")
    ax_t.text(4.1, 91, "Saturation / Drift\n(K >= 4)", color="#F85149", fontsize=8.5, fontweight="bold", ha="center")
    ax_t.annotate("Peak: 46.2%\n(+20.2%)", xy=(3, 46.2), xytext=(3.2, 52.0),
                 arrowprops=dict(facecolor="#58A6FF", shrink=0.08, width=1.2, headwidth=4),
                 color="#58A6FF", fontweight="bold", fontsize=8.5)
    ax_t.set_xlabel("Ponder Steps (K)", fontsize=11, fontweight="bold", color=text_color)
    ax_t.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold", color=text_color)
    ax_t.set_title("(A) K-Step Compute Scaling Trajectory", fontsize=12, fontweight="bold", color=text_color)
    ax_t.set_xticks(steps)
    ax_t.set_ylim(20, 98)
    ax_t.legend(loc="lower left", bbox_to_anchor=(0.02, 0.48), frameon=True, facecolor=card_color, edgecolor=border_color, fontsize=8)

    # Copy Panel 2: Marginal Gain
    ax_m = axes[0, 1]
    ax_m.set_facecolor(card_color)
    ax_m.grid(True, linestyle="--", alpha=0.25, color="#8B949E")
    bars1 = ax_m.bar(x - width/2, aa_gain, width, label="AA-LCR", color="#58A6FF", alpha=0.9)
    bars2 = ax_m.bar(x + width/2, poly_gain, width, label="PolyMATH", color="#BC8CFF", alpha=0.9)
    ax_m.axhline(0, color="#8B949E", linestyle="-", linewidth=1)
    for bar in bars1:
        h = bar.get_height()
        ax_m.text(bar.get_x() + bar.get_width()/2, h + (0.3 if h>=0 else -0.7), f"{h:+.1f}%",
                  ha="center", va="bottom" if h>=0 else "top", color="#58A6FF", fontsize=8, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax_m.text(bar.get_x() + bar.get_width()/2, h + (0.3 if h>=0 else -0.7), f"{h:+.1f}%",
                  ha="center", va="bottom" if h>=0 else "top", color="#BC8CFF", fontsize=8, fontweight="bold")
    ax_m.set_xlabel("Transition (K -> K+1)", fontsize=11, fontweight="bold", color=text_color)
    ax_m.set_ylabel("Marginal Delta Δ (%)", fontsize=11, fontweight="bold", color=text_color)
    ax_m.set_title("(B) Marginal Gain (Diminishing Returns Proof)", fontsize=12, fontweight="bold", color=text_color)
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(["K=0->1", "K=1->2", "K=2->3", "K=3->4", "K=4->5"])
    ax_m.set_ylim(-2, 14.5)
    ax_m.legend(loc="upper right", frameon=True, facecolor=card_color, edgecolor=border_color, fontsize=8.5)

    # Copy Panel 3: Latency Pareto Frontier
    ax_p = axes[1, 0]
    ax_p.set_facecolor(card_color)
    ax_p.grid(True, linestyle="--", alpha=0.25, color="#8B949E")
    ax_p.plot(cot_overhead, cot_acc, color="#F85149", linestyle="--", marker="s", linewidth=2.0, markersize=7, label="Explicit CoT (Tokens)")
    ax_p.plot(dl_overhead, dl_acc, color="#3FB950", linestyle="-", marker="o", linewidth=2.5, markersize=8, label="Dual-Loop (Latent)")
    for m in dl_methods:
        lbl = m["name"].replace("Model ", "")
        ax_p.annotate(f"{lbl}\n({m['aa_lcr_acc']}%, +{m['deliberation_overhead_ms']}ms)",
                      xy=(m["deliberation_overhead_ms"], m["aa_lcr_acc"]),
                      xytext=(m["deliberation_overhead_ms"] + 60, m["aa_lcr_acc"] - 1.5),
                      color="#3FB950", fontsize=8, fontweight="bold")
    for m in cot_methods[1:]:
        lbl = m["name"].replace("Explicit CoT ", "CoT ")
        ax_p.annotate(f"{lbl}\n(+{m['deliberation_overhead_ms']:.0f}ms)",
                      xy=(m["deliberation_overhead_ms"], m["aa_lcr_acc"]),
                      xytext=(m["deliberation_overhead_ms"] - 250, m["aa_lcr_acc"] + 1.2),
                      color="#F85149", fontsize=8, fontweight="bold")
    ax_p.set_xlabel("Added Inference Latency (ms)", fontsize=11, fontweight="bold", color=text_color)
    ax_p.set_ylabel("AA-LCR Accuracy (%)", fontsize=11, fontweight="bold", color=text_color)
    ax_p.set_title("(C) Accuracy vs Added Latency (Pareto Frontier)", fontsize=12, fontweight="bold", color=text_color)
    ax_p.set_xlim(-100, 3800)
    ax_p.set_ylim(23, 50)
    ax_p.legend(loc="lower right", frameon=True, facecolor=card_color, edgecolor=border_color, fontsize=9)
    ax_p.text(600, 27, "Dual-Loop Advantage:\n• +20.2% in only +3.8 ms\n• 99.89% faster than CoT-300\n• Zero token inflation",
              fontsize=8.5, color="#F0F6FC", bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262D", edgecolor="#3FB950", lw=1.2))

    # Copy Panel 4: Compute Overhead FLOPs
    ax_f = axes[1, 1]
    ax_f.set_facecolor(card_color)
    ax_f.grid(True, linestyle="--", alpha=0.25, color="#8B949E")
    bars_f = ax_f.barh(y_pos, gflops, color=colors, alpha=0.85, height=0.55)
    for bar in bars_f:
        w = bar.get_width()
        txt = f"{w:.2f} GFLOPs" if w < 10 else f"{w:.0f} GFLOPs"
        ax_f.text(w + 20, bar.get_y() + bar.get_height()/2, txt,
                  va="center", color=text_color, fontsize=8, fontweight="bold")
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(names, fontsize=9, color=text_color)
    ax_f.set_xlabel("Added Compute (GFLOPs)", fontsize=11, fontweight="bold", color=text_color)
    ax_f.set_title("(D) Added Compute Budget Comparison", fontsize=12, fontweight="bold", color=text_color)
    ax_f.set_xlim(0, 1600)
    ax_f.text(450, 2.0, "Dual-Loop Outer Loop (K=3):\nOnly +0.40 GFLOPs total\n(< 0.04% of prompt prefill)",
              fontsize=9.0, color="#3FB950", bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262D", edgecolor="#3FB950", lw=1.2))

    plt.suptitle("Dual-Loop Cognitive Controller: Technical Paper Trade-Off & Ablation Suite",
                 fontsize=16, fontweight="bold", color=text_color, y=0.99)
    plt.tight_layout(rect=[0, 0.035, 1, 0.97])
    fig_master.text(
        0.5, 0.012,
        "Benchmark Setup: NVIDIA RTX 4090 (24GB) | PyTorch 2.4 | bfloat16 | Batch Size = 1 | Prompt Length N = 256 | "
        "*Latent FLOPs Definition: Computes recurrent controller updates only (L_thought=8, M=16 CWM slots), not a full 24-layer backbone pass.",
        ha="center", fontsize=9.0, color="#8B949E"
    )
    fig_master.savefig("paper_tradeoffs_and_ablation.png", dpi=300, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig_master)
    print("[OK] paper_tradeoffs_and_ablation.png saved!")

if __name__ == "__main__":
    main()
