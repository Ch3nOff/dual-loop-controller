import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

AUTHTEST_JSON = "eval_results/qwen35_2b_authentic_suite_n160.json"
BASE_JSON = "eval_results/qwen35_2b_full_base_k0.json"
LOOP_JSON = "eval_results/qwen35_2b_full_dualloop_k2.json"
OUTPUT_PNG = "full_benchmark_scoreboard.png"
TASKS = ["ARC-Easy", "ARC-Challenge", "OpenBookQA", "PIQA"]

TASK_DISPLAY_NAMES = {
    "ARC-Easy": "ARC-Easy (Sains)",
    "ARC-Challenge": "ARC-Challenge (Nalar)",
    "OpenBookQA": "OpenBookQA (Fakta)",
    "PIQA": "PIQA (Fisika/Commonsense)"
}

def main():
    if os.path.exists(AUTHTEST_JSON):
        with open(AUTHTEST_JSON, "r", encoding="utf-8") as f:
            auth_data = json.load(f)
        use_auth = True
    elif os.path.exists(BASE_JSON) and os.path.exists(LOOP_JSON):
        with open(BASE_JSON, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        with open(LOOP_JSON, "r", encoding="utf-8") as f:
            loop_data = json.load(f)
        use_auth = False
    else:
        print(f"[!] Target evaluation files not found.")
        sys.exit(1)

    base_scores = []
    base_errs = []
    loop_scores = []
    loop_errs = []
    deltas = []

    print("=" * 90)
    print(" AUTHENTIC MULTI-TASK BENCHMARK RESULTS: QWEN3.5-2B + DUAL-LOOP (ACC_NORM)")
    print("=" * 90)

    total_samples = 0
    total_both_correct = 0
    total_both_wrong = 0
    total_improved = 0
    total_degraded = 0
    all_rescued_questions = []

    if use_auth:
        for task in TASKS:
            t_data = auth_data["tasks"][task]
            b_acc = t_data["base_acc_norm"]
            l_acc = t_data["loop_acc_norm"]
            d_acc = t_data["delta_norm"]
            # Estimate standard error for Bernoulli trials: sqrt(p * (1-p) / N)
            N = t_data["samples"]
            b_err = np.sqrt((b_acc/100.0) * (1.0 - b_acc/100.0) / N) * 100.0
            l_err = np.sqrt((l_acc/100.0) * (1.0 - l_acc/100.0) / N) * 100.0

            base_scores.append(b_acc)
            base_errs.append(b_err)
            loop_scores.append(l_acc)
            loop_errs.append(l_err)
            deltas.append(d_acc)

            print(f"Task: {task:<15} | Base (K=0): {b_acc:5.1f}% (+/-{b_err:4.1f}%) | Dual-Loop (K=2): {l_acc:5.1f}% (+/-{l_err:4.1f}%) | Delta: {d_acc:+5.1f}%")

            for s in t_data.get("samples_detail", []):
                total_samples += 1
                b_ok = s["base_norm_ok"]
                l_ok = s["loop_norm_ok"]
                if b_ok and l_ok:
                    total_both_correct += 1
                elif not b_ok and not l_ok:
                    total_both_wrong += 1
                elif not b_ok and l_ok:
                    total_improved += 1
                    all_rescued_questions.append({
                        "task": task,
                        "idx": s["idx"],
                        "question": s["question"],
                        "target_text": str(s["target"])
                    })
                else:
                    total_degraded += 1
    else:
        # Fallback to older lm-eval keys if needed
        pass

    mean_base = float(np.mean(base_scores))
    mean_loop = float(np.mean(loop_scores))
    mean_delta = mean_loop - mean_base

    print("-" * 90)
    print(f"OVERALL SUITE MEAN:   Base (K=0): {mean_base:5.1f}% | Dual-Loop (K=2): {mean_loop:5.1f}% | Net Delta: {mean_delta:+5.1f}%")
    print(f"DECISION TRANSITIONS: Both Correct: {total_both_correct} | Rescued: {total_improved} | Both Wrong: {total_both_wrong} | Degraded: {total_degraded}")
    print("=" * 90)

    print(f"\n[+] Total Questions Rescued by Dual-Loop Deliberation: {len(all_rescued_questions)}")
    for q in all_rescued_questions:
        print(f"  [{q['task']}] #{q['idx']} {q['question'][:75]}... -> Answer: {q['target_text']}")

    # --- Render Publication Scorecard Chart ---
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9), facecolor="#0D1117")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1.0], hspace=0.38, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#161B22")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#30363D")
        ax.spines["bottom"].set_color("#30363D")
        ax.grid(axis="y", linestyle="--", alpha=0.2, color="#8B949E")

    # Panel 1: Multi-Task Side-by-Side Barchart
    x = np.arange(len(TASKS))
    bar_w = 0.32

    labels = [TASK_DISPLAY_NAMES.get(t, t) for t in TASKS]
    r1 = ax1.bar(x - bar_w/2, base_scores, bar_w, yerr=base_errs, capsize=5,
                 label="Base Qwen3.5-2B (K=0)", color="#4B5563", edgecolor="#9CA3AF", linewidth=1.2)
    r2 = ax1.bar(x + bar_w/2, loop_scores, bar_w, yerr=loop_errs, capsize=5,
                 label="Dual-Loop Controller (K=2 Trained)", color="#0284C7", edgecolor="#38BDF8", linewidth=1.2)

    ax1.set_ylabel("Normalized Accuracy (%)", fontsize=11, color="#C9D1D9")
    ax1.set_title("Authentic EleutherAI Multi-Benchmark Scorecard: Qwen3.5-2B (N=40 samples/task, +/-1 SE)",
                  fontsize=13, fontweight="bold", color="#F0F6FC", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11, color="#C9D1D9", fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper left", framealpha=0.6, facecolor="#161B22", edgecolor="#30363D")

    for i, (b_rect, l_rect) in enumerate(zip(r1, r2)):
        bh = b_rect.get_height()
        lh = l_rect.get_height()
        ax1.text(b_rect.get_x() + b_rect.get_width()/2., bh + 4.5, f"{bh:.1f}%", ha="center", va="bottom", fontsize=10, color="#E5E7EB", fontweight="bold")
        ax1.text(l_rect.get_x() + l_rect.get_width()/2., lh + 4.5, f"{lh:.1f}%", ha="center", va="bottom", fontsize=10, color="#38BDF8", fontweight="bold")
        d = deltas[i]
        d_color = "#10B981" if d > 0 else "#EF4444" if d < 0 else "#8B949E"
        ax1.text(x[i], max(bh, lh) + 12.0, f"Delta {d:+.1f}%", ha="center", va="bottom", fontsize=10, color=d_color, fontweight="bold")

    # Panel 2: Overall Mean Score Comparison
    overall_x = [0, 1]
    overall_scores = [mean_base, mean_loop]
    overall_colors = ["#4B5563", "#0284C7"]
    overall_bars = ax2.bar(overall_x, overall_scores, width=0.45, color=overall_colors, edgecolor=["#9CA3AF", "#38BDF8"], linewidth=1.2)
    ax2.set_xticks(overall_x)
    ax2.set_xticklabels(["Base Model\n(K=0)", "Dual-Loop\n(K=2 Trained)"], fontsize=11, color="#C9D1D9", fontweight="bold")
    ax2.set_ylabel("Average Accuracy (%)", fontsize=11, color="#C9D1D9")
    ax2.set_title(f"Overall Suite Mean: Delta {mean_delta:+.1f}%", fontsize=12, fontweight="bold", color="#F0F6FC", pad=12)
    ax2.set_ylim(0, 100)

    for bar in overall_bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 4.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=11, color="#F0F6FC", fontweight="bold")

    # Panel 3: Decision Transition Breakdown
    trans_counts = [total_both_correct, total_improved, total_both_wrong, total_degraded]
    trans_labels = [
        f"Invariant Correct ({total_both_correct})",
        f"Rescued by Dual-Loop ({total_improved})",
        f"Invariant Wrong ({total_both_wrong})",
        f"Degraded ({total_degraded})"
    ]
    trans_colors = ["#10B981", "#38BDF8", "#6B7280", "#EF4444"]
    y_pos = np.arange(len(trans_labels))
    b3 = ax3.barh(y_pos, trans_counts, color=trans_colors, edgecolor="#1E293B", height=0.55)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(trans_labels, fontsize=10, color="#C9D1D9")
    ax3.set_xlabel(f"Total Samples (N={total_samples})", fontsize=11, color="#C9D1D9")
    ax3.set_title("Decision Transitions Across Full Suite", fontsize=12, fontweight="bold", color="#F0F6FC", pad=12)
    ax3.set_xlim(0, max(max(trans_counts) + 15, 30))
    ax3.grid(axis="x", linestyle="--", alpha=0.2, color="#8B949E")

    for bar in b3:
        w = bar.get_width()
        pct = (w / total_samples) * 100.0 if total_samples > 0 else 0
        ax3.text(w + 1.0, bar.get_y() + bar.get_height()/2., f"{w} ({pct:.0f}%)", va="center", fontsize=10, color="#F0F6FC", fontweight="bold")

    plt.suptitle("Authentic Multi-Benchmark Empirical Audit: Qwen3.5-2B + Dual-Loop Controller",
                 fontsize=16, fontweight="bold", color="#F0F6FC", y=0.98)

    plt.savefig(OUTPUT_PNG, dpi=300, facecolor="#0D1117", bbox_inches="tight")
    print(f"\n[+] Full benchmark suite chart saved successfully to: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
