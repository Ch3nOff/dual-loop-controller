import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_JSON = "eval_results/qwen35_2b_base_k0.json"
LOOP_JSON = "eval_results/qwen35_2b_dualloop_k2.json"
OUTPUT_PNG = "qwen35_2b_real_benchmark.png"
TASK = "arc_easy"

def main():
    if not os.path.exists(BASE_JSON) or not os.path.exists(LOOP_JSON):
        print(f"[!] Target evaluation files not found:\n    {BASE_JSON}\n    {LOOP_JSON}")
        sys.exit(1)

    with open(BASE_JSON, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    with open(LOOP_JSON, "r", encoding="utf-8") as f:
        loop_data = json.load(f)

    base_res = base_data["results"][TASK]
    loop_res = loop_data["results"][TASK]

    base_acc = base_res["acc,none"] * 100.0
    base_acc_err = base_res.get("acc_stderr,none", 0.0) * 100.0
    base_norm = base_res["acc_norm,none"] * 100.0
    base_norm_err = base_res.get("acc_norm_stderr,none", 0.0) * 100.0

    loop_acc = loop_res["acc,none"] * 100.0
    loop_acc_err = loop_res.get("acc_stderr,none", 0.0) * 100.0
    loop_norm = loop_res["acc_norm,none"] * 100.0
    loop_norm_err = loop_res.get("acc_norm_stderr,none", 0.0) * 100.0

    print("=" * 80)
    print(" AUTHENTIC BENCHMARK RESULTS: QWEN3.5-2B (ARC-EASY, N=50)")
    print("=" * 80)
    print(f"Base Model (K=0):     Raw Acc = {base_acc:.2f}% (+/-{base_acc_err:.2f}%), Norm Acc = {base_norm:.2f}% (+/-{base_norm_err:.2f}%)")
    print(f"Dual-Loop (K=2):      Raw Acc = {loop_acc:.2f}% (+/-{loop_acc_err:.2f}%), Norm Acc = {loop_norm:.2f}% (+/-{loop_norm_err:.2f}%)")
    print(f"Raw Delta:            {loop_acc - base_acc:+.2f}%")
    print(f"Normalized Delta:     {loop_norm - base_norm:+.2f}%")
    print("=" * 80)

    # Sample-level forensic analysis
    base_samples = base_data.get("samples", {}).get(TASK, [])
    loop_samples = loop_data.get("samples", {}).get(TASK, [])

    both_correct = 0
    both_wrong = 0
    improved = 0
    degraded = 0
    target_ll_shifts = []
    discrepancies = []

    for i, (b, l) in enumerate(zip(base_samples, loop_samples)):
        target = b["target"]
        b_resps = b.get("filtered_resps", [])
        l_resps = l.get("filtered_resps", [])

        b_logprobs = [r[0] for r in b_resps]
        l_logprobs = [r[0] for r in l_resps]

        b_pred = int(np.argmax(b_logprobs)) if b_logprobs else -1
        l_pred = int(np.argmax(l_logprobs)) if l_logprobs else -1

        b_ok = (b_pred == target)
        l_ok = (l_pred == target)

        if b_ok and l_ok:
            both_correct += 1
        elif not b_ok and not l_ok:
            both_wrong += 1
        elif not b_ok and l_ok:
            improved += 1
        else:
            degraded += 1

        shift = (l_logprobs[target] - b_logprobs[target]) if (b_logprobs and l_logprobs) else 0.0
        target_ll_shifts.append(shift)

        if b_pred != l_pred:
            discrepancies.append({
                "sample_idx": i,
                "doc_id": b.get("doc_id", i),
                "question": b.get("doc", {}).get("question", ""),
                "choices": b.get("doc", {}).get("choices", {}).get("text", []),
                "target": target,
                "b_pred": b_pred,
                "l_pred": l_pred,
                "b_logprobs": [round(x, 3) for x in b_logprobs],
                "l_logprobs": [round(x, 3) for x in l_logprobs],
                "b_correct": b_ok,
                "l_correct": l_ok
            })

    target_ll_shifts = np.array(target_ll_shifts)

    print(f"\n[+] Decision Transition Matrix (N={len(base_samples)}):")
    print(f"    Both Correct (Invariant): {both_correct} ({both_correct/len(base_samples)*100:.1f}%)")
    print(f"    Both Wrong (Invariant):   {both_wrong} ({both_wrong/len(base_samples)*100:.1f}%)")
    print(f"    Dual-Loop Fixed:          {improved} ({improved/len(base_samples)*100:.1f}%)")
    print(f"    Dual-Loop Degraded:       {degraded} ({degraded/len(base_samples)*100:.1f}%)")
    print(f"\n[+] Mean Target Log-Likelihood Shift: {np.mean(target_ll_shifts):+.4f} nats")

    if discrepancies:
        print(f"\n[+] Sample Discrepancies Count: {len(discrepancies)}")
        for d in discrepancies:
            print(f"\n--- Sample #{d['sample_idx']} (Doc ID: {d['doc_id']}) ---")
            print(f"Question: {d['question']}")
            for idx, c in enumerate(d['choices']):
                mark = " [CORRECT ANSWER]" if idx == d["target"] else ""
                print(f"  [{chr(65+idx)}] {c}{mark}")
            print(f"  Base (K=0):     Choice {chr(65+d['b_pred'])} ({'CORRECT' if d['b_correct'] else 'WRONG'}) | Logprobs: {d['b_logprobs']}")
            print(f"  Dual-Loop (K=2): Choice {chr(65+d['l_pred'])} ({'CORRECT' if d['l_correct'] else 'WRONG'}) | Logprobs: {d['l_logprobs']}")

    # --- Render Publication Chart ---
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 9), facecolor="#0D1117")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#161B22")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#30363D")
        ax.spines["bottom"].set_color("#30363D")
        ax.grid(axis="y", linestyle="--", alpha=0.2, color="#8B949E")

    # Subplot 1: Bar Chart of Accuracy
    categories = ["Raw Accuracy (acc)", "Length-Norm (acc_norm)"]
    x = np.arange(len(categories))
    bar_width = 0.32

    rects1 = ax1.bar(x - bar_width/2, [base_acc, base_norm], bar_width,
                     yerr=[base_acc_err, base_norm_err], capsize=5,
                     label="Base Qwen3.5-2B (K=0)", color="#4B5563", edgecolor="#9CA3AF", linewidth=1.2)
    rects2 = ax1.bar(x + bar_width/2, [loop_acc, loop_norm], bar_width,
                     yerr=[loop_acc_err, loop_norm_err], capsize=5,
                     label="Dual-Loop Controller (K=2)", color="#0284C7", edgecolor="#38BDF8", linewidth=1.2)

    ax1.set_ylabel("Accuracy (%)", fontsize=11, color="#C9D1D9")
    ax1.set_title("ARC-Easy Measured Accuracy (N=50, +/-1 std err)", fontsize=13, fontweight="bold", color="#F0F6FC", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, color="#C9D1D9")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left", framealpha=0.6, facecolor="#161B22", edgecolor="#30363D")

    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 5.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, color="#E5E7EB", fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 5.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, color="#38BDF8", fontweight="bold")

    # Subplot 2: Transition breakdown
    trans_counts = [both_correct, both_wrong, improved, degraded]
    trans_labels = [
        f"Invariant Correct ({both_correct})",
        f"Invariant Wrong ({both_wrong})",
        f"Dual-Loop Fixed ({improved})",
        f"Dual-Loop Degraded ({degraded})"
    ]
    trans_colors = ["#10B981", "#6B7280", "#0284C7", "#EF4444"]

    y_pos = np.arange(len(trans_labels))
    bars2 = ax2.barh(y_pos, trans_counts, color=trans_colors, edgecolor="#1E293B", height=0.55)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(trans_labels, fontsize=10, color="#C9D1D9")
    ax2.set_xlabel(f"Sample Count (N={len(base_samples)})", fontsize=11, color="#C9D1D9")
    ax2.set_title("Decision Transition Matrix (K=0 vs K=2)", fontsize=13, fontweight="bold", color="#F0F6FC", pad=12)
    ax2.set_xlim(0, max(max(trans_counts) + 6, 20))
    ax2.grid(axis="x", linestyle="--", alpha=0.2, color="#8B949E")

    for bar in bars2:
        w = bar.get_width()
        pct = (w / len(base_samples)) * 100.0 if len(base_samples) > 0 else 0
        ax2.text(w + 0.5, bar.get_y() + bar.get_height()/2., f"{w} ({pct:.0f}%)", va="center", fontsize=10, color="#F0F6FC", fontweight="bold")

    # Subplot 3: Per-Sample Log-Likelihood Shift
    sample_indices = np.arange(len(target_ll_shifts))
    bar_colors = ["#0284C7" if v > 0 else "#EF4444" if v < 0 else "#6B7280" for v in target_ll_shifts]

    ax3.bar(sample_indices, target_ll_shifts, color=bar_colors, width=0.7, edgecolor="none")
    ax3.axhline(0, color="#8B949E", linestyle="-", linewidth=1.0, alpha=0.5)
    mean_val = np.mean(target_ll_shifts) if len(target_ll_shifts) > 0 else 0.0
    ax3.axhline(mean_val, color="#F59E0B", linestyle="--", linewidth=1.5,
                label=f"Mean Shift: {mean_val:+.3f} nats")

    ax3.set_xlabel("Question Index (0 .. 49)", fontsize=11, color="#C9D1D9")
    ax3.set_ylabel("Delta Log-Likelihood (nats)", fontsize=11, color="#C9D1D9")
    ax3.set_title("Per-Sample Target Token Log-Likelihood Shift (Dual-Loop K=2 vs Base K=0)", fontsize=13, fontweight="bold", color="#F0F6FC", pad=12)
    ax3.set_xlim(-1, len(sample_indices))
    ax3.legend(loc="lower right", framealpha=0.4, facecolor="#161B22", edgecolor="#30363D")

    # Annotate discrepancies if any
    for d in discrepancies:
        idx = d["sample_idx"]
        val = target_ll_shifts[idx]
        is_fixed = d["l_correct"]
        txt = f"Q{idx} ({'Fixed' if is_fixed else 'Degraded'})"
        arrow_color = "#38BDF8" if is_fixed else "#EF4444"
        offset = 2.0 if val >= 0 else -2.0
        ax3.annotate(txt, xy=(idx, val), xytext=(idx, val + offset),
                     arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.2),
                     color="#E0E7FF" if is_fixed else "#FCA5A5", fontsize=8, fontweight="bold", ha="center")

    plt.suptitle("Authentic EleutherAI LM-Evaluation Harness Audit: Qwen3.5-2B + Dual-Loop Controller",
                 fontsize=16, fontweight="bold", color="#F0F6FC", y=0.98)

    plt.savefig(OUTPUT_PNG, dpi=300, facecolor="#0D1117", bbox_inches="tight")
    print(f"\n[+] High-res chart saved successfully to: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
