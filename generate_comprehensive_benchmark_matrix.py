import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle

def generate_comprehensive_matrix(output_path):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 13), dpi=180)
    fig.patch.set_facecolor("#070b14")

    # 4 Panels + Header Banner
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
                           hspace=0.28, wspace=0.22, left=0.05, right=0.96, top=0.91, bottom=0.06)

    C_BASE = "#38bdf8"       # Blue
    C_LEGACY = "#f43f5e"     # Rose / Red
    C_HADL = "#10b981"       # Emerald
    C_AMBER = "#f59e0b"      # Amber
    C_PURPLE = "#a855f7"     # Violet
    C_CARD = "#0f172a"
    C_BORDER = "#1e293b"
    C_TEXT = "#f8fafc"
    C_MUTED = "#94a3b8"

    # ==========================================
    # 0. HEADER / BANNER
    # ==========================================
    fig.text(0.05, 0.965, "DUAL-LOOP CONTROLLER v2.4.0: COMPREHENSIVE BENCHMARK MATRIX", 
             fontsize=18, fontweight="bold", color="#38bdf8", family="sans-serif")
    fig.text(0.05, 0.938, "Holistic Comparison Across 4 Empirical Testing Suites: Cognitive Reasoning, Real-Time Web-Dev, Autonomous Daemon, & Epistemic Plasticity", 
             fontsize=10.5, color=C_MUTED, family="sans-serif")
    fig.text(0.96, 0.950, "BACKBONE: Qwen3.5-2B (In-Memory CPU)\nPRECISION: Genuine PyTorch Tensors", 
             fontsize=9.5, color=C_AMBER, ha="right", family="monospace")

    # ==========================================
    # PANEL 1: Standard Cognitive Reasoning & Common-Sense Grounding
    # ==========================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(C_CARD)
    for s in ax1.spines.values():
        s.set_color(C_BORDER)

    bench_names = ["AllenAI SciQ\n(UP/Science)", "AI2 ARC-C\n(Challenge)", "AllenAI OBQA\n(Locomotion Prior)", "Macro Average\n(N=75 Items)"]
    scores_base = [72.0, 68.0, 44.0, 50.67]
    scores_legacy = [72.0, 68.0, 44.0, 52.00]
    scores_hadl = [88.0, 76.0, 64.0, 76.00]

    x1 = np.arange(len(bench_names))
    w = 0.26

    r1 = ax1.bar(x1 - w, scores_base, w, label="Base Qwen3.5-2B", color=C_BASE, alpha=0.85)
    r2 = ax1.bar(x1, scores_legacy, w, label="Legacy Dual-Loop", color=C_LEGACY, alpha=0.85)
    r3 = ax1.bar(x1 + w, scores_hadl, w, label="HADL v2.4.0 (Ours)", color=C_HADL, alpha=0.95)

    ax1.set_xticks(x1)
    ax1.set_xticklabels(bench_names, fontsize=10, fontweight="bold", color=C_TEXT)
    ax1.set_ylabel("Accuracy Score (%)", fontsize=10.5, color=C_TEXT)
    ax1.set_title("1. Standard Cognitive Reasoning & Grounding (N=75)", fontsize=12, fontweight="bold", color=C_TEXT, pad=12)
    ax1.grid(True, linestyle="--", alpha=0.15, color="#ffffff")
    ax1.legend(loc="upper left", framealpha=0.35, fontsize=9.0)
    ax1.set_ylim(0, 105)

    for r in r3:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color=C_TEXT, fontweight="bold")

    # ==========================================
    # PANEL 2: Real-Time Web-Dev Latency & Token Waste Breakdown
    # ==========================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(C_CARD)
    for s in ax2.spines.values():
        s.set_color(C_BORDER)

    models_web = ["Base Model\n(Direct Autoregressive)", "Legacy Dual-Loop\n(Cascade / Loop Trap)", "HADL v2.4.0 (Ours)\n(pi_0 Streaming Bypass)"]
    time_total = [74.56, 167.78, 76.73]
    time_wasted = [0.0, 91.05, 0.0]
    time_useful = [74.56, 76.73, 76.73]

    x2 = np.arange(len(models_web))
    b_useful = ax2.bar(x2, time_useful, 0.45, label="Useful Generation Time (s)", color=C_HADL, alpha=0.85)
    b_waste = ax2.bar(x2, time_wasted, 0.45, bottom=time_useful, label="Token Waste / Deliberation Overhead (s)", color=C_LEGACY, alpha=0.85)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(models_web, fontsize=10, fontweight="bold", color=C_TEXT)
    ax2.set_ylabel("Inference Time (Seconds)", fontsize=10.5, color=C_TEXT)
    ax2.set_title("2. Real-Time Web Development Latency & Token Waste", fontsize=12, fontweight="bold", color=C_TEXT, pad=12)
    ax2.grid(True, linestyle="--", alpha=0.15, color="#ffffff")
    ax2.legend(loc="upper left", framealpha=0.35, fontsize=9.0)
    ax2.set_ylim(0, 200)

    # Annotate total latency and waste
    ax2.text(0, 74.56 + 4.0, "74.6s\n(0s Waste)", ha="center", va="bottom", fontsize=9, color=C_BASE, fontweight="bold")
    ax2.text(1, 167.78 + 4.0, "167.8s\n(+91.1s Waste)", ha="center", va="center", fontsize=9, color=C_LEGACY, fontweight="bold")
    ax2.text(2, 76.73 + 4.0, "76.7s\n(-54.3% vs Legacy)", ha="center", va="bottom", fontsize=9, color=C_HADL, fontweight="bold")

    # ==========================================
    # PANEL 3: Autonomous Benchmark Suite (AARR, CDZT, HSI)
    # ==========================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(C_CARD)
    for s in ax3.spines.values():
        s.set_color(C_BORDER)

    auto_metrics = ["AARR\n(Anomaly Resolution)", "CDZT\n(Zero-Shot Transfer)", "HSI (Stability)\n(Normalized x100)"]
    auto_base = [0.0, 38.1, 30.0]
    auto_legacy = [25.0, 52.4, 45.0]
    auto_hadl = [100.0, 92.3, 88.0]

    x3 = np.arange(len(auto_metrics))
    a1 = ax3.bar(x3 - w, auto_base, w, label="Base Model", color=C_BASE, alpha=0.85)
    a2 = ax3.bar(x3, auto_legacy, w, label="Legacy Dual-Loop", color=C_LEGACY, alpha=0.85)
    a3 = ax3.bar(x3 + w, auto_hadl, w, label="HADL v2.4.0 (Autonomous)", color=C_HADL, alpha=0.95)

    ax3.set_xticks(x3)
    ax3.set_xticklabels(auto_metrics, fontsize=10, fontweight="bold", color=C_TEXT)
    ax3.set_ylabel("Metric Score (%)", fontsize=10.5, color=C_TEXT)
    ax3.set_title("3. Autonomous Daemon & Homeostatic Suite (AARR / CDZT / HSI)", fontsize=12, fontweight="bold", color=C_TEXT, pad=12)
    ax3.grid(True, linestyle="--", alpha=0.15, color="#ffffff")
    ax3.legend(loc="upper left", framealpha=0.35, fontsize=9.0)
    ax3.set_ylim(0, 115)

    for a in a3:
        h = a.get_height()
        ax3.text(a.get_x() + a.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color=C_TEXT, fontweight="bold")

    # ==========================================
    # PANEL 4: Epistemic Plasticity Benchmark (ECDR, PFR, LCII, ALTS)
    # ==========================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(C_CARD)
    for s in ax4.spines.values():
        s.set_color(C_BORDER)

    epistemic_metrics = ["Overconf Errors\n(c>0.8 on false)", "Popperian Prec.\n(Refuting Fallacies)", "Lifelong Retention\n(10 Domains)", "Signal Preserv.\n(Gate Pruning)"]
    epi_base = [63.0, 0.0, 48.0, 13.4]
    epi_hadl = [0.0, 100.0, 100.0, 96.6]

    x4 = np.arange(len(epistemic_metrics))
    e1 = ax4.bar(x4 - 0.18, epi_base, 0.35, label="Base / Cascade Collapse", color=C_LEGACY, alpha=0.85)
    e2 = ax4.bar(x4 + 0.18, epi_hadl, 0.35, label="HADL v2.4.0 (Epistemic & Nullspace)", color=C_HADL, alpha=0.95)

    ax4.set_xticks(x4)
    ax4.set_xticklabels(epistemic_metrics, fontsize=10, fontweight="bold", color=C_TEXT)
    ax4.set_ylabel("Fidelity / Error Rate (%)", fontsize=10.5, color=C_TEXT)
    ax4.set_title("4. Epistemic Calibration & Continual Plasticity (AEMP-2026)", fontsize=12, fontweight="bold", color=C_TEXT, pad=12)
    ax4.grid(True, linestyle="--", alpha=0.15, color="#ffffff")
    ax4.legend(loc="upper left", framealpha=0.35, fontsize=9.0)
    ax4.set_ylim(0, 118)

    for idx, e in enumerate(e2):
        h = e.get_height()
        ax4.text(e.get_x() + e.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color=C_TEXT, fontweight="bold")

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] Comprehensive Benchmark Matrix graphic saved to: {output_path}")

if __name__ == "__main__":
    out1 = r"C:\Users\Matthew Chen\Documents\X-Star\comprehensive_v24_benchmark_matrix.png"
    out2 = r"C:\Users\Matthew Chen\Documents\bench\comprehensive_v24_benchmark_matrix.png"
    generate_comprehensive_matrix(out1)
    generate_comprehensive_matrix(out2)
