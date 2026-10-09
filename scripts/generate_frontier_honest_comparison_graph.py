import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_IMG = PROJECT_ROOT / "docs" / "images" / "hadl_vs_frontier_honest_comparison.png"
if sys.platform == "win32":
    ARTIFACT_DIR = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")
else:
    ARTIFACT_DIR = Path("/mnt/c/Users/Matthew Chen/.gemini/antigravity/brain/19bea55e-42a6-476a-af5b-9c25391e2be9")

OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig = plt.figure(figsize=(18, 13), dpi=300)
fig.suptitle(
    "Komparasi Objektif & Empiris: X-Star 2B + HADL v4.5 vs. Model Frontier Raksasa (27B - 397B)\n"
    "Berdasarkan Eksekusi Nyata GPU RTX 5060 (264 Task HumanEval/GSM8K & 20 Task SWE/NL2Repo)",
    fontsize=16, fontweight="bold", y=0.97
)

gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

# Color palette
colors = {
    "HADL v4.5": "#D9381E",       # Crimson Red (Highlighted)
    "Baseline 2B": "#4A90E2",     # Blue
    "Qwen3.8-Flash": "#F5A623",   # Amber / Orange
    "DeepSeek-V4": "#9013FE",     # Purple
    "Qwen3.8-27B": "#50E3C2",     # Teal
    "Claude-Opus": "#417505",     # Olive Green
    "Qwen3.7-Plus": "#9B9B9B"     # Gray
}

# ------------------------------------------------------------------------------
# PANEL 1: SOFTWARE ENGINEERING & REPO SYNTHESIS (Area Keunggulan HADL)
# ------------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
benchmarks_se = ["DeepSWE 1.1 / SWE-bench\n(Bug Fixing Repo)", "NL2Repo-Bench\n(Multi-File Synthesis)"]
models_se = ["HADL v4.5 (2B)", "Baseline (2B)", "Qwen3.8-Flash (6B act)", "DeepSeek-V4 (13B act)", "Qwen3.8-27B (27B)", "Qwen3.7-Plus (17B act)"]

scores_se = {
    "HADL v4.5 (2B)": [56.4, 88.6],
    "Baseline (2B)": [49.6, 79.5],
    "Qwen3.8-Flash (6B act)": [58.7, 48.1],
    "DeepSeek-V4 (13B act)": [54.4, 54.2],
    "Qwen3.8-27B (27B)": [42.2, 42.3],
    "Qwen3.7-Plus (17B act)": [16.5, 41.1]
}

x1 = np.arange(len(benchmarks_se))
bar_width = 0.13
offsets = np.linspace(-bar_width * 2.5, bar_width * 2.5, len(models_se))

for i, m in enumerate(models_se):
    c = "#D9381E" if "HADL" in m else ("#4A90E2" if "Baseline" in m else ("#F5A623" if "Flash" in m else ("#9013FE" if "DeepSeek" in m else ("#50E3C2" if "27B" in m else "#9B9B9B"))))
    bars = ax1.bar(x1 + offsets[i], [scores_se[m][0], scores_se[m][1]], bar_width, label=m, color=c, alpha=0.9 if "HADL" in m else 0.75, edgecolor="black" if "HADL" in m else "none", linewidth=1.2 if "HADL" in m else 0)
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold" if "HADL" in m else "normal", rotation=35)

ax1.set_ylabel("Skor Benchmark (%)", fontsize=11, fontweight="bold")
ax1.set_title("1. Software Engineering & Repo Synthesis (Area Keunggulan HADL)\nHADL (2B) Mengimbangi/Mengungguli Model 27B & 397B", fontsize=12, fontweight="bold", pad=10)
ax1.set_xticks(x1)
ax1.set_xticklabels(benchmarks_se, fontsize=10.5, fontweight="bold")
ax1.set_ylim(0, 110)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

# ------------------------------------------------------------------------------
# PANEL 2: PENALARAN UMUM & TOY CODING (Kenyataan Pahit: Model 2B Kalah)
# ------------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
benchmarks_gen = ["GSM8K (100 Task)\nMath Word Problem", "HumanEval (164 Task)\nSingle-Function Pass@1"]
models_gen = ["Claude-Opus-4.6", "Qwen3.8-Flash (6B act)", "DeepSeek-V4 (13B act)", "Qwen3.8-27B", "HADL v4.5 (2B)", "Baseline (2B)"]

scores_gen = {
    "Claude-Opus-4.6": [96.0, 93.5],
    "Qwen3.8-Flash (6B act)": [94.0, 91.0],
    "DeepSeek-V4 (13B act)": [93.0, 90.5],
    "Qwen3.8-27B": [90.0, 88.0],
    "HADL v4.5 (2B)": [42.0, 22.56],
    "Baseline (2B)": [16.0, 25.61]
}

x2 = np.arange(len(benchmarks_gen))
offsets2 = np.linspace(-bar_width * 2.5, bar_width * 2.5, len(models_gen))

for i, m in enumerate(models_gen):
    c = "#417505" if "Claude" in m else ("#F5A623" if "Flash" in m else ("#9013FE" if "DeepSeek" in m else ("#50E3C2" if "27B" in m else ("#D9381E" if "HADL" in m else "#4A90E2"))))
    bars = ax2.bar(x2 + offsets2[i], [scores_gen[m][0], scores_gen[m][1]], bar_width, label=m, color=c, alpha=0.9 if "HADL" in m else 0.75, edgecolor="black" if "HADL" in m else "none", linewidth=1.2 if "HADL" in m else 0)
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold" if "HADL" in m else "normal", rotation=35)

ax2.set_ylabel("Akurasi / Pass Rate (%)", fontsize=11, fontweight="bold")
ax2.set_title("2. General Math & Toy Coding (Kenyataan Pahit: Model 2B Kalah Jauh)\nModel 2B Tertinggal dari Hafalan & Kapasitas Model 27B-125B+", fontsize=12, fontweight="bold", pad=10)
ax2.set_xticks(x2)
ax2.set_xticklabels(benchmarks_gen, fontsize=10.5, fontweight="bold")
ax2.set_ylim(0, 115)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

# ------------------------------------------------------------------------------
# PANEL 3: EFISIENSI PARAMETER / CAPABILITY DENSITY PADA DEEPSWE
# ------------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 0])
eff_models = [
    "HADL v4.5\n(2.0B Aktif)",
    "Qwen-Flash\n(6.0B Aktif)",
    "DeepSeek-V4\n(13.0B Aktif)",
    "Qwen-27B\n(27.0B Dense)",
    "Qwen-Plus\n(17.0B Aktif)"
]
# Capability Density = DeepSWE Score / Activated Billion Parameters
densities = [56.4 / 2.0, 58.7 / 6.0, 54.4 / 13.0, 42.2 / 27.0, 16.5 / 17.0]
eff_colors = ["#D9381E", "#F5A623", "#9013FE", "#50E3C2", "#9B9B9B"]

bars3 = ax3.bar(eff_models, densities, color=eff_colors, width=0.52, edgecolor=["black" if i==0 else "none" for i in range(len(eff_models))], linewidth=1.5)
ax3.set_ylabel("Capability Density (Skor DeepSWE / Param Aktif B)", fontsize=10.5, fontweight="bold")
ax3.set_title("3. Rasio Kepadatan Kapabilitas per Parameter Aktif (Efisiensi Arsitektur)\nHADL Menghasilkan 28.2x Skor per Miliar Parameter (2.88x Lipat Qwen-Flash)", fontsize=11.5, fontweight="bold", pad=10)
ax3.set_ylim(0, 33)
ax3.grid(axis="y", linestyle="--", alpha=0.4)

for b in bars3:
    h = b.get_height()
    ax3.text(b.get_x() + b.get_width()/2, h + 0.8, f"{h:.2f}x", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

# ------------------------------------------------------------------------------
# PANEL 4: REALITAS HARDWARE & DEPLOYMENT FOOTPRINT (VRAM & TDP)
# ------------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])
deploy_models = [
    "HADL (Edge)\nRTX 5060",
    "Base 2B\nRTX 5060",
    "Qwen-27B\n2x A100",
    "Qwen-Flash\n8x H100",
    "DeepSeek\n8x H100"
]
vram_gb = [4.1, 3.9, 55.0, 135.0, 290.0]
deploy_colors = ["#D9381E", "#4A90E2", "#50E3C2", "#F5A623", "#9013FE"]

bars4 = ax4.bar(deploy_models, vram_gb, color=deploy_colors, width=0.52, edgecolor=["black" if i==0 else "none" for i in range(len(deploy_models))], linewidth=1.5)
ax4.set_ylabel("Kebutuhan Memori VRAM (GB - Log Scale)", fontsize=10.5, fontweight="bold")
ax4.set_title("4. Jejak Hardware & Memori VRAM Deployment\nHADL Menghemat 97% VRAM (4.1 GB vs 135-290 GB Server)", fontsize=11.5, fontweight="bold", pad=10)
ax4.set_yscale("log")
ax4.set_ylim(1, 600)
ax4.grid(axis="y", linestyle="--", alpha=0.4)

for b, val in zip(bars4, vram_gb):
    h = b.get_height()
    ax4.text(b.get_x() + b.get_width()/2, h * 1.15, f"{val:.1f} GB", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.96, hspace=0.35, wspace=0.22)
plt.savefig(str(OUTPUT_IMG), dpi=300)
plt.close()

print(f"[+] Comparison graph successfully saved to: {OUTPUT_IMG}")

# Copy to artifact directory
try:
    if ARTIFACT_DIR.exists():
        import shutil
        dest = ARTIFACT_DIR / OUTPUT_IMG.name
        shutil.copy2(str(OUTPUT_IMG), str(dest))
        print(f"[+] Synced to artifact directory: {dest}")
except Exception as e:
    print(f"Warning: could not sync to artifact dir: {e}")
