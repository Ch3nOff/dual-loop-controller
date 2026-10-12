#!/usr/bin/env python3
"""
Plot Autonomous Brain Benchmark Comparison
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_OUT = PROJECT_ROOT / "eval_results" / "autonomous_brain_benchmark_comparison.png"
IMG_OUT.parent.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Graph 1: Evolution of Architecture Accuracy
models = [
    "1. Base Model\n(Frozen 2B)",
    "2. Model Sebelumnya\n(External Hook HADL)",
    "3. Fusi Naif Statis\n(Linear Additive Fail)",
    "4. Model Otonom Sekarang\n(Native Self-Evolving Brain)"
]
scores = [50.0, 90.0, 42.0, 90.0]
colors = ["#78909C", "#FF7043", "#E53935", "#2E7D32"]

bars1 = ax1.bar(models, scores, color=colors, width=0.55)
ax1.set_title("Evolusi Akurasi: Eksternal Wrapper vs Native Internal Brain (50 Soal)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Akurasi Total (%)", fontsize=10)
ax1.set_ylim(0, 105)

for b, s in zip(bars1, scores):
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2., h + 1.5, f"{s:.1f}%\n({int(s*0.5)}/50)", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Graph 2: Breakdown per Pillar Keilmuan (Base vs Autonomous Native Brain)
pillars = ["Math &\nQuant", "Science &\nAcademic", "Coding &\nSoftware", "Commonsense\n& Logic", "Instruction\n& Chat"]
base_pillar = [22.2, 50.0, 42.9, 81.8, 44.4]
auto_pillar = [77.8, 92.9, 85.7, 90.9, 100.0]

x = np.arange(len(pillars))
w = 0.35

ax2.bar(x - w/2, base_pillar, width=w, label="Base Model (Frozen)", color="#78909C")
ax2.bar(x + w/2, auto_pillar, width=w, label="Autonomous Native Brain (90%)", color="#2E7D32")

ax2.set_title("Breakdown 5 Pillar: Native Brain vs Model Dasar", fontsize=11, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(pillars, fontsize=9)
ax2.set_ylabel("Akurasi (%)", fontsize=10)
ax2.set_ylim(0, 115)
ax2.legend(loc="upper left", fontsize=9)

for i in range(len(pillars)):
    ax2.text(x[i] + w/2, auto_pillar[i] + 1.5, f"{auto_pillar[i]:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1B5E20")

plt.tight_layout()
plt.savefig(str(IMG_OUT), dpi=300)
plt.close()
print(f"[+] Saved comparison plot to: {IMG_OUT}")
