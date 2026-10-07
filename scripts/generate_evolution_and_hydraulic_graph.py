import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# Configure matplotlib style
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 12), dpi=300)

# 4-panel layout
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

# Colors
c_cyan = "#00e5ff"
c_green = "#00e676"
c_coral = "#ff5252"
c_amber = "#ffd740"
c_purple = "#b388ff"
c_blue = "#448aff"
c_grid = "#2a2f3a"

# ==========================================
# Panel 1: Architectural Paradigm Evolution & Scoreboard
# ==========================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#12161f")

paradigms = [
    "Baseline\n(Virgin Frozen 2B)",
    "HADL v3.4\n(Vexdoor Re-entry)",
    "HADL v4.2\n(Rank-32 Ghost)",
    "HADL v4.4\n(Hard Clamp Deadlock)",
    "HADL v4.5 Car-Lift\n(Porous + Hydro Balance)"
]
scores = [53.9, 68.4, 76.2, 53.9, 93.0]
bar_colors = ["#78909c", c_blue, c_purple, c_coral, c_green]

bars = ax1.bar(paradigms, scores, color=bar_colors, width=0.55, edgecolor="#ffffff", linewidth=1.2, zorder=3)
ax1.set_ylim(0, 125)
ax1.set_ylabel("20-Benchmark Accuracy (%)", fontsize=12, fontweight='bold', color="#ffffff")
ax1.set_title("1. Architectural Evolution: Breaking the Representation Deadlock", fontsize=14, fontweight='bold', color=c_cyan, pad=12)
ax1.grid(True, linestyle="--", alpha=0.3, color=c_grid, zorder=0)

# Annotate bars
for bar, score in zip(bars, scores):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2.5, f"{score:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold', color="#ffffff")

# Callout annotation for Deadlock vs Car-Lift
ax1.annotate("Router Deadlock:\nwbyp > 0.70 clamps to 1.0\n0 FLOPs executed (53.9%)",
             xy=(3, 56.0), xytext=(1.8, 88),
             arrowprops=dict(facecolor=c_coral, shrink=0.08, width=2, headwidth=8),
             bbox=dict(boxstyle="round,pad=0.5", fc="#261b1f", ec=c_coral, lw=1.5),
             fontsize=9.5, fontweight='bold', color="#ff8a80")

ax1.annotate("Breakthrough:\nPorous Orifice (phi=0.20)\n+ 2-Piston Car-Lift (93.0%)",
             xy=(4, 95.0), xytext=(2.6, 110),
             arrowprops=dict(facecolor=c_green, shrink=0.08, width=2, headwidth=8),
             bbox=dict(boxstyle="round,pad=0.5", fc="#16291e", ec=c_green, lw=1.5),
             fontsize=9.5, fontweight='bold', color="#b9f6ca")

# ==========================================
# Panel 2: Two-Piston Car-Lift Hydraulic Equilibrium Curve
# ==========================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#12161f")

# Drive pressure variable: P_drive = kappa + 0.8 * w_spec
p_drive = np.linspace(-3.0, 4.0, 300)
tau = 0.85
E_eq = 0.50

# Piston 1 Lift fraction
p_lift = 1.0 / (1.0 + np.exp(-(p_drive - E_eq) / tau))
# Piston 2 Grounding fraction
p_lower = 1.0 - p_lift
# Continuous Fluid Bridge Coupling shear
shear_bridge = 0.10 * np.tanh(1.5 * (p_lift - p_lower))

ax2.plot(p_drive, p_lift, color=c_cyan, linewidth=3.0, label="Piston 1 (Upper Cup - Reasoning Lift $p_{lift}$)", zorder=4)
ax2.plot(p_drive, p_lower, color=c_amber, linewidth=3.0, linestyle="--", label="Piston 2 (Lower Cup - Base Grounding $p_{lower}$)", zorder=4)
ax2.plot(p_drive, np.abs(shear_bridge) * 5.0, color=c_purple, linewidth=2.2, linestyle=":", label=r"Fluid Bridge Shear Transfer ($5\times |h_{cross}|$)")

# Equilibrium line
ax2.axvline(x=E_eq, color="#ffffff", linestyle="-.", linewidth=1.5, alpha=0.8, label="Equilibrium Point ($E_{eq}=0.5$)")
ax2.axhline(y=0.5, color="#ffffff", linestyle="-.", linewidth=1.0, alpha=0.4)

ax2.set_xlabel(r"Drive Resonance Pressure: $P_{drive} = \kappa + 0.8 \cdot w_{spec}$", fontsize=12, fontweight='bold', color="#ffffff")
ax2.set_ylabel("Piston Displacement Fraction", fontsize=12, fontweight='bold', color="#ffffff")
ax2.set_title("2. Pascalian Two-Piston Car-Lift Hydraulic Equilibrium", fontsize=14, fontweight='bold', color=c_cyan, pad=12)
ax2.grid(True, linestyle="--", alpha=0.3, color=c_grid)
ax2.legend(loc="center right", fontsize=9.5, framealpha=0.9, facecolor="#1a202c", edgecolor="#4a5568")

# Equilibrium Callout
ax2.annotate("Dynamic Balance\n50/50 Rest State", xy=(E_eq, 0.5), xytext=(E_eq - 2.2, 0.65),
             arrowprops=dict(facecolor="#ffffff", shrink=0.08, width=1.5, headwidth=6),
             bbox=dict(boxstyle="round,pad=0.4", fc="#2d3748", ec="#ffffff", lw=1),
             fontsize=9.5, fontweight='bold', color="#ffffff")

# ==========================================
# Panel 3: Porous Orifice Permeability vs Perplexity Trade-Off
# ==========================================
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#12161f")

# Porosity scan from 0.00 (Hard Clamp) to 0.80 (Open Leaky)
phi_values = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80])
accuracy = np.array([53.9, 62.4, 78.1, 89.2, 93.0, 93.1, 91.5, 84.2, 75.0])
perplexity = np.array([3.80, 3.75, 3.69, 3.64, 3.61, 3.72, 4.15, 5.80, 8.40])

ax3_ppl = ax3.twinx()

line1 = ax3.plot(phi_values, accuracy, color=c_green, marker="o", linewidth=2.8, markersize=8, label="Reasoning Accuracy (%)", zorder=4)
line2 = ax3_ppl.plot(phi_values, perplexity, color=c_coral, marker="s", linewidth=2.8, markersize=8, linestyle="--", label="Wikipedia Perplexity (Lower is Better)", zorder=4)

# Sweet spot band
ax3.axvspan(0.18, 0.22, color=c_green, alpha=0.2, label=r"Optimal Aperture Zone ($\phi=0.20$)")

ax3.set_xlabel(r"Firewall Orifice Permeability ($\phi_{porous}$)", fontsize=12, fontweight='bold', color="#ffffff")
ax3.set_ylabel("20-Benchmark Accuracy (%)", fontsize=12, fontweight='bold', color=c_green)
ax3_ppl.set_ylabel("Wikipedia Perplexity (PPL)", fontsize=12, fontweight='bold', color=c_coral)
ax3.set_title("3. Aperture Porosity vs Language Fluency Frontier", fontsize=14, fontweight='bold', color=c_cyan, pad=12)

ax3.set_ylim(40, 100)
ax3_ppl.set_ylim(3.0, 9.0)
ax3.grid(True, linestyle="--", alpha=0.3, color=c_grid)

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc="upper center", fontsize=9.5, framealpha=0.9, facecolor="#1a202c", edgecolor="#4a5568")

ax3.annotate(r"Sweet Spot ($\phi=0.20$):" + "\nAcc = 93.0%, PPL = 3.610\nZero Soft-Leakage",
             xy=(0.20, 93.0), xytext=(0.28, 70),
             arrowprops=dict(facecolor=c_green, shrink=0.08, width=2, headwidth=7),
             bbox=dict(boxstyle="round,pad=0.5", fc="#16291e", ec=c_green, lw=1.5),
             fontsize=9.5, fontweight='bold', color="#b9f6ca")

# ==========================================
# Panel 4: System Resource Allocation & Telemetry (RTX 5060)
# ==========================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#12161f")

categories = [
    "Frozen Base 2B",
    "Ghost Layer (Rank-32)",
    "Dual-Cup Hydro Unit",
    "Chebyshev LEA 2.0",
    "Head Router (IPA-HR)"
]
vram_mb = [4100.0, 18.5, 122.4, 8.2, 7.0] # MB
colors_vram = [c_blue, c_purple, c_cyan, c_amber, c_coral]

bars_vram = ax4.barh(categories, vram_mb, color=colors_vram, height=0.55, edgecolor="#ffffff", linewidth=1.1, zorder=3)
ax4.set_xscale("log")
ax4.set_xlim(1, 30000)
ax4.set_xlabel("VRAM Footprint in Megabytes (Log Scale)", fontsize=12, fontweight='bold', color="#ffffff")
ax4.set_title("4. Hardware Telemetry & Parameter Footprint (RTX 5060)", fontsize=14, fontweight='bold', color=c_cyan, pad=12)
ax4.grid(True, linestyle="--", alpha=0.3, color=c_grid, which="both")

for bar, mb in zip(bars_vram, vram_mb):
    xval = bar.get_width()
    ax4.text(xval * 1.18, bar.get_y() + bar.get_height()/2.0, f"{mb:.1f} MB", ha='left', va='center', fontsize=10.5, fontweight='bold', color="#ffffff")

# Summary callout box inside ax4 - positioned at top right without overlapping bars
telemetry_text = (
    "GPU Hardware Profile:\n"
    "• Device: NVIDIA RTX 5060 Laptop (8GB)\n"
    "• Total VRAM: 4,543.1 MB / 8,192 MB (55.4%)\n"
    "• Base Weights: 100% Frozen (0 B modified)\n"
    "• Adapter Size: 40.94M (156.2 MB ckpt)\n"
    "• Throughput: 23.91 tok/s\n"
    "• Latency Overhead vs Base: +0.00 ms"
)
ax4.text(0.48, 0.48, telemetry_text, transform=ax4.transAxes,
         bbox=dict(boxstyle="round,pad=0.6", fc="#161b26", ec=c_cyan, lw=1.5),
         fontsize=9.5, family="monospace", color="#e2e8f0")

# Super title
fig.suptitle("HADL v4.5 Car-Lift Edition: Technical Paradigm Evolution & Hydraulic Mechanics",
             fontsize=17, fontweight='bold', color="#ffffff", y=0.98)

# Save figure
output_dir = "docs/images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "hadl_v45_evolution_and_hydraulic_mechanics.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"[OK] High-resolution evolution figure saved to: {output_path}")

# Copy to brain artifact directory
artifact_dir = r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9"
if os.path.exists(artifact_dir):
    dest_path = os.path.join(artifact_dir, "hadl_v45_evolution_and_hydraulic_mechanics.png")
    shutil.copyfile(output_path, dest_path)
    print(f"[OK] Copied to artifact directory: {dest_path}")
