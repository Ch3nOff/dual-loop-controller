import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load existing data
with open("eval_results/hadl_v44_large_scale_benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

categories = ["MMLU (50 Q)", "ARC (50 Q)", "GSM8k (25 Q)", "HumanEval (25 Q)", "Target Code (25 Q)"]
base_vals = [r["base"] for r in data["table"]]
naive_vals = [r["naive"] for r in data["table"]]
hadl_vals = [r["hadl"] for r in data["table"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), facecolor='#0B0F19', gridspec_kw={'width_ratios': [3.0, 1.2]})
ax1.set_facecolor('#0B0F19')
ax2.set_facecolor('#0B0F19')

x = np.arange(len(categories))
width = 0.26

r1 = ax1.bar(x - width, base_vals, width, label='Base Qwen-2B (Zero-Shot)', color='#64748B', alpha=0.85, edgecolor='#94A3B8')
r2 = ax1.bar(x, naive_vals, width, label='Naive FT (Single Domain SFT)', color='#EF4444', alpha=0.85, edgecolor='#F87171')
r3 = ax1.bar(x + width, hadl_vals, width, label='HADL v4.4 Dual-Loop (Ours)', color='#10B981', alpha=0.95, edgecolor='#34D399')

ax1.set_ylabel('Accuracy Score (%)', fontsize=12, fontweight='bold', color='#F8FAFC')
ax1.set_title('Empirical Benchmark (525 Evaluations): General Retention vs Single-Domain SFT', fontsize=13, fontweight='bold', color='#F8FAFC', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10.5, fontweight='bold', color='#E2E8F0')
ax1.set_ylim(0, 115)
ax1.grid(axis='y', linestyle='--', alpha=0.2, color='#94A3B8')
ax1.legend(loc='upper left', facecolor='#1E293B', edgecolor='#334155', fontsize=10.5, labelcolor='#F8FAFC')

for bar in r1 + r2 + r3:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 1.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#F8FAFC')

# Panel 2: Perplexity Comparison
ppl_names = ['Base', 'Naive FT', 'HADL v4.4']
ppl_vals = [data["perplexity"]["base"], data["perplexity"]["naive_ft"], data["perplexity"]["hadl_v44"]]
ppl_colors = ['#64748B', '#EF4444', '#10B981']

bars_ppl = ax2.bar(ppl_names, ppl_vals, color=ppl_colors, width=0.55, edgecolor='#334155')
ax2.set_ylabel('Perplexity (Lower is Better)', fontsize=11, fontweight='bold', color='#F8FAFC')
ax2.set_title('Wikipedia Prose PPL (1,000+ Tokens)', fontsize=12, fontweight='bold', color='#F8FAFC', pad=15)
ax2.set_ylim(0, max(ppl_vals) * 1.35)
ax2.grid(axis='y', linestyle='--', alpha=0.2, color='#94A3B8')
ax2.tick_params(colors='#E2E8F0', labelsize=10.5)

for bar in bars_ppl:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.25, f'{h:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#F8FAFC')

plt.tight_layout()
out_png = Path("docs/images/hadl_v44_large_scale_benchmark_graph.png")
plt.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

# Copy to artifact dir
artifact_dir = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")
shutil.copy(out_png, artifact_dir / "hadl_v44_large_scale_benchmark_graph.png")
print("Re-plotted cleanly with precise sample sizes.")
