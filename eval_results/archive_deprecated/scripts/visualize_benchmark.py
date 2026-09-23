import os
import json
import matplotlib.pyplot as plt

# Full 20 datasets from official Qwen3.5-2B scorecard (llm-stats.com)
data = {
    "MMLU-Redux": 83.2,
    "IFEval": 81.8,
    "C-Eval": 75.6,
    "Global PIQA": 71.0,
    "MMLU-Pro": 67.8,
    "MMMLU": 63.7,
    "MAXIFE": 61.2,
    "Include": 55.8,
    "MMLU-ProX": 52.4,
    "GPQA": 51.4,
    "t2-bench": 48.6,
    "NOVA-63": 46.2,
    "WMT24++": 45.4,
    "BFCL-V4": 43.1,
    "IFBench": 41.2,
    "LongBench v2": 38.6,
    "SuperGPQA": 37.2,
    "Multi-Challenge": 34.0,
    "PolyMATH": 26.8,
    "AA-LCR": 26.0,
}

# Sort descending by score
sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

tasks = list(sorted_data.keys())
scores = list(sorted_data.values())

# Dark-mode styling matching llm-stats.com scorecard exactly
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(18, 7), facecolor="#0E0E10")
ax.set_facecolor("#0E0E10")

# Bar styling
bars = ax.bar(tasks, scores, color="#5B4DF6", width=0.58, edgecolor="none")

# Highlight the weakest reasoning bottleneck datasets with accent
for i, task in enumerate(tasks):
    if task in ["PolyMATH", "AA-LCR", "Multi-Challenge"]:
        bars[i].set_color("#7A6BF8")

# Numerical values on top of each bar
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1.2,
        f"{score:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#D0D0D5",
        fontweight="bold"
    )

# Title, labels & subtitle
fig.text(0.06, 0.94, "Qwen3.5-2B Performance Across Datasets", fontsize=18, fontweight="bold", color="#FFFFFF")
fig.text(0.06, 0.90, "Scores sourced from the model's scorecard, paper, or official blog posts", fontsize=11, color="#8E8E93")
fig.text(0.06, 0.86, "llm-stats.com - Wed Sep 16 2026", fontsize=9, color="#636366")

ax.set_ylabel("Score (%)", fontsize=11, color="#8E8E93", labelpad=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#2C2C2E")
ax.spines["bottom"].set_color("#2C2C2E")
ax.grid(axis="y", linestyle="--", alpha=0.15, color="#555555")
plt.xticks(rotation=45, ha="right", fontsize=9.5, color="#C7C7CC")
plt.yticks(range(0, 101, 20), [f"{y}" for y in range(0, 101, 20)], fontsize=9.5, color="#8E8E93")
plt.ylim(0, 98)

plt.subplots_adjust(top=0.82, bottom=0.20, left=0.06, right=0.97)

output_file = "benchmark_barchart.png"
plt.savefig(output_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
print(f"Full 20-dataset chart saved to: {os.path.abspath(output_file)}")

# Also update JSON
os.makedirs("eval_results", exist_ok=True)
with open("eval_results/qwen35_2b_eval_summary.json", "w") as f:
    json.dump(sorted_data, f, indent=2)
print("Updated eval_results/qwen35_2b_eval_summary.json")
