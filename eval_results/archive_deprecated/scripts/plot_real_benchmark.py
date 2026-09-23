import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load authentic evaluation files
with open('eval_results/archive_deprecated/eval_results_1789538565.json', 'r', encoding='utf-8') as f:
    base_data = json.load(f)

with open('eval_results/archive_deprecated/eval_results_1789538521.json', 'r', encoding='utf-8') as f:
    loop_data = json.load(f)

base_res = base_data['results']['arc_easy']
loop_res = loop_data['results']['arc_easy']

base_acc = base_res['acc,none'] * 100.0
base_acc_err = base_res['acc_stderr,none'] * 100.0
base_norm = base_res['acc_norm,none'] * 100.0
base_norm_err = base_res['acc_norm_stderr,none'] * 100.0

loop_acc = loop_res['acc,none'] * 100.0
loop_acc_err = loop_res['acc_stderr,none'] * 100.0
loop_norm = loop_res['acc_norm,none'] * 100.0
loop_norm_err = loop_res['acc_norm_stderr,none'] * 100.0

base_samples = base_data['samples']['arc_easy']
loop_samples = loop_data['samples']['arc_easy']

# Analyze individual sample transitions
both_correct = 0
both_wrong = 0
improved = 0
degraded = 0
target_ll_shifts = []

for b, l in zip(base_samples, loop_samples):
    target = b['target']
    b_logprobs = [r[0] for r in b['filtered_resps']]
    l_logprobs = [r[0] for r in l['filtered_resps']]
    
    b_pred = int(np.argmax(b_logprobs))
    l_pred = int(np.argmax(l_logprobs))
    
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
        
    target_ll_shifts.append(l_logprobs[target] - b_logprobs[target])

target_ll_shifts = np.array(target_ll_shifts)

# --- Plotting ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(15, 9), facecolor='#0D1117')
gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#161B22')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363D')
    ax.spines['bottom'].set_color('#30363D')
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#8B949E')

# Subplot 1: Raw Accuracy Comparison
labels = ['Raw Accuracy (acc)', 'Normalized Acc (acc_norm)']
x = np.arange(len(labels))
width = 0.32

rects1 = ax1.bar(x - width/2, [base_acc, base_norm], width, yerr=[base_acc_err, base_norm_err],
                 capsize=5, label='Base Qwen-0.5B (K=0)', color='#4B5563', edgecolor='#9CA3AF', linewidth=1.2)
rects2 = ax1.bar(x + width/2, [loop_acc, loop_norm], width, yerr=[loop_acc_err, loop_norm_err],
                 capsize=5, label='Dual-Loop Augmented (K=2)', color='#6366F1', edgecolor='#A5B4FC', linewidth=1.2)

ax1.set_ylabel('Accuracy (%)', fontsize=11, color='#C9D1D9')
ax1.set_title('ARC-Easy Measured Accuracy (N=50, ±1 std err)', fontsize=13, fontweight='bold', color='#F0F6FC', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10, color='#C9D1D9')
ax1.set_ylim(0, 95)
ax1.legend(loc='upper left', framealpha=0.6, facecolor='#161B22', edgecolor='#30363D')

# Value labels on bars
for rect in rects1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., h + 7.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, color='#E5E7EB', fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., h + 7.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, color='#A5B4FC', fontweight='bold')

# Subplot 2: Decision Transition Breakdown
transition_counts = [both_correct, both_wrong, improved, degraded]
transition_labels = [
    f'Invariant Correct ({both_correct})',
    f'Invariant Wrong ({both_wrong})',
    f'Dual-Loop Fixed ({improved})',
    f'Dual-Loop Degraded ({degraded})'
]
transition_colors = ['#10B981', '#6B7280', '#3B82F6', '#EF4444']

y_pos = np.arange(len(transition_labels))
bars2 = ax2.barh(y_pos, transition_counts, color=transition_colors, edgecolor='#1E293B', height=0.55)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(transition_labels, fontsize=10, color='#C9D1D9')
ax2.set_xlabel('Sample Count (N=50)', fontsize=11, color='#C9D1D9')
ax2.set_title('Prediction Transition Matrix (K=0 vs K=2)', fontsize=13, fontweight='bold', color='#F0F6FC', pad=12)
ax2.set_xlim(0, 32)
ax2.grid(axis='x', linestyle='--', alpha=0.2, color='#8B949E')

for bar in bars2:
    w = bar.get_width()
    pct = (w / len(base_samples)) * 100.0
    ax2.text(w + 0.6, bar.get_y() + bar.get_height()/2., f"{w} ({pct:.0f}%)", va='center', fontsize=10, color='#F0F6FC', fontweight='bold')

# Subplot 3: Target Token Log-Likelihood Shift per Question
sample_indices = np.arange(len(target_ll_shifts))
bar_colors = ['#3B82F6' if v > 0 else '#EF4444' if v < 0 else '#6B7280' for v in target_ll_shifts]

ax3.bar(sample_indices, target_ll_shifts, color=bar_colors, width=0.7, edgecolor='none')
ax3.axhline(0, color='#8B949E', linestyle='-', linewidth=1.0, alpha=0.5)
ax3.axhline(np.mean(target_ll_shifts), color='#F59E0B', linestyle='--', linewidth=1.5,
            label=f'Mean Shift: {np.mean(target_ll_shifts):+.2f} nats')

ax3.set_xlabel('Question Index (0 .. 49)', fontsize=11, color='#C9D1D9')
ax3.set_ylabel('Δ Log-Likelihood (nats)', fontsize=11, color='#C9D1D9')
ax3.set_title('Per-Sample Target Token Log-Likelihood Shift (Dual-Loop K=2 vs Base K=0)', fontsize=13, fontweight='bold', color='#F0F6FC', pad=12)
ax3.set_xlim(-1, 50)
ax3.legend(loc='lower right', framealpha=0.4, facecolor='#161B22', edgecolor='#30363D')

# Annotate highlighted questions
ax3.annotate('Q22: Climate (Fixed)', xy=(22, target_ll_shifts[22]), xytext=(22, target_ll_shifts[22] + 2.5),
             arrowprops=dict(arrowstyle='->', color='#3B82F6', lw=1.2), color='#60A5FA', fontsize=9, fontweight='bold', ha='center')
ax3.annotate('Q48: Lead Apron (Fixed)', xy=(48, target_ll_shifts[48]), xytext=(44, target_ll_shifts[48] + 2.5),
             arrowprops=dict(arrowstyle='->', color='#3B82F6', lw=1.2), color='#60A5FA', fontsize=9, fontweight='bold', ha='center')
ax3.annotate('Q7: Mass Unit (Degraded)', xy=(7, target_ll_shifts[7]), xytext=(7, target_ll_shifts[7] - 2.0),
             arrowprops=dict(arrowstyle='->', color='#EF4444', lw=1.2), color='#F87171', fontsize=9, fontweight='bold', ha='center')

# Title & subtitle
plt.suptitle('Authentic EleutherAI LM-Evaluation Harness Audit: Qwen2.5-0.5B + Dual-Loop Controller',
             fontsize=16, fontweight='bold', color='#F0F6FC', y=0.98)

output_img = 'real_benchmark_analysis.png'
plt.savefig(output_img, dpi=300, facecolor='#0D1117', bbox_inches='tight')
print(f"[Done] Real benchmark chart saved to: {output_img}")
