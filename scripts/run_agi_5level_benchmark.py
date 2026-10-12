"""
AGI-5L Maturity Benchmark Runner:
Evaluates X-Star 2B Autonomous Brain across all 5 Levels of AI / AGI:
1. Level 1: Chatbots (Pengetahuan & Bahasa)
2. Level 2: Reasoners (Logika & Pemecahan Masalah)
3. Level 3: Agents (Aksi Otonom & Adaptasi)
4. Level 4: Innovators (Penemuan Sains Mandiri)
5. Level 5: Organizations (Superintelligence / ASI)

Produces:
- Full empirical audit JSON
- Verbatim execution log
- Publication-quality Radar Chart matching the user's exact 5-level taxonomy
"""

import sys
import os
import json
import time
import re
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.suite_agi_5level_benchmark import get_agi_5level_suite

MODEL_DIR = PROJECT_ROOT / "dist" / "xstar_2b_autonomous_brain"
EVAL_RESULTS = PROJECT_ROOT / "eval_results"
EVAL_RESULTS.mkdir(parents=True, exist_ok=True)

AUDIT_FILE = EVAL_RESULTS / "agi_5level_maturity_audit.json"
LOG_FILE = EVAL_RESULTS / "agi_5level_maturity_log.txt"
RADAR_FILE = EVAL_RESULTS / "agi_5level_maturity_radar.png"
ARTIFACT_DIR = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")
RADAR_ARTIFACT = ARTIFACT_DIR / "agi_5level_maturity_radar.png"


def run_benchmark():
    import xstar
    print("=" * 80)
    print("AGI-5L MATURITY BENCHMARK: X-STAR 2B EMPIRICAL EVALUATION")
    print("Assessing 50 Curated Questions Across 5 Progressive Levels of AGI")
    print("Hardware: Local NVIDIA RTX 5060 Laptop GPU (Thermal Safety Yield Enabled)")
    print("=" * 80)

    print(f"[*] Loading model from {MODEL_DIR}...")
    model = xstar.load(str(MODEL_DIR))
    print(f"[+] Loaded model successfully on {model.device}")

    suite = get_agi_5level_suite()
    total_q = len(suite)

    levels = [
        "Level 1: Chatbots",
        "Level 2: Reasoners",
        "Level 3: Agents",
        "Level 4: Innovators",
        "Level 5: Organizations"
    ]

    level_stats = {
        lvl: {"total": 0, "correct": 0, "latency": 0.0, "tokens": 0}
        for lvl in levels
    }

    results = []
    total_tokens = 0
    t_start_all = time.time()

    with open(LOG_FILE, "w", encoding="utf-8") as f_log:
        f_log.write("=======================================================================\n")
        f_log.write("AGI-5L MATURITY BENCHMARK: X-STAR 2B EMPIRICAL AUDIT LOG\n")
        f_log.write("=======================================================================\n\n")

        for idx, item in enumerate(suite, 1):
            q_id = item["id"]
            level = item["level"]
            domain = item["domain"]
            q_text = item["question"]
            gt = item["correct"]

            prompt = (
                f"<|im_start|>system\n"
                f"You are X-Star 2B Autonomous Brain. Answer the multiple-choice question by choosing the single correct option letter (A, B, C, or D).\n"
                f"State only the single letter answer directly.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"[{level} | {domain}]\n"
                f"{q_text}\n\n"
                f"Choose the correct option letter (A, B, C, or D):<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"The correct answer is: "
            )

            t0_q = time.time()
            out = model.generate(prompt, max_new_tokens=40, temperature=0.0, strip_thinking=True)
            t_elapsed = time.time() - t0_q

            # Thermal / display driver safety pause (protect laptop GPU)
            time.sleep(0.08)

            tok_count = len(out.split()) * 1.3
            total_tokens += tok_count

            # Extract choice
            pred = None
            m = re.search(r'([A-D])\b', out)
            if m:
                pred = m.group(1).upper()
            else:
                # Check numeric fallback (1=A, 2=B, 3=C, 4=D)
                m_num = re.search(r'\b([1-4])\b', out)
                if m_num:
                    num_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
                    pred = num_map[m_num.group(1)]

            is_correct = (pred == gt)

            # Update stats
            level_stats[level]["total"] += 1
            if is_correct:
                level_stats[level]["correct"] += 1
            level_stats[level]["latency"] += t_elapsed
            level_stats[level]["tokens"] += tok_count

            status_str = "PASS [✓]" if is_correct else "FAIL [✗]"
            running_total = sum(s["correct"] for s in level_stats.values())
            running_acc = (running_total / idx) * 100.0

            print(f"[{idx:02d}/{total_q:02d}] {q_id:<6} | {level:<22} | {domain:<32} | {status_str} (Pred: {pred}, GT: {gt}) | {t_elapsed:.2f}s | Acc: {running_acc:5.1f}%")

            rec = {
                "id": q_id,
                "level": level,
                "domain": domain,
                "correct": is_correct,
                "pred": pred,
                "gt": gt,
                "latency_sec": t_elapsed,
                "tokens": tok_count,
                "response_snippet": out[:200]
            }
            results.append(rec)

            f_log.write(f"[{q_id}] {level} | {domain}\n")
            f_log.write(f"Prompt:\n{q_text}\n")
            f_log.write(f"Pred: {pred} | GT: {gt} | Result: {status_str} | Latency: {t_elapsed:.2f}s\n")
            f_log.write(f"Output:\n{out}\n")
            f_log.write("-" * 75 + "\n\n")
            f_log.flush()

    total_time = time.time() - t_start_all
    overall_correct = sum(s["correct"] for s in level_stats.values())
    overall_acc = (overall_correct / total_q) * 100.0

    print("\n" + "=" * 80)
    print("AGI-5L MATURITY BENCHMARK COMPLETE!")
    print(f"Total Questions Evaluated : {total_q}")
    print(f"Total Correct             : {overall_correct} / {total_q} ({overall_acc:.2f}%)")
    print(f"Total Tokens Generated    : {int(total_tokens):,}")
    print(f"Total Evaluation Time     : {total_time:.2f}s (~{total_time/60:.1f} mins)")
    print(f"Hardware Throughput       : {total_tokens / max(total_time, 0.001):.2f} tok/s")
    print("-" * 80)
    print("LEVEL-BY-LEVEL MATURITY BREAKDOWN:")
    for lvl in levels:
        st = level_stats[lvl]
        pct = (st["correct"] / st["total"]) * 100.0
        print(f"  {lvl:<24} : {st['correct']:2d}/{st['total']:2d} ({pct:5.1f}%) | Avg Latency: {st['latency']/st['total']:.2f}s")
    print("=" * 80)

    # Save Audit JSON
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": "AGI-5L Maturity Benchmark (5 Levels of AGI)",
        "model": "dist/xstar_2b_autonomous_brain",
        "total_questions": total_q,
        "overall_correct": overall_correct,
        "overall_accuracy_pct": overall_acc,
        "total_time_sec": total_time,
        "total_tokens": int(total_tokens),
        "throughput_tok_s": total_tokens / max(total_time, 0.001),
        "levels": {
            lvl: {
                "correct": level_stats[lvl]["correct"],
                "total": level_stats[lvl]["total"],
                "accuracy_pct": (level_stats[lvl]["correct"] / level_stats[lvl]["total"]) * 100.0,
                "avg_latency_sec": level_stats[lvl]["latency"] / level_stats[lvl]["total"]
            }
            for lvl in levels
        },
        "results": results
    }

    with open(AUDIT_FILE, "w", encoding="utf-8") as f_aud:
        json.dump(audit_data, f_aud, indent=2)
    print(f"[+] Saved audit JSON to: {AUDIT_FILE}")

    # Generate Radar Chart
    generate_radar_chart(audit_data)


def generate_radar_chart(audit_data):
    """Draws publication-quality radar maturity chart matching user's exact photo."""
    categories = [
        "Level 1: Chatbots\n(Pengetahuan & Bahasa)",
        "Level 2: Reasoners\n(Logika & Pemecahan Masalah)",
        "Level 3: Agents\n(Aksi Otonom & Adaptasi)",
        "Level 4: Innovators\n(Penemuan Sains Mandiri)",
        "Level 5: Organizations\n(Superintelligence / ASI)"
    ]

    levels_keys = [
        "Level 1: Chatbots",
        "Level 2: Reasoners",
        "Level 3: Agents",
        "Level 4: Innovators",
        "Level 5: Organizations"
    ]

    # Scores from audit
    xstar_scores = [audit_data["levels"][k]["accuracy_pct"] for k in levels_keys]

    # Frontier AI baseline (from the user's uploaded image)
    frontier_ai_scores = [100.0, 78.0, 28.0, 12.0, 6.0]

    # Threshold for full AGI (Levels 1-3 complete)
    agi_threshold = [100.0, 100.0, 100.0, 100.0, 100.0]

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Close polygons
    angles += angles[:1]
    xstar_scores += xstar_scores[:1]
    frontier_ai_scores += frontier_ai_scores[:1]
    agi_threshold += agi_threshold[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), dpi=300)

    # Shift starting angle to top (Level 1 at top)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid lines and labels
    ax.set_rgrids([25, 50, 75, 100], ["25%", "50%", "75%", "100% Penuh"], fontsize=10, color="gray")
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11, fontweight="bold", color="#0f172a")
    ax.set_ylim(0, 115)

    # Plot 1: Batas Ambang AGI Penuh (Green dotted line)
    ax.plot(angles, agi_threshold, linestyle=":", color="#10b981", linewidth=2.5, label="Batas Ambang AGI Penuh (Level 1-3 Selesai)")

    # Plot 2: Status Frontier AI Saat Ini (Red solid line + light fill)
    ax.plot(angles, frontier_ai_scores, color="#ef4444", linewidth=2.5, marker="o", markersize=6, label="Status Frontier AI Saat Ini")
    ax.fill(angles, frontier_ai_scores, color="#ef4444", alpha=0.15)

    # Plot 3: X-Star 2B Autonomous Brain (Deep Blue line + cyan fill)
    ax.plot(angles, xstar_scores, color="#2563eb", linewidth=3.5, marker="s", markersize=8, label=f"X-Star 2B Autonomous Brain ({audit_data['overall_accuracy_pct']:.1f}% Avg)")
    ax.fill(angles, xstar_scores, color="#3b82f6", alpha=0.25)

    # Annotate points on X-Star
    for a, s in zip(angles[:-1], xstar_scores[:-1]):
        ax.annotate(f"{s:.0f}%", xy=(a, s), xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color="#1e40af")

    ax.set_title("Radar Kematangan Tingkatan AGI\n(Berdasarkan Klasifikasi 5 Level AI)", fontsize=15, fontweight="bold", color="#0f172a", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 0.05), frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(RADAR_FILE, dpi=300, bbox_inches="tight")
    plt.savefig(RADAR_ARTIFACT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Radar Charts to:\n  - {RADAR_FILE}\n  - {RADAR_ARTIFACT}")


if __name__ == "__main__":
    run_benchmark()
