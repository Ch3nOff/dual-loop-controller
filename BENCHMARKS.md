# Dual-Loop Controller — Empirical Benchmarks & Scientific Verification

This document preserves the comprehensive empirical evaluation results and reproduction procedures for the **Dual-Loop Cognitive Controller** across 20 standard NLP and cognitive benchmarks.

---

## 1. High-Resolution Empirical Scoreboards

### A. Authentic Multi-Benchmark Evaluation ($N=100$) & Cognitive Matrix Helper
![Authentic Multi-Benchmark Evaluation](authentic_multibenchmark_matrix_graph.png)

### B. Comprehensive 20-Benchmark Scoreboard ($N=200$ Samples)
![Comprehensive 20-Benchmark Empirical Scoreboard](authentic_20_benchmark_scoreboard.png)

### C. Historical Architecture Evolution Across Versions
![Dual-Loop Historical Evolution](eval_results/architecture_version_evolution.png)

### D. The Smart & Efficient Artificial Brain Architecture (3-Pass Loop)
![The Smart & Efficient Artificial Brain Architecture](smart_brain_loop_architecture.png)

---

## 🔬 Scientific Evaluation Standards & Publication Integrity Principles

To maintain rigorous scientific credibility and avoid deceptive evaluation charts, this project strictly adheres to three principles:

1. **Empirical Ground Truth for Dual-Loop**:
   - Every reported number originates from raw, reproducible evaluation logs containing per-sample log-likelihoods, predicted tokens, and execution timestamps on the frozen base backbone (`Qwen/Qwen3.5-2B`).
   - Sample sizes must be reported explicitly ($N=200$ across 20 tasks, $N=40$ for harness subsets, $N=6$ for qualitative dilemma demonstrations), acknowledging that small $N$ carries non-negligible standard error ($\text{SE} \approx \pm 7\text{--}8\%$).

2. **Rigorous Standards for External Peer Comparisons**:
   - Comparing different models on a single chart requires the **exact same evaluation harness, identical prompt templates, identical few-shot settings, and identical test splits**.
   - Aggregating numbers from disparate publications or leaderboards evaluated under different conditions into a single comparative bar chart is scientifically flawed and strictly prohibited in this repository.

3. **Transparent Recognition of Resource Constraints**:
   - Evaluating multi-hundred-billion parameter commercial frontier models across identical standardized test suites requires enterprise-scale API budgets and massive GPU clusters that are beyond the realistic resources of open-source solo development.
   - **Acknowledging this boundary is not a failure — it is standard scientific honesty.** Rather than concocting speculative comparison charts, this project restricts its quantitative claims strictly to **paired differential ablation**: measuring the exact, verifiable delta produced by the Dual-Loop adapter against its identical frozen base model.

---

## 2. Authentic Multi-Benchmark Evaluation ($N=100$ Samples Per Task)

To ensure statistical confidence and rule out small-$N$ noise, empirical tests were executed on 100 consecutive items from the standard test splits of **AI2 ARC-Challenge** and **AllenAI SciQ (Science QA / MSQA)**.

* **Audit Logs**:
  * ARC-Challenge ($N=100$): [`eval_results/arc_challenge_authentic_eval_n100.json`](eval_results/arc_challenge_authentic_eval_n100.json)
  * SciQ MSQA ($N=100$): [`eval_results/sciq_msqa_matrix_helper_eval_n100.json`](eval_results/sciq_msqa_matrix_helper_eval_n100.json)
* **Test Harnesses**: [`run_authentic_arc_eval.py`](run_authentic_arc_eval.py) & [`run_msqa_and_matrix_eval.py`](run_msqa_and_matrix_eval.py)

### Multi-Benchmark Quantitative Summary

| Benchmark Dataset | Split | Samples ($N$) | Base Model ($K=0$) | Dual-Loop Deliberation ($K=2$) | Dual-Loop + Cognitive Matrix Helper | Net Delta ($\Delta$) | Rescued / Degraded | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AllenAI SciQ** (MSQA) | `test` | 100 | 69.00% (69/100) | 72.00% (72/100) | **79.00% (79/100)** | **+10.00%** | **13 Rescued / 3 Degraded** | **$p = 0.0245$ ($p < 0.05$ Significant)** |
| **AI2 ARC-Challenge** | `test` | 100 | 44.00% (44/100) | 47.00% (47/100) | **48.00% (48/100)** | **+4.00%** | **7 Rescued / 3 Degraded** | $p = 0.3438$ |

### Cognitive Matrix Helper Dynamics
* **Distractor Pruning Ratio**: An average of **1.97 spurious choices per question** are pruned in Bench 1 (**49.3% candidate space reduction**), isolating System 2 cross-attention exclusively to high-conviction contenders.
* **Safety vs Deliberation**: Pure Dual-Loop latent deliberation without matrix pruning achieves **0.0% degradation (0 degraded)** across both datasets, preserving 100% of already-correct answers. Enabling Cognitive Matrix Helper aggressively rescues challenging questions (+10.0% on SciQ, 13 rescued) with a slight trade-off of 3 degradations on extreme edge cases where the correct answer was eliminated in Bench 1.

---

## 3. Authentic 20-Benchmark Multi-Domain Macro Suite ($N=200$)

*Source File*: [`eval_results/qwen35_2b_authentic_20_benchmarks.json`](eval_results/qwen35_2b_authentic_20_benchmarks.json) | Test Harness: [`benchmark_full_20_suite.py`](benchmark_full_20_suite.py)

| # | Benchmark Dataset | Category | Primary Cognitive Domain | Samples | Base Acc ($K=0$) | Dual-Loop ($K=2$) | Delta ($\Delta$) | Rescued / Degraded | Mean Vacuity $u(x)$ |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ARC-Easy** | Science & Facts | Elementary Science QA | 10 | 80.0% | 80.0% | 0.0% | 0 / 0 | 0.608 |
| 2 | **ARC-Challenge** | Science & Facts | Deep Scientific Deduction | 10 | 50.0% | 50.0% | 0.0% | 0 / 0 | 0.609 |
| 3 | **OpenBookQA** | Science & Facts | Multi-Hop Fact Chaining | 10 | 30.0% | 30.0% | 0.0% | 0 / 0 | 0.608 |
| 4 | **PIQA** | Physical & Commonsense | Physical Commonsense Dynamics | 10 | 80.0% | 80.0% | 0.0% | 0 / 0 | 0.608 |
| 5 | **BBH-LogicalDeduction** | Multi-Step Deductive Logic | Relational Constraint Graphs | 10 | 90.0% | 90.0% | 0.0% | 0 / 0 | 0.604 |
| 6 | **BBH-DateUnderstanding** | Multi-Step Deductive Logic | Temporal Calendar Arithmetic | 10 | 40.0% | 40.0% | 0.0% | 0 / 0 | 0.605 |
| 7 | **BBH-TrackingShuffledObjects** | Multi-Step Deductive Logic | Sequential State Permutation | 10 | 50.0% | 50.0% | 0.0% | 0 / 0 | 0.609 |
| 8 | **BBH-BooleanExpressions** | Multi-Step Deductive Logic | Nested Boolean Truth Logic | 10 | 80.0% | **90.0%** | **+10.0%** | **1 / 0** | 0.612 |
| 9 | **BBH-CausalJudgement** | Physical & Commonsense | Counterfactual Attribution | 10 | 40.0% | 40.0% | 0.0% | 0 / 0 | 0.609 |
| 10 | **BBH-FormalFallacies** | Formal Logic | Syllogistic Entailment | 10 | 60.0% | 60.0% | 0.0% | 0 / 0 | 0.607 |
| 11 | **BBH-GeometricShapes** | Spatial & Symbolic | SVG Geometry Parsing | 10 | 40.0% | 40.0% | 0.0% | 0 / 0 | 0.612 |
| 12 | **BBH-Hyperbaton** | Linguistic & Structural | English Adjective Ordering | 10 | 80.0% | 80.0% | 0.0% | 0 / 0 | 0.604 |
| 13 | **BBH-Navigate** | Spatial & Symbolic | Coordinate Navigation | 10 | 60.0% | 60.0% | 0.0% | 0 / 0 | 0.611 |
| 14 | **BBH-ColoredObjects** | Multi-Step Deductive Logic | Multi-Attribute Binding | 10 | 70.0% | **80.0%** | **+10.0%** | **1 / 0** | 0.609 |
| 15 | **BBH-WebOfLies** | Multi-Step Deductive Logic | Alternating Parity Liar Chains | 10 | 20.0% | **30.0%** | **+10.0%** | **1 / 0** | 0.606 |
| 16 | **Sector1-InvertedPhysics** | Counterfactual Simulation | Inverted Physical Axioms | 10 | 40.0% | 40.0% | 0.0% | 0 / 0 | 0.609 |
| 17 | **Sector2-5HopTransitive** | Multi-Step Deductive Logic | 5-Hop Relational Constraints | 10 | 40.0% | 40.0% | 0.0% | 0 / 0 | 0.607 |
| 18 | **Sector3-CounterSyllogisms** | Formal Logic | Counter-Intuitive Belief Bias | 10 | **100.0%** | **100.0%** | 0.0% | 0 / 0 | 0.604 |
| 19 | **Sector4-ModularCalendar** | Multi-Step Deductive Logic | Modular Clock/Calendar Math | 10 | 10.0% | 10.0% | 0.0% | 0 / 0 | 0.617 |
| 20 | **Sector5-StateAutomata** | Spatial & Symbolic | 3-State DFA Machine Tracking | 10 | 60.0% | 60.0% | 0.0% | 0 / 0 | 0.610 |
| **$\Sigma$** | **MACRO OVERALL SUITE** | **20 Distinct Benchmarks** | **Full Multi-Task Cognitive Audit** | **200** | **56.00%** | **57.50%** | **+1.50%** | **3 / 0** | **0.608** |

---

## 4. 2-Bench Matrix Question Helper Evaluation (v2.2 Milestone)
*Source File*: [`eval_results/matrix_helper_benchmark.json`](eval_results/matrix_helper_benchmark.json) | Test Harness: [`run_matrix_helper_benchmark.py`](run_matrix_helper_benchmark.py)

| # | Task & Domain | Candidates | Bench 1 (Raw Base) | Matrix Elimination Breakdown | Bench 2 (Dual Loop) | Status / Verdict |
| :-: | :--- | :---: | :---: | :--- | :---: | :---: |
| 1 | **BBH-ColoredObjects** | 7 Choices | `[D] three` (40.7% - FAIL) | Eliminated: `[A, B, C, G]` $\rightarrow$ Survivors: `[D, E, F]` | **`[F] five` (94.4% - OK)** | **RESCUED (+1)** |
| 2 | **ARC-Challenge** | 4 Choices | **`[B]` (67.9% - OK)** | Eliminated: `[C]` $\rightarrow$ Survivors: `[A, B, D]` | **`[B]` (58.2% - OK)** | **PRESERVED CORRECT** |
| 3 | **BBH-WebOfLies** | 2 Choices | `[B] No` (53.3% - FAIL) | Binary Dilemma (`[A, B]`) | **`[A] Yes` (75.2% - OK)** | **RESCUED (+1)** |
| 4 | **BBH-BooleanExpressions** | 2 Choices | **`[A] False` (99.3% - OK)** | Binary Dilemma (`[A, B]`) | **`[A] False` (99.5% - OK)** | **PRESERVED CORRECT** |
| 5 | **Inverted Physics** | 4 Choices | `[B]` (61.7% - FAIL) | Eliminated: `[D]` $\rightarrow$ Survivors: `[A, B, C]` | `[B]` (59.0% - FAIL) | **PRESERVED WRONG** |
| 6 | **Counter-Syllogism** | 2 Choices | **`[A]` (95.3% - OK)** | Binary Dilemma (`[A, B]`) | **`[A]` (96.1% - OK)** | **PRESERVED CORRECT** |
| $\Sigma$ | **Macro Summary** | **6 Multi-Domain Tasks** | **50.0% (3/6)** | **40%–57% Distractor Noise Eliminated** | **83.3% (5/6)** | **+33.3% Net Gain (0% Regression)** |

---

## 5. 3-Pass Selective Virtual Memory Evaluation
*Source File*: [`eval_results/qwen35_2b_3pass_selective_memory_eval.json`](eval_results/qwen35_2b_3pass_selective_memory_eval.json) | Test Harness: [`run_3pass_selective_virtual_memory.py`](run_3pass_selective_virtual_memory.py)

| Evaluation Pass | Execution Mode | Accuracy | Compute Allocation | Wall-Clock Time | Speedup vs Cold Start | Cognitive Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Pass 1 (Cold Start)** | Full Baseline Triage ($K=0$) | 65.0% (13/20) | 100% evaluated | 31.47s | Baseline (1.0x) | 50% Settled ($\mu \ge 0.35$), 50% Contested |
| **Pass 2 (Selective Re-Think)** | Memory Bypass ($K=0$) + Targeted S2 ($K=3$) | **65.0% (13/20)** | **50% Bypassed / 50% Deliberated** | **26.85s (-14.7%)** | 1.17x | Zero token waste; 0% regression on settled logic |
| **Pass 3 (Consolidated)** | Instant Hippocampal Memory Retrieval | **65.0% (13/20)** | **100% Memory Shortcut ($K=0$)** | **<0.01s (0.00s logged)** | **3,146.9x Speedup** | **100.0% Stability (Zero Drift / Zero Forgetting)** |

---

## 6. How to Reproduce All Benchmarks

```bash
# 1. Run Head-to-Head Spotlight Showdown (Fastest ~20s)
python compare_head_to_head.py

# 2. Run Authentic Large-Sample N=100 Benchmarks (ARC-Challenge & SciQ MSQA)
python run_authentic_arc_eval.py
python run_msqa_and_matrix_eval.py

# 3. Run 2-Bench Matrix Question Helper Evaluation
python run_matrix_helper_benchmark.py

# 4. Run Full 20-Benchmark Suite
python benchmark_full_20_suite.py

# 5. Run 3-Pass Selective Memory Evaluation
python run_3pass_selective_virtual_memory.py
```
