# Archived & Deprecated Prototype Benchmark Files

> [!WARNING]
> **DEPRECATED DIRECTORY**: The files in this folder are early experimental scratch runs, preliminary exploratory stubs (N=5-10/task), legacy .pt model evaluations, raw floating-point cache dumps, or uncompressed harness outputs from early development iterations (v2.0 - v2.1).
> 
> **DO NOT USE THESE FILES FOR OFFICIAL SCIENTIFIC COMPARISONS.**
> All authoritative, statistically calibrated benchmarks (N=50-200) reside in the parent directory: [../](../).

---

## 📋 Inventory of Archived Files & Official Replacements

| Archived / Deprecated File | Historical Role | Reason for Deprecation | Official Modern Replacement |
| :--- | :--- | :--- | :--- |
| **qwen35_2b_authentic_20_benchmarks.json** | Preliminary 20-benchmark baseline (=10$/task) | Small sample size (=10$/task) masked constraint deduction dynamics (e.g. logging \% \to 90\%$ on BBH-LogicalDeduction, conflicting with =200$ statistical ground truth). | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) & [../qwen35_2b_multistep_n200_eval.json](../qwen35_2b_multistep_n200_eval.json) |
| **qwen35_2b_authentic_suite_n160.json** | 160-sample 4-task evaluation | Evaluated legacy .pt checkpoint with unnormalized/normalized metric confusion. | [../sciq_msqa_matrix_helper_eval_n100.json](../sciq_msqa_matrix_helper_eval_n100.json) & [../arc_challenge_authentic_eval_n100.json](../arc_challenge_authentic_eval_n100.json) |
| **qwen35_2b_latest_architecture_eval.json** | Early architecture ablation (=20$/task) | Evaluated legacy .pt checkpoint prior to official Safetensors release. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| **qwen35_2b_3pass_selective_memory_eval.json** | 20-sample (=5$/task) 3-pass test | Thin stub (386 bytes), no per-sample logs, .0\text{s}$ logged lookup time, .0\%$ delta. | [../wrong_log_persistence_eval.json](../wrong_log_persistence_eval.json) & [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) (Mode 2) |
| **qwen35_2b_3pass_memory_evaluation.png** | Visualization for 3-pass stub | Rendered from the deprecated 20-sample stub. | [../wrong_log_persistence_graph.png](../wrong_log_persistence_graph.png) |
| **matrix_helper_benchmark.json** | 6-item qualitative demonstration | Qualitative proof-of-concept only (=6$). | [../sciq_msqa_matrix_helper_eval_n100.json](../sciq_msqa_matrix_helper_eval_n100.json) (=100$) |
| **daptive_comparison_results.json** | Early comparison summary | Summary dictionary without per-sample verification or timestamps. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| **qwen35_2b_three_way_comparison.json / .png** | v2.0 baseline vs early adapter | Superseded by multi-system 20-task evaluation. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) & [.png](../authentic_20_benchmarks_all_systems.png) |
| **ix_ablation_results.json** | 3-item scratch test | 639-byte unreferenced scratch file. | [../underperforming_benchmarks_5x_run.json](../underperforming_benchmarks_5x_run.json) |
| **cached_obqa_25_raw_scores.json** | Intermediate logit cache | Scratch list of raw float scores without context. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| **	oy_model_225k_plasticity_eval.json** | Tiny 225k parameter toy model | Evaluated on toy synthetic architecture, not the official Qwen/Qwen3.5-2B model. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| **
ealtime_benchmark_results.json** | Scratch timing probe | Unreferenced intermediate run without item details. | [../side_by_side_wronglog_eval.json](../side_by_side_wronglog_eval.json) |
| **lended_eval_n20.json** | 20-item blended exploration | Small exploratory sample without significance testing. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| **matrix_5x_run_experiment.json** | Intermediate 5x matrix exploration | Replaced by full side-by-side wrong-log persistence runs. | [../side_by_side_wronglog_eval.json](../side_by_side_wronglog_eval.json) |
| **system_comparison_graph.png / comprehensive_dual_loop_behavior.png** | Early iteration charts | Early design draft charts replaced by high-resolution publication graphs. | [../authentic_20_benchmarks_all_systems.png](../authentic_20_benchmarks_all_systems.png) |
| **images/** | Historical diagram & plot artifacts | Superseded by v2.4.0 visual assets (`hadl_v24_system_architecture.png`, `comprehensive_v24_benchmark_matrix.png`). | Root / README |
| **scripts/** | Historical exploratory run & plotting scripts | Superseded by v2.4.0 unified benchmark suite in `dual_loop/benchmarks/`. | `dual_loop/benchmarks/` |

