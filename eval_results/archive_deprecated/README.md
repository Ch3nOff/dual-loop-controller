# Archived & Deprecated Prototype Benchmark Files

> [!WARNING]
> **DEPRECATED DIRECTORY**: The files in this folder are early experimental scratch runs, preliminary exploratory stubs (N=5-6), raw floating-point cache dumps, or uncompressed legacy evaluations from early development iterations (v2.0 - v2.1).
> 
> **DO NOT USE THESE FILES FOR OFFICIAL SCIENTIFIC COMPARISONS.**
> All authoritative, statistically calibrated benchmarks (N=75-200) reside in the parent directory: [../](../).

---

## Inventory of Archived Files & Official Replacements

| Archived / Deprecated File | Historical Role | Reason for Deprecation | Official Modern Replacement |
| :--- | :--- | :--- | :--- |
| qwen35_2b_3pass_selective_memory_eval.json | 20-sample (N=5/task) 3-pass test | Thin stub (386 bytes), no per-sample logs, 0.0s logged lookup time, 0.0% delta. | [../wrong_log_persistence_eval.json](../wrong_log_persistence_eval.json) & [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) (Mode 2) |
| qwen35_2b_3pass_memory_evaluation.png | Visualization for 3-pass stub | Rendered from the deprecated 20-sample stub. | [../wrong_log_persistence_graph.png](../wrong_log_persistence_graph.png) |
| matrix_helper_benchmark.json | 6-item qualitative demonstration | Qualitative proof-of-concept only (N=6). | [../sciq_msqa_matrix_helper_eval_n100.json](../sciq_msqa_matrix_helper_eval_n100.json) (N=100) |
| daptive_comparison_results.json | Early comparison summary | Summary dictionary without per-sample verification or timestamps. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| qwen35_2b_three_way_comparison.json / .png | v2.0 baseline vs early adapter | Superseded by multi-system 20-task evaluation. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) & [.png](../authentic_20_benchmarks_all_systems.png) |
| ix_ablation_results.json | 3-item scratch test | 639-byte unreferenced scratch file. | [../underperforming_benchmarks_5x_run.json](../underperforming_benchmarks_5x_run.json) |
| cached_obqa_25_raw_scores.json | Intermediate logit cache | Scratch list of raw float scores without context. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| 	oy_model_225k_plasticity_eval.json | Tiny 225k parameter toy model | Evaluated on toy synthetic architecture, not the official Qwen/Qwen3.5-2B model. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| 
ealtime_benchmark_results.json | Scratch timing probe | Unreferenced intermediate run without item details. | [../side_by_side_wronglog_eval.json](../side_by_side_wronglog_eval.json) |
| lended_eval_n20.json | 20-item blended exploration | Small exploratory sample without significance testing. | [../qwen35_2b_authentic_suite_n160.json](../qwen35_2b_authentic_suite_n160.json) |
| matrix_5x_run_experiment.json | Intermediate 5x matrix exploration | Replaced by full side-by-side wrong-log persistence runs. | [../side_by_side_wronglog_eval.json](../side_by_side_wronglog_eval.json) |
| eval_results_1789538521.json / ...565.json | Raw lm-eval dumps | Redundant unindexed harness dumps. | [../arc_challenge_authentic_eval_n100.json](../arc_challenge_authentic_eval_n100.json) |
| qwen35_2b_base_k0.json / ...dualloop_k2.json | Redundant lm-eval outputs | Duplicate copies of the harness raw dumps. | [../qwen35_2b_authentic_20_benchmarks.json](../qwen35_2b_authentic_20_benchmarks.json) |
| qwen35_2b_full_base_k0.json / ...dualloop_k2.json | Uncompressed lm-eval dumps | Superseded by structured, verifiable JSON logs with per-sample tracking. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| system_comparison_graph.png / comprehensive_dual_loop_behavior.png | Early iteration charts | Early design draft charts replaced by high-resolution publication graphs. | [../authentic_20_benchmarks_all_systems.png](../authentic_20_benchmarks_all_systems.png) |
