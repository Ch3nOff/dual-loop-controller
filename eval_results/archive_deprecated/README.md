# Archived & Deprecated Prototype Benchmark Files

> [!WARNING]
> **DEPRECATED & RETRACTED DIRECTORY**: The files in this folder are early experimental scratch runs, preliminary exploratory stubs ($N=5-10$/task), unverified peer comparisons, or retracted files failing mathematical consistency checks.
> 
> **DO NOT USE THESE FILES FOR OFFICIAL SCIENTIFIC COMPARISONS.**
> All authoritative, statistically calibrated benchmarks ($N=50-200$) reside in the parent directory: [../](../).

---

## 📋 Inventory of Archived Files & Reason for Deprecation / Retraction

| Archived / Deprecated File | Historical Role | Reason for Deprecation / Retraction | Official Modern Replacement |
| :--- | :--- | :--- | :--- |
| **self_awareness_benchmark_results.json** | Self-awareness & introspection evaluation claim | **RETRACTED**: Fatal arithmetic contradiction (`elapsed_seconds: 0.82` vs claimed condition `latency_ms: 1280.0`), unverified divergent neuron counts ($187,402$ vs $185,344$ vs $2.31\times 10^9$), missing per-item sample logs, and unscientific labeling. | Retracted without replacement. Official introspective reasoning is evaluated via verified epistemic plasticity: [../epistemic_plasticity_benchmark.json](../epistemic_plasticity_benchmark.json) |
| **glm4_peer_comparison.json** | GLM-4 external peer comparison matrix | **QUARANTINED**: Small unverified summary stub (4.1 KB) without per-item execution logs, timestamps, or raw multi-system runner logs. | Quarantined pending calibrated peer evaluation runner with granular per-sample logging. |
| **glm4_v24_benchmark.json** | GLM-4 9B benchmark summary | **QUARANTINED**: Lacks per-item sample logs and physical elapsed-time telemetry against external models. | Quarantined pending calibrated execution runner. |
| **images/self_awareness_architecture_benchmark.png** | Visualization for self-awareness benchmark | Rendered from retracted arithmetic-contradictory JSON. | Retracted. |
| **images/glm4_peer_comparison_matrix.png** | Peer comparison matrix plot | Rendered from quarantined peer comparison stub. | Quarantined. |
| **images/multimodal_peer_comparison_benchmark.png** | Comparative chart for multimodal peers | Rendered from unverified peer comparison stub. | Quarantined. |
| **scripts/benchmark_self_awareness.py** | Generator script for retracted benchmark | Generated inconsistent latency and duration metrics. | Retracted. |
| **scripts/compare_glm4_peers.py** | Generator script for peer comparison | Hardcoded comparative metric stubs. | Quarantined. |
| **qwen35_2b_authentic_20_benchmarks.json** | Preliminary 20-benchmark baseline ($N=10$/task) | Small sample size ($N=10$/task) masked constraint deduction dynamics. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) & [../qwen35_2b_multistep_n200_eval.json](../qwen35_2b_multistep_n200_eval.json) |
| **qwen35_2b_authentic_suite_n160.json** | 160-sample 4-task evaluation | Evaluated legacy `.pt` checkpoint with unnormalized/normalized metric confusion. | [../sciq_msqa_matrix_helper_eval_n100.json](../sciq_msqa_matrix_helper_eval_n100.json) & [../arc_challenge_authentic_eval_n100.json](../arc_challenge_authentic_eval_n100.json) |
| **qwen35_2b_latest_architecture_eval.json** | Early architecture ablation ($N=20$/task) | Evaluated legacy `.pt` checkpoint prior to official Safetensors release. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| **qwen35_2b_3pass_selective_memory_eval.json** | 20-sample ($N=5$/task) 3-pass test | Thin stub (386 bytes), no per-sample logs, $0.0\text{s}$ logged lookup time, $0.0\%$ delta. | [../wrong_log_persistence_eval.json](../wrong_log_persistence_eval.json) & [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) (Mode 2) |
| **qwen35_2b_3pass_memory_evaluation.png** | Visualization for 3-pass stub | Rendered from the deprecated 20-sample stub. | [../wrong_log_persistence_graph.png](../wrong_log_persistence_graph.png) |
| **matrix_helper_benchmark.json** | 6-item qualitative demonstration | Qualitative proof-of-concept only ($N=6$). | [../sciq_msqa_matrix_helper_eval_n100.json](../sciq_msqa_matrix_helper_eval_n100.json) ($N=100$) |
| **adaptive_comparison_results.json** | Early comparison summary | Summary dictionary without per-sample verification or timestamps. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| **qwen35_2b_three_way_comparison.json / .png** | v2.0 baseline vs early adapter | Superseded by multi-system 20-task evaluation. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) & [.png](../authentic_20_benchmarks_all_systems.png) |
| **fix_ablation_results.json** | 3-item scratch test | 639-byte unreferenced scratch file. | [../underperforming_benchmarks_5x_run.json](../underperforming_benchmarks_5x_run.json) |
| **cached_obqa_25_raw_scores.json** | Intermediate logit cache | Scratch list of raw float scores without context. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| **toy_model_225k_plasticity_eval.json** | Tiny 225k parameter toy model | Evaluated on toy synthetic architecture, not the official Qwen/Qwen3.5-2B model. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| **realtime_benchmark_results.json** | Scratch timing probe | Unreferenced intermediate run without item details. | [../side_by_side_wronglog_eval.json](../side_by_side_wronglog_eval.json) |
| **blended_eval_n20.json** | 20-item blended exploration | Small exploratory sample without significance testing. | [../authentic_20_benchmarks_all_systems.json](../authentic_20_benchmarks_all_systems.json) |
| **matrix_5x_run_experiment.json** | Intermediate 5x matrix exploration | Replaced by full side-by-side wrong-log persistence runs. | [../side_by_side_wronglog_eval.json](../side_by_side_wronglog_eval.json) |
| **system_comparison_graph.png / comprehensive_dual_loop_behavior.png** | Early iteration charts | Early design draft charts replaced by high-resolution publication graphs. | [../authentic_20_benchmarks_all_systems.png](../authentic_20_benchmarks_all_systems.png) |
| **epistemic_plasticity_benchmark_summary_only.json** | Early summary-only stub (2056 bytes) | Contained aggregated metric dictionaries without per-sample logs or elapsed execution breakdown. | [../epistemic_plasticity_benchmark.json](../epistemic_plasticity_benchmark.json) |
| **images/** | Historical diagram & plot artifacts | Superseded by v2.4.0 visual assets (`hadl_v24_system_architecture.png`, `comprehensive_v24_benchmark_matrix.png`). | Root / README |
| **scripts/** | Historical exploratory run & plotting scripts | Superseded by verified benchmark suite in `dual_loop/benchmarks/`. | `dual_loop/benchmarks/` |
