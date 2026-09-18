import os
import json
import shutil
from huggingface_hub import HfApi

REPO_ID = "CH3NDev/dual-loop-qwen3.5-2b"
api = HfApi()

print(f"[*] Connecting to Hugging Face Hub for {REPO_ID}...")
user_info = api.whoami()
print(f"[+] Authenticated as: {user_info['name']}")

# Verify prerequisite files
required_files = [
    ("dual_loop/checkpoints/adapter_model.safetensors", "adapter_model.safetensors"),
    ("adapter_config.json", "adapter_config.json"),
    ("hf_model_card.md", "README.md"),
    ("eval_results/architecture_version_evolution.png", "architecture_version_evolution.png"),
    ("authentic_20_benchmark_scoreboard.png", "authentic_20_benchmark_scoreboard.png"),
    ("smart_brain_loop_architecture.png", "smart_brain_loop_architecture.png"),
    ("authentic_multibenchmark_matrix_graph.png", "authentic_multibenchmark_matrix_graph.png"),
    ("side_by_side_benchmark_wronglog_graph.png", "side_by_side_benchmark_wronglog_graph.png"),
    ("underperforming_benchmarks_5x_graph.png", "underperforming_benchmarks_5x_graph.png"),
    ("wrong_log_persistence_graph.png", "wrong_log_persistence_graph.png"),
    ("eval_results/side_by_side_wronglog_eval.json", "eval_results/side_by_side_wronglog_eval.json"),
    ("eval_results/underperforming_benchmarks_5x.json", "eval_results/underperforming_benchmarks_5x.json"),
    ("eval_results/wrong_log_persistence_bench.json", "eval_results/wrong_log_persistence_bench.json"),
    ("eval_results/matrix_helper_benchmark.json", "eval_results/matrix_helper_benchmark.json"),
    ("eval_results/qwen35_2b_authentic_20_benchmarks.json", "eval_results/qwen35_2b_authentic_20_benchmarks.json"),
    ("eval_results/arc_challenge_authentic_eval_n100.json", "eval_results/arc_challenge_authentic_eval_n100.json"),
    ("eval_results/sciq_msqa_matrix_helper_eval_n100.json", "eval_results/sciq_msqa_matrix_helper_eval_n100.json"),
    (".eval_results/ai2_arc.yaml", ".eval_results/ai2_arc.yaml"),
    (".eval_results/openbookqa.yaml", ".eval_results/openbookqa.yaml"),
    (".eval_results/piqa.yaml", ".eval_results/piqa.yaml"),
    (".eval_results/bbh.yaml", ".eval_results/bbh.yaml"),
    (".eval_results/matrix_helper.yaml", ".eval_results/matrix_helper.yaml"),
    (".eval_results/macro_20_benchmarks.yaml", ".eval_results/macro_20_benchmarks.yaml")
]

# Optional legacy checkpoint files if present
if os.path.exists("dual_loop/checkpoints/qwen35_2b_deliberation_adapter.pt"):
    required_files.append(("dual_loop/checkpoints/qwen35_2b_deliberation_adapter.pt", "qwen35_2b_deliberation_adapter.pt"))
if os.path.exists("full_benchmark_scoreboard.png"):
    required_files.append(("full_benchmark_scoreboard.png", "full_benchmark_scoreboard.png"))

print(f"[*] Total assets queued for upload: {len(required_files)}")

for local_path, repo_path in required_files:
    if not os.path.exists(local_path):
        print(f"[!] Warning: {local_path} not found, skipping...")
        continue
    print(f"[*] Uploading {local_path} -> {repo_path}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=REPO_ID
    )
    print(f"    [OK] Uploaded {repo_path}")

print(f"\n[+] SUCCESS: All latest model weights, configs, benchmark graphs, and model card are live on Hugging Face Hub:")
print(f"    https://huggingface.co/{REPO_ID}")

# Synchronize Space
SPACE_ID = "CH3NDev/dual-loop-controller-demo"
print(f"\n[*] Uploading spaces_demo to Space {SPACE_ID}...")
try:
    api.upload_folder(
        folder_path="spaces_demo",
        repo_id=SPACE_ID,
        repo_type="space"
    )
    print(f"[+] Successfully synchronized Space: https://huggingface.co/spaces/{SPACE_ID}")
except Exception as e:
    print(f"[!] Error synchronizing Space: {e}")
