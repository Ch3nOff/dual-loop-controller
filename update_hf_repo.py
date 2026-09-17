import os
import json
import shutil
from huggingface_hub import HfApi

REPO_ID = "CH3NDev/dual-loop-qwen3.5-2b"
api = HfApi()

print(f"[*] Preparing upload for {REPO_ID}...")

# 1. Update adapter_config.json
config_2b = {
    "adapter_type": "dual_loop_cognitive_controller",
    "architecture": "DualLoopQwenModel",
    "base_model_name_or_path": "Qwen/Qwen3.5-2B",
    "compatible_base_models": [
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B"
    ],
    "d_model": 2048,
    "n_heads": 8,
    "num_thought_tokens": 4,
    "max_ponder_steps": 3,
    "num_cwm_slots": 16,
    "capacity_factor": 0.5,
    "adapter_mode": "residual",
    "target_layer_idx": 11,
    "target_layer_type": "full_attention",
    "rezero_gating": True,
    "rezero_alpha_learned": 0.0514,
    "total_adapter_parameters": 110224469,
    "trainable_ratio_pct": 5.857,
    "torch_dtype": "float32",
    "empirical_suite_results_n160": {
        "arc_easy": {
            "samples": 40,
            "base_acc": 0.725, "loop_acc": 0.700, "delta_acc": "-2.5%",
            "base_acc_norm": 0.700, "loop_acc_norm": 0.775, "delta_acc_norm": "+7.5%"
        },
        "arc_challenge": {
            "samples": 40,
            "base_acc": 0.500, "loop_acc": 0.550, "delta_acc": "+5.0%",
            "base_acc_norm": 0.525, "loop_acc_norm": 0.575, "delta_acc_norm": "+5.0%"
        },
        "openbookqa": {
            "samples": 40,
            "base_acc": 0.050, "loop_acc": 0.050, "delta_acc": "0.0%",
            "base_acc_norm": 0.250, "loop_acc_norm": 0.225, "delta_acc_norm": "-2.5%"
        },
        "piqa": {
            "samples": 40,
            "base_acc": 0.700, "loop_acc": 0.750, "delta_acc": "+5.0%",
            "base_acc_norm": 0.675, "loop_acc_norm": 0.625, "delta_acc_norm": "-5.0%"
        },
        "suite_mean": {
            "samples": 160,
            "base_acc": 0.4938, "loop_acc": 0.5125, "delta_acc": "+1.88%",
            "base_acc_norm": 0.5375, "loop_acc_norm": 0.5500, "delta_acc_norm": "+1.25%"
        }
    }
}

with open("adapter_config.json", "w", encoding="utf-8") as f:
    json.dump(config_2b, f, indent=2)

# 2. Model Card README.md
if not os.path.exists("hf_model_card.md"):
    raise FileNotFoundError("hf_model_card.md not found! Please create it before running update.")
print("[*] Using hf_model_card.md for Hugging Face README...")

# 3. Upload files to Hugging Face Hub
print("[*] Uploading adapter_model.safetensors...")
api.upload_file(
    path_or_fileobj="dual_loop/checkpoints/adapter_model.safetensors",
    path_in_repo="adapter_model.safetensors",
    repo_id=REPO_ID
)

print("[*] Uploading qwen35_2b_adapter.pt...")
api.upload_file(
    path_or_fileobj="dual_loop/checkpoints/qwen35_2b_deliberation_adapter.pt",
    path_in_repo="qwen35_2b_adapter.pt",
    repo_id=REPO_ID
)

print("[*] Uploading adapter_config.json...")
api.upload_file(
    path_or_fileobj="adapter_config.json",
    path_in_repo="adapter_config.json",
    repo_id=REPO_ID
)

print("[*] Uploading full_benchmark_scoreboard.png...")
api.upload_file(
    path_or_fileobj="full_benchmark_scoreboard.png",
    path_in_repo="full_benchmark_scoreboard.png",
    repo_id=REPO_ID
)

print("[*] Uploading qwen35_2b_three_way_comparison.png...")
api.upload_file(
    path_or_fileobj="eval_results/qwen35_2b_three_way_comparison.png",
    path_in_repo="qwen35_2b_three_way_comparison.png",
    repo_id=REPO_ID
)

print("[*] Uploading comprehensive_dual_loop_behavior.png...")
api.upload_file(
    path_or_fileobj="comprehensive_dual_loop_behavior.png",
    path_in_repo="comprehensive_dual_loop_behavior.png",
    repo_id=REPO_ID
)

print("[*] Uploading latest_architecture_benchmark.png...")
api.upload_file(
    path_or_fileobj="latest_architecture_benchmark.png",
    path_in_repo="latest_architecture_benchmark.png",
    repo_id=REPO_ID
)

print("[*] Uploading README.md...")
api.upload_file(
    path_or_fileobj="hf_model_card.md",
    path_in_repo="README.md",
    repo_id=REPO_ID
)

print("[+] All assets successfully published to Hugging Face Hub: https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b")
