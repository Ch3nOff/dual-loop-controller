import json

with open("/mnt/c/Users/Matthew Chen/Documents/X-Star/eval_results/large_scale_264_benchmark.json") as f:
    d = json.load(f)

print("=== 264 BENCHMARK SUMMARY ===")
print("GPU:", d["gpu_name"])
print("Total Tasks Evaluated:", d["total_tasks_evaluated"])
print("\n[HUMANEVAL - 164 TASKS (100% Full Benchmark)]")
print("Baseline Pass@1:", d["baseline"]["humaneval"]["pass_rate_pct"], f"% ({d['baseline']['humaneval']['passed_tasks']}/164)")
print("HADL v4.5 Pass@1:", d["hadl_v45"]["humaneval"]["pass_rate_pct"], f"% ({d['hadl_v45']['humaneval']['passed_tasks']}/164)")
print("HumanEval TPS: Baseline =", d["baseline"]["humaneval"]["throughput_tps"], "tps | HADL =", d["hadl_v45"]["humaneval"]["throughput_tps"], "tps")
print("HumanEval Time: Baseline =", d["baseline"]["humaneval"]["total_time_sec"], "s | HADL =", d["hadl_v45"]["humaneval"]["total_time_sec"], "s")

print("\n[GSM8K - 100 TASKS (Official Multi-step Math)]")
print("Baseline Accuracy:", d["baseline"]["gsm8k"]["accuracy_pct"], f"% ({d['baseline']['gsm8k']['correct_tasks']}/100)")
print("HADL v4.5 Accuracy:", d["hadl_v45"]["gsm8k"]["accuracy_pct"], f"% ({d['hadl_v45']['gsm8k']['correct_tasks']}/100)")
print("GSM8K TPS: Baseline =", d["baseline"]["gsm8k"]["throughput_tps"], "tps | HADL =", d["hadl_v45"]["gsm8k"]["throughput_tps"], "tps")
print("GSM8K Time: Baseline =", d["baseline"]["gsm8k"]["total_time_sec"], "s | HADL =", d["hadl_v45"]["gsm8k"]["total_time_sec"], "s")
