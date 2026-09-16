# Evaluation Results Directory

This directory is designated for authentic raw JSON outputs produced by the EleutherAI LM Evaluation Harness (`lm-eval`):

```bash
python run_lm_eval.py --model Qwen/Qwen3.5-2B --tasks mmlu,ifeval,gpqa --device cuda:0
```

### Current Status
* **Qwen3.5-2B / Qwen2.5-0.5B Adapters**: **UNBENCHMARKED**.
* Previous synthetic mockups and hand-constructed reference tables have been completely removed to maintain strict scientific integrity.
* Only genuine evaluation outputs containing full `lm-eval` schemas (`results`, `configs`, `n-shot`, Git commit hashes) will be stored here upon real GPU execution.
