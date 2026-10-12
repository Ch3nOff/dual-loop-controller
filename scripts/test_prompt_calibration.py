import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import xstar
import re
from scripts.suite_agi_5level_benchmark import get_agi_5level_suite

model = xstar.load("dist/xstar_2b_autonomous_brain")
suite = get_agi_5level_suite()

correct = 0
for idx, item in enumerate(suite[:10], 1):
    q_text = item["question"]
    gt = item["correct"]
    prompt = (
        f"<|im_start|>system\n"
        f"You are X-Star 2B Autonomous Brain. Answer the multiple-choice question by choosing the single correct option letter (A, B, C, or D).\n"
        f"State only the single letter answer directly.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{q_text}\n\n"
        f"Choose the correct option letter (A, B, C, or D):<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"The correct answer is: "
    )
    out = model.generate(prompt, max_new_tokens=40, temperature=0.0, strip_thinking=True)
    m = re.search(r"([A-D])\b", out)
    pred = m.group(1) if m else "None"
    is_ok = (pred == gt)
    if is_ok:
        correct += 1
    print(f"[{item['id']}] GT: {gt} | Pred: {pred} | Match: {is_ok} | Out: {repr(out[:60])}")

print(f"\nScore on first 10 questions: {correct}/10 ({correct*10.0}%)")
