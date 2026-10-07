import sys
sys.path.insert(0, r"c:\Users\Matthew Chen\Documents\X-Star")

import math
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from scripts.train_and_evaluate_hadl_v4_empirical import FULL_16_TASK_SUITE, check_expanded_eval_correctness

MODEL_ID = "Qwen/Qwen3.5-2B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
)

class Task1StoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True).lower()
        if "final answer:" in text and any(c in text for c in ["i", "a", "b", "c", "d"]):
            return True
        return False

# What if we prompt Task 1 with strict brevity or use a LogitsProcessor that penalizes re-verification?
class AntiReverificationLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.double_check_toks = tokenizer.encode("Let's double-check", add_special_tokens=False) + tokenizer.encode("Alternatively", add_special_tokens=False)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        text = self.tokenizer.decode(input_ids[0][-30:], skip_special_tokens=True)
        if "I * I = I" in text or "E = I" in text:
            # Boost Final Answer
            tok_nl = self.tokenizer.encode("\n", add_special_tokens=False)[0]
            scores[0, tok_nl] += 5.0
        return scores

task1 = FULL_16_TASK_SUITE[0]
messages = [{"role": "user", "content": task1["prompt"]}]
txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inp = tokenizer(txt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    out = base_model.generate(
        inp.input_ids,
        max_new_tokens=600,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

gen = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
chk = check_expanded_eval_correctness(task1["id"], task1["ground_truth"], gen)
print(f"Base Task 1 check: {chk['is_correct']}, Extracted: {chk['extracted_answer']}")
print(f"End of output:\n{gen[-300:]}")
