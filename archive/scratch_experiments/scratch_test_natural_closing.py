import sys
sys.path.insert(0, r"c:\Users\Matthew Chen\Documents\X-Star")

import math
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from scripts.train_and_evaluate_hadl_v4_empirical import FULL_16_TASK_SUITE, check_expanded_eval_correctness

MODEL_ID = "Qwen/Qwen3.5-2B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
)

class NaturalClosingLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, beta: float = 8.0, max_period: int = 12):
        super().__init__()
        self.tokenizer = tokenizer
        self.beta = beta
        self.max_period = max_period
        
        # Discourage endless "Step X:" or "By Axiom" after step 5
        self.step_toks = tokenizer.encode("Step", add_special_tokens=False) + tokenizer.encode(" Step", add_special_tokens=False)
        self.axiom_toks = tokenizer.encode("Axiom", add_special_tokens=False) + tokenizer.encode(" Axiom", add_special_tokens=False)
        self.lemma_toks = tokenizer.encode("Lemma", add_special_tokens=False) + tokenizer.encode(" Lemma", add_special_tokens=False)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            tokens = input_ids[b].tolist()
            detected_period = 0
            repeat_count = 0
            cycle_tokens = []
            
            for p in range(1, self.max_period + 1):
                if len(tokens) >= 2 * p:
                    if tokens[-p:] == tokens[-2*p:-p]:
                        count = 2
                        k = 3
                        while len(tokens) >= k * p and tokens[-k*p:-(k-1)*p] == tokens[-p:]:
                            count += 1
                            k += 1
                        if count > repeat_count:
                            repeat_count = count
                            detected_period = p
                            cycle_tokens = tokens[-p:]
                            
            if repeat_count >= 2:
                # Log-Cosh penalty
                cosh_val = math.cosh(min(float(repeat_count), 8.0))
                penalty = self.beta * math.log(cosh_val)
                for tid in cycle_tokens:
                    scores[b, tid] -= penalty
                if cycle_tokens:
                    scores[b, cycle_tokens[0]] -= (penalty * 2.0)
                    
                # If in a loop, strongly penalize continuing "Step" / "Axiom"
                for st in self.step_toks + self.axiom_toks + self.lemma_toks:
                    scores[b, st] -= (penalty * 1.5)
                    
        return scores

task1 = FULL_16_TASK_SUITE[0]
messages = [{"role": "user", "content": task1["prompt"]}]
txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inp = tokenizer(txt, return_tensors="pt").to("cuda:0")

processor = NaturalClosingLogitsProcessor(tokenizer=tokenizer, beta=10.0)
proc_list = LogitsProcessorList([processor])

with torch.no_grad():
    out = base_model.generate(
        inp.input_ids,
        max_new_tokens=400,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=proc_list
    )

gen = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
chk = check_expanded_eval_correctness(task1["id"], task1["ground_truth"], gen)
print(f"Task 1 Result: {chk['is_correct']} | Extracted: {chk['extracted_answer']} | Tokens: {len(out[0]) - inp.input_ids.shape[1]}")
print(f"Generation:\n{gen}")
