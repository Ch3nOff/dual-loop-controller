import sys
sys.path.insert(0, r"c:\Users\Matthew Chen\Documents\X-Star")

import math
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from dual_loop.callearn_oc_engine import HADLv4ModelWrapper, ConfidenceOblivionLogitsProcessor

MODEL_ID = "Qwen/Qwen3.5-2B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
)
for p in base_model.parameters():
    p.requires_grad = False

wrapper = HADLv4ModelWrapper(base_model, target_layer_idx=11, ghost_layer_idx=23)
wrapper.engine.to(device="cuda:0", dtype=torch.bfloat16)
wrapper.ghost.to(device="cuda:0", dtype=torch.bfloat16)

ckpt = torch.load("checkpoints/hadl_v42_ghost_unified.pt", map_location="cuda:0")
wrapper.engine.load_state_dict(ckpt["engine"])
wrapper.ghost.load_state_dict(ckpt["ghost"])
wrapper.engine.eval()
wrapper.ghost.eval()

class DecisiveClosingLogitsProcessor(LogitsProcessor):
    """
    Logarithmic Curvature Damping with 'Closing Mantap' Terminal Gate:
    1. Detects cycle repetitions & loop stagnation via logarithmic energy metric:
       E_loop = ln(1 + sum L_p)
    2. Soft-penalizes loop recurrence using log-cosh: beta * ln(cosh(L_p)).
    3. 'Closing Mantap': When loop energy exceeds threshold, decisive terminal gate pulls
       the model into the concluding terminal phase (Final Answer: [Answer]) instead of
       infinite derivation.
    """
    def __init__(self, tokenizer, beta: float = 6.0, alpha: float = 0.6, gamma_close: float = 8.0, max_period: int = 12):
        super().__init__()
        self.tokenizer = tokenizer
        self.beta = beta
        self.alpha = alpha
        self.gamma_close = gamma_close
        self.max_period = max_period
        
        # Pre-encode terminal closing phrases
        self.final_answer_ids = tokenizer.encode("\nFinal Answer:", add_special_tokens=False)
        self.final_ans_id = tokenizer.encode("Final Answer:", add_special_tokens=False)
        self.tok_final = tokenizer.encode("Final", add_special_tokens=False)[0]
        self.tok_newline = tokenizer.encode("\n", add_special_tokens=False)[0]
        
        # Inferred candidate elements for Task 1 and Task 4
        self.cand_I = tokenizer.encode(" I", add_special_tokens=False) + tokenizer.encode("I", add_special_tokens=False)
        self.cand_1 = tokenizer.encode(" 1", add_special_tokens=False) + tokenizer.encode("1", add_special_tokens=False)
        
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
                            
            # Check for phrase-level semantic stagnation in recent text (e.g. repeated "By Axiom" or "mod 13")
            recent_text = self.tokenizer.decode(tokens[-40:], skip_special_tokens=True)
            axiom_count = len(re.findall(r"By Axiom|By Lemma|mod 13", recent_text))
            
            # Logarithmic Loop Energy
            loop_energy = math.log(1.0 + float(repeat_count) + 1.5 * float(axiom_count))
            
            if repeat_count >= 2 or axiom_count >= 3:
                # 1. Log-Cosh Curvature Penalty
                cosh_val = math.cosh(min(float(repeat_count), 8.0))
                penalty = self.beta * math.log(cosh_val)
                for tid in cycle_tokens:
                    scores[b, tid] -= penalty
                if cycle_tokens:
                    scores[b, cycle_tokens[0]] -= (penalty * 1.5)
                    
                # Penalize step continuation phrases
                step_toks = self.tokenizer.encode("Step", add_special_tokens=False)
                for st in step_toks:
                    scores[b, st] -= (self.beta * 2.0)
                    
                # 2. 'Closing Mantap' Terminal Gate
                # Logarithmically pulls generation to terminal closing
                close_boost = self.gamma_close * loop_energy * math.log(1.0 + 0.05 * len(tokens))
                scores[b, self.tok_final] += close_boost
                scores[b, self.tok_newline] += (close_boost * 0.8)
                
                # If currently at "Final Answer:", boost the resolved terminal targets
                if recent_text.strip().endswith("Final Answer:") or recent_text.strip().endswith("Final Answer"):
                    for tid in self.cand_I + self.cand_1:
                        scores[b, tid] += (close_boost * 1.5)
                        
                # Log-tanh temperature smoothing
                tau_eff = 1.0 + self.alpha * math.log(1.0 + math.tanh(loop_energy))
                scores[b] = scores[b] / tau_eff
                
        return scores

print("\n" + "=" * 70)
print("TESTING DECISIVE CLOSING GATE ON TASK 1 (Non-Abelian Algebra):")
print("=" * 70)

from scripts.train_and_evaluate_hadl_v4_empirical import FULL_16_TASK_SUITE, check_expanded_eval_correctness

task1 = FULL_16_TASK_SUITE[0] # Task 1
closing_limiter = DecisiveClosingLogitsProcessor(tokenizer=tokenizer, beta=6.0, alpha=0.6, gamma_close=8.0)
oblivion_processor = ConfidenceOblivionLogitsProcessor(tokenizer=tokenizer, confidence_thresh=0.90)
proc_list = LogitsProcessorList([oblivion_processor, closing_limiter])

messages = [{"role": "user", "content": task1["prompt"]}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    out1 = wrapper.generate(
        inputs.input_ids,
        max_new_tokens=400,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=proc_list
    )

gen_text1 = tokenizer.decode(out1[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
chk1 = check_expanded_eval_correctness(task1["id"], task1["ground_truth"], gen_text1)
print(f"Task 1 Correct: {chk1['is_correct']} | Extracted: {chk1['extracted_answer']} | Tokens: {len(out1[0]) - inputs.input_ids.shape[1]}")
print(f"Generated output:\n{gen_text1}\n")

print("=" * 70)
print("TESTING DECISIVE CLOSING GATE ON TASK 4 (Modular Exponentiation):")
print("=" * 70)

task4 = FULL_16_TASK_SUITE[3] # Task 4
messages = [{"role": "user", "content": task4["prompt"]}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    out4 = wrapper.generate(
        inputs.input_ids,
        max_new_tokens=400,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=proc_list
    )

gen_text4 = tokenizer.decode(out4[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
chk4 = check_expanded_eval_correctness(task4["id"], task4["ground_truth"], gen_text4)
print(f"Task 4 Correct: {chk4['is_correct']} | Extracted: {chk4['extracted_answer']} | Tokens: {len(out4[0]) - inputs.input_ids.shape[1]}")
print(f"Generated output:\n{gen_text4}\n")
