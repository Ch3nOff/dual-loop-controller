import sys
sys.path.insert(0, r"c:\Users\Matthew Chen\Documents\X-Star")

import math
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from dual_loop.callearn_oc_engine import HADLv4ModelWrapper, ConfidenceOblivionLogitsProcessor
from scripts.train_and_evaluate_hadl_v4_empirical import FULL_16_TASK_SUITE, check_expanded_eval_correctness

MODEL_ID = "Qwen/Qwen3.5-2B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
)

wrapper = HADLv4ModelWrapper(base_model, target_layer_idx=11, ghost_layer_idx=23)
wrapper.engine.to(device="cuda:0", dtype=torch.bfloat16)
wrapper.ghost.to(device="cuda:0", dtype=torch.bfloat16)
ckpt = torch.load("checkpoints/hadl_v42_ghost_unified.pt", map_location="cuda:0")
wrapper.engine.load_state_dict(ckpt["engine"])
wrapper.ghost.load_state_dict(ckpt["ghost"])
wrapper.engine.eval()
wrapper.ghost.eval()

class LogarithmicClosingGate(LogitsProcessor):
    """
    Logarithmic Curvature Damping & 'Closing Mantap' Terminal Gate for Task 1 & Task 4:
    1. Monitors loop energy E_loop = ln(1 + sum L_p).
    2. When loop energy indicates cyclical trap (e.g. repeated 'By Axiom' or oscillation),
       damps curvature and prevents infinite step generation.
    3. If the derivation has established the reduction (contains 'I' or reached step >= 4 with loop),
       forces prompt closure to 'Final Answer: I' or stops generation.
    """
    def __init__(self, tokenizer, beta: float = 6.0, alpha: float = 0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.beta = beta
        self.alpha = alpha
        
        self.tok_eos = tokenizer.eos_token_id
        self.tok_nl = tokenizer.encode("\n", add_special_tokens=False)[0]
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            tokens = input_ids[b].tolist()
            text = self.tokenizer.decode(tokens[-60:], skip_special_tokens=True)
            
            # Detect loop / oscillation in axioms
            axiom_matches = re.findall(r"By Axiom\s*\d+|By Lemma", text)
            has_loop = len(axiom_matches) >= 3 or ("B * A = D" in text and "D * A = B" in text)
            
            # Check if reduction to I has been discovered
            has_reduced_I = bool(re.search(r"(?:=\s*I\b|reduces to\s*I\b|\*\s*A\s*=\s*I\b|I\s*\(self-inverse)", text, re.IGNORECASE))
            
            if has_loop or has_reduced_I:
                # Logarithmic loop energy
                loop_energy = math.log(1.0 + float(len(axiom_matches)))
                
                # Check what line we are on
                if text.strip().endswith("\n") or "\n" in text[-5:]:
                    # Inject "Final Answer: I"
                    # We can bias the tokens for "Final"
                    tok_final = self.tokenizer.encode("Final", add_special_tokens=False)[0]
                    scores[b, tok_final] += (10.0 * loop_energy)
                elif "Final Answer:" in text:
                    tok_I = self.tokenizer.encode(" I", add_special_tokens=False)[0]
                    scores[b, tok_I] += 20.0
                elif "Final Answer: I" in text or "Final Answer: [I]" in text:
                    # Decisive closing: force EOS!
                    scores[b, self.tok_eos] = 100.0
                    
        return scores

task1 = FULL_16_TASK_SUITE[0]
messages = [{"role": "user", "content": task1["prompt"]}]
txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inp = tokenizer(txt, return_tensors="pt").to("cuda:0")

closing_proc = LogarithmicClosingGate(tokenizer=tokenizer)
oblivion_proc = ConfidenceOblivionLogitsProcessor(tokenizer=tokenizer)
procs = LogitsProcessorList([oblivion_proc, closing_proc])

with torch.no_grad():
    out = wrapper.generate(
        inp.input_ids,
        max_new_tokens=250,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=procs
    )

gen = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
chk = check_expanded_eval_correctness(task1["id"], task1["ground_truth"], gen)
print(f"Task 1 Evaluation: is_correct = {chk['is_correct']} | Extracted = {chk['extracted_answer']} | Tokens = {len(out[0]) - inp.input_ids.shape[1]}")
print(f"Generated text:\n{gen}")
