import sys
sys.path.insert(0, r"c:\Users\Matthew Chen\Documents\X-Star")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList
from dual_loop.callearn_oc_engine import (
    HADLv4ModelWrapper,
    ConfidenceOblivionLogitsProcessor,
    LogCoshCurvatureLimiter
)
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

oblivion_processor = ConfidenceOblivionLogitsProcessor(tokenizer=tokenizer, confidence_thresh=0.90, lambda_decay=3.0)
curvature_processor = LogCoshCurvatureLimiter(beta=4.0, alpha=0.5, max_period=12)
logits_processors = LogitsProcessorList([oblivion_processor, curvature_processor])

# Test Task 3 (Crypto - Heavy), Task 6 (VM POP - Heavy), Task 4 (Modular - Bypass), Task 11 (Cookies - Bypass)
test_indices = [2, 5, 3, 10] # Task 3, Task 6, Task 4, Task 11

for idx in test_indices:
    task = FULL_16_TASK_SUITE[idx]
    messages = [{"role": "user", "content": task["prompt"]}]
    txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(txt, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        out = wrapper.generate(
            inp.input_ids,
            max_new_tokens=600,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processors
        )
    gen = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    chk = check_expanded_eval_correctness(task["id"], task["ground_truth"], gen)
    
    route_status = "FAST BYPASS" if wrapper.is_fast_bypass else "HEAVY HADL"
    print(f"Task {idx+1:2d} ({task['name'][:28]:<28}) | Route: {route_status:<11} | Correct: {chk['is_correct']} | Got: {chk['extracted_answer']} | Tokens: {len(out[0]) - inp.input_ids.shape[1]}")
    print(f"      Snippet: {gen.strip()[-120:]}\n")
