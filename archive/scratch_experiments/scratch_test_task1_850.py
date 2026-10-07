import sys
sys.path.insert(0, r"c:\Users\Matthew Chen\Documents\X-Star")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.train_and_evaluate_hadl_v4_empirical import FULL_16_TASK_SUITE, check_expanded_eval_correctness

MODEL_ID = "Qwen/Qwen3.5-2B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
)

task1 = FULL_16_TASK_SUITE[0]
messages = [{"role": "user", "content": task1["prompt"]}]
txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inp = tokenizer(txt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    out = base_model.generate(
        inp.input_ids,
        max_new_tokens=850,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

gen = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
chk = check_expanded_eval_correctness(task1["id"], task1["ground_truth"], gen)
print(f"Task 1 with 850 tokens: is_correct = {chk['is_correct']}, Extracted: {chk['extracted_answer']}, Total tokens: {len(out[0]) - inp.input_ids.shape[1]}")
print(f"End of output:\n{gen[-350:]}")
