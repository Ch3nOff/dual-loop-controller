import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('.')
from scripts.evaluate_option1_extended_tokens import TEST_TASKS, check_answer_correctness

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'Qwen/Qwen3.5-2B'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)

print('=== BASE MODEL EVALUATION WITH 1000 TOKENS ===')
for idx, task in enumerate(TEST_TASKS, 1):
    messages = [{'role': 'user', 'content': task['prompt']}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    t0 = time.time()
    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=1000, do_sample=False)
    lat = time.time() - t0
    gen_ids = out[0][inputs.input_ids.shape[1]:]
    resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
    metrics = check_answer_correctness(task['id'], task['ground_truth'], resp)
    print(f"Task {idx}: {task['name']}")
    print(f"  Result: {metrics['extracted_answer']} | Correct: {metrics['is_correct']} | Tokens: {len(gen_ids)} | Lat: {lat:.2f}s")
    print("  Tail:", resp.splitlines()[-2:] if resp.splitlines() else resp)
