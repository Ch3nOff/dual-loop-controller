import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "dist/xstar_2b_autonomous_brain"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)

question = "Bagikan 1 kutipan inspiratif tentang kerja keras dan siapa tokohnya."
prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Test 1: with max_new_tokens = 35 (the script limit)
out_35 = model.generate(**inputs, max_new_tokens=35, do_sample=False)
text_35 = tokenizer.decode(out_35[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

# Test 2: with natural max_new_tokens = 100 (relaxed limit, waiting for natural EOS)
out_100 = model.generate(**inputs, max_new_tokens=100, do_sample=False)
text_100 = tokenizer.decode(out_100[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

print("=" * 60)
print(f"[TEST 1: max_new_tokens = 35]")
print(f"Generated tokens: {len(out_35[0]) - inputs['input_ids'].shape[1]}")
print(f"Output: {repr(text_35)}")
print(f"Ends with EOS?: {'<|im_end|>' in text_35}")
print("=" * 60)
print(f"[TEST 2: max_new_tokens = 100]")
print(f"Generated tokens: {len(out_100[0]) - inputs['input_ids'].shape[1]}")
print(f"Output: {repr(text_100)}")
print(f"Ends with EOS?: {'<|im_end|>' in text_100}")
print("=" * 60)
