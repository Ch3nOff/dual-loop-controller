import sys
sys.path.insert(0, r"c:\Users\Matthew Chen\Documents\X-Star")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.benchmark_general_and_conversation import REAL_WORLD_BENCHMARK_TASKS, evaluate_task_quality
from scripts.retrain_qwen_27b_grade_dualloop import MODEL_ID

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)

print("=" * 80)
print("TESTING NATIVE BASE QWEN / FORCED FAST BYPASS ON 12 REAL-WORLD TASKS")
print("=" * 80)

passed = 0
for idx, task in enumerate(REAL_WORLD_BENCHMARK_TASKS, 1):
    messages = [{"role": "user", "content": task["prompt"]}]
    txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(txt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = base_model.generate(
            inp.input_ids,
            max_new_tokens=650,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_text = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Evaluate with is_bypass=True
    metrics = evaluate_task_quality(task, gen_text, is_bypass=True, route_probs=[1.0, 0.0])
    
    tag = "[PASS]" if metrics["is_passed"] else "[FAIL]"
    if metrics["is_passed"]:
        passed += 1
        
    print(f"[{idx:2d}/12] {tag} {task['category']:<22} | {task['name']}")
    print(f"       Score: {metrics['content_score_pct']}% | Anti-patterns: {metrics['detected_anti_patterns']}")
    print(f"       Preview: {gen_text[:160]}...\n")

print("=" * 80)
print(f"Summary on Forced Fast Bypass: {passed} / {len(REAL_WORLD_BENCHMARK_TASKS)} Passed ({passed/len(REAL_WORLD_BENCHMARK_TASKS)*100:.1f}%)")
print("=" * 80)
