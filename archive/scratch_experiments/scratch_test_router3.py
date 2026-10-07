import sys
sys.path.insert(0, r"c:\Users\Matthew Chen\Documents\X-Star")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.train_and_evaluate_hadl_v4_empirical import FULL_16_TASK_SUITE

MODEL_ID = "Qwen/Qwen3.5-2B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
)

embed_weight = base_model.model.embed_tokens.weight.data # [248077, 2048]
d_model = embed_weight.shape[1]

# Specific trigger phrases for the heavy routes:
# If a prompt triggers ANY heavy domain, it routes to that Heavy route.
# Otherwise, it routes to Fast Bypass (Route 0)!
heavy_domain_triggers = {
    1: ["X-Hash", "cryptographic permutation", "S0'", "S1'", "T0", "T1", "XOR"], # Crypto
    2: ["Stack VM", "PUSH", "POP", "DUP_ODD", "SWAP_TOP", "ADD_FOLD", "SUB_FOLD", "bytecode"], # Stack VM
    3: ["Non-Commutative Algebra", "Axioms:", "Universe-K", "irreducible element", "Game-World Omega"] # Axiom / Counterfactual
}

def embed_phrase(phrase: str) -> torch.Tensor:
    ids = tokenizer.encode(phrase, add_special_tokens=False)
    vecs = embed_weight[ids].float()
    return F.normalize(vecs.mean(dim=0), p=2, dim=-1)

# Prototype vectors for Route 1, Route 2, Route 3
p_crypto = torch.stack([embed_phrase(p) for p in heavy_domain_triggers[1]]).mean(dim=0)
p_crypto = F.normalize(p_crypto, p=2, dim=-1)

p_vm = torch.stack([embed_phrase(p) for p in heavy_domain_triggers[2]]).mean(dim=0)
p_vm = F.normalize(p_vm, p=2, dim=-1)

p_axiom = torch.stack([embed_phrase(p) for p in heavy_domain_triggers[3]]).mean(dim=0)
p_axiom = F.normalize(p_axiom, p=2, dim=-1)

# Construct M_v: [D, 3] for heavy domains
M_v = torch.stack([p_crypto, p_vm, p_axiom], dim=1).to("cuda:0", dtype=torch.bfloat16) # [D, 3]
beta = torch.tensor([1.2, 1.2, 1.2], device="cuda:0", dtype=torch.bfloat16)

print("Evaluating M_r = M_t x M_v * exp(beta):")
for i, task in enumerate(FULL_16_TASK_SUITE, 1):
    messages = [{"role": "user", "content": task["prompt"]}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")
    
    with torch.no_grad():
        M_t = base_model.model.embed_tokens(input_ids) # [1, S, D]
        M_t_norm = F.normalize(M_t, p=2, dim=-1)
        
        # M_r = M_t x M_v
        M_r = torch.matmul(M_t_norm, M_v) # [1, S, 3]
        M_r_scaled = M_r * torch.exp(beta) # [1, S, 3]
        
        # Peak similarity per heavy route
        max_sim_per_route, _ = torch.max(M_r_scaled[0], dim=0) # [3]
        max_heavy_sim = torch.max(max_sim_per_route).item()
        
        # Threshold: if max_heavy_sim >= threshold -> Heavy Route, else -> Fast Bypass (Route 0)
        is_heavy = (max_heavy_sim >= 0.70)
        route_str = "Heavy" if is_heavy else "Fast Bypass"
        
    expected = "Heavy" if i in [1, 2, 3, 5, 6, 7, 8, 13] else "Fast Bypass"
    print(f"Task {i:2d} ({task['domain'][:20]:<20}) -> {route_str:<12} (max_sim={max_heavy_sim:.3f}) | {task['name'][:28]}")
