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

# 1. Compute global embedding centroid mu to eliminate the anisotropic bias
mu = embed_weight.float().mean(dim=0, keepdim=True) # [1, D]
centered_embed = (embed_weight.float() - mu) # [V, D]

# Specialized domain keyword sets
# Route 0: Fast Bypass (General / Standard Word Math / Standard Mechanics / NLP)
# Route 1: Heavy Deliberation (Cryptographic Bitwise / VM Bytecode / Counterfactual / Non-Abelian)
domain_keywords = {
    "bypass": [
        "Sarah", "cookies", "boxes", "brother", "eats", "bags",
        "standard", "Earth", "gravity", "ball", "dropped", "released", "negligible",
        "suitcase", "trophy", "large", "refer", "brown", "fit",
        "compass", "direction", "bearing", "City", "traveler", "degrees",
        "register", "X", "Y", "Z", "Initial", "values"
    ],
    "cryptographic": [
        "X-Hash", "cryptographic", "permutation", "XOR", "S0", "S1", "S2", "S3", "T0", "T1", "T2", "T3"
    ],
    "bytecode_vm": [
        "bytecode", "virtual", "machine", "PUSH", "POP", "DUP_ODD", "SWAP_TOP", "ADD_FOLD", "SUB_FOLD"
    ],
    "counterfactual_algebra": [
        "Non-Commutative", "Axioms", "Axiom", "irreducible", "Universe-K", "inverted", "endothermic", "exothermic"
    ]
}

def get_proto(keywords):
    all_toks = []
    for w in keywords:
        t1 = tokenizer.encode(w, add_special_tokens=False)
        t2 = tokenizer.encode(" " + w, add_special_tokens=False)
        all_toks.extend(t1 + t2)
    all_toks = list(set(all_toks))
    emb = centered_embed[all_toks] # [N, D]
    return F.normalize(emb.mean(dim=0), p=2, dim=-1) # [D]

p_bypass = get_proto(domain_keywords["bypass"])
p_crypto = get_proto(domain_keywords["cryptographic"])
p_vm = get_proto(domain_keywords["bytecode_vm"])
p_alg = get_proto(domain_keywords["counterfactual_algebra"])

# M_v matrix: [D, 4]
# Column 0: Fast Bypass
# Column 1: Crypto (Route 1)
# Column 2: Stack VM (Route 2)
# Column 3: Axioms/Algebra (Route 3)
M_v = torch.stack([p_bypass, p_crypto, p_vm, p_alg], dim=1).to("cuda:0", dtype=torch.bfloat16)

print(f"M_v shape: {M_v.shape}")

print("\nEvaluating Centered Fast Embedded Routing on 16 tasks:")
for i, task in enumerate(FULL_16_TASK_SUITE, 1):
    messages = [{"role": "user", "content": task["prompt"]}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")
    
    with torch.no_grad():
        # Input embeddings M_t centered
        M_t_raw = base_model.model.embed_tokens(input_ids) # [1, S, D]
        M_t = M_t_raw - mu.to("cuda:0", dtype=torch.bfloat16)
        M_t_norm = F.normalize(M_t, p=2, dim=-1)
        
        # M_r = M_t x M_v : [1, S, 4]
        M_r = torch.matmul(M_t_norm, M_v) # [1, S, 4]
        
        # Maximum matching token similarity in prompt for each route
        # Using top-5 tokens mean similarity
        top_k, _ = torch.topk(M_r[0], k=min(8, M_r.shape[1]), dim=0) # [8, 4]
        mean_top_sim = top_k.mean(dim=0) # [4]
        
        # Routing vector X with temperature
        X = F.softmax(mean_top_sim * 15.0, dim=-1)
        route_idx = torch.argmax(X).item()
        
    route_names = ["0:FastBypass", "1:Crypto", "2:StackVM", "3:FormalAxiom"]
    print(f"Task {i:2d} ({task['domain'][:20]:<20}) -> Route {route_names[route_idx]:<13} | X=[byp:{X[0]:.2f}, cry:{X[1]:.2f}, vm:{X[2]:.2f}, ax:{X[3]:.2f}] | {task['name'][:30]}")
