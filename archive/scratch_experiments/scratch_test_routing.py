import sys
sys.path.insert(0, r"c:\Users\Matthew Chen\Documents\X-Star")

import torch
import torch.nn as nn
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

# Define anchor tokens for each route:
# Route 0: General / Standard (Bypass heavy adapter)
# Route 1: Cryptographic / Bitwise (X-Hash, S-box, XOR)
# Route 2: Bytecode Stack VM (PUSH, POP, DUP, STACK)
# Route 3: Formal Axioms & Counterfactuals (Axioms, non-abelian, Universe-K)

anchor_keywords = {
    0: ["general", "cookies", "boxes", "Sarah", "ball", "Earth", "gravity", "compass", "suitcase", "trophy", "register", "X", "Y", "Z"],
    1: ["cryptographic", "permutation", "X-Hash", "XOR", "mod", "16", "S0", "S1", "T0", "T1", "hash"],
    2: ["Stack", "VM", "PUSH", "POP", "DUP_ODD", "SWAP_TOP", "ADD_FOLD", "SUB_FOLD", "bytecode"],
    3: ["Axiom", "Axioms", "Non-Commutative", "Universe-K", "Fermat", "irreducible", "non-abelian", "algebra"]
}

# Build prototype vectors M_v: [D, 4]
prototypes = []
for r in range(4):
    words = anchor_keywords[r]
    all_tok_ids = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        toks_sp = tokenizer.encode(" " + w, add_special_tokens=False)
        all_tok_ids.extend(toks + toks_sp)
    all_tok_ids = list(set(all_tok_ids))
    emb = embed_weight[all_tok_ids].float() # [N, D]
    proto = F.normalize(emb.mean(dim=0), p=2, dim=-1) # [D]
    prototypes.append(proto)

M_v = torch.stack(prototypes, dim=1).to(device="cuda:0", dtype=torch.bfloat16) # [D, 4]
beta = torch.tensor([0.0, 0.5, 0.5, 0.5], device="cuda:0", dtype=torch.bfloat16) # [4]

print(f"M_v shape: {M_v.shape}")

print("\nEvaluating Fast Embedded Routing on 16 tasks:")
for i, task in enumerate(FULL_16_TASK_SUITE, 1):
    messages = [{"role": "user", "content": task["prompt"]}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")
    
    # Compute M_t: [1, S, D]
    with torch.no_grad():
        M_t = base_model.model.embed_tokens(input_ids) # [1, S, D]
        # Fast compile matrix multiplication: M_r = M_t x M_v
        M_r = torch.matmul(M_t, M_v) # [1, S, 4]
        # Multiply with exponent factor: M_r * exp(beta)
        M_r_exp = M_r * torch.exp(beta) # [1, S, 4]
        
        # Max-affinity across tokens for each route
        # Using top-5 mean or max pooling
        top_k_vals, _ = torch.topk(M_r_exp[0], k=min(10, M_r_exp.shape[1]), dim=0) # [10, 4]
        affinity = top_k_vals.mean(dim=0) # [4]
        
        # Softmax route probability X
        X = F.softmax(affinity * 2.0, dim=-1)
        pred_route = torch.argmax(X).item()
        
    print(f"Task {i:2d} ({task['domain'][:22]:<22}): route={pred_route} | X=[r0:{X[0]:.2f}, r1:{X[1]:.2f}, r2:{X[2]:.2f}, r3:{X[3]:.2f}] | name={task['name'][:28]}")
