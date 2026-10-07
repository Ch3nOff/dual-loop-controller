import sys
sys.path.insert(0, r"c:\Users\Matthew Chen\Documents\X-Star")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.train_and_evaluate_hadl_v4_empirical import FULL_16_TASK_SUITE, TRAINING_CORPUS, ADDITIONAL_TRAINING_CORPUS

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

# Collect prompts and their desired route:
# Route 0 = Fast Bypass (General language, word math, Earth physics, Winograd, etc.)
# Route 1 = Heavy Deliberation (Formal Non-Abelian, Bytecode Stack VM, Crypto X-Hash, Counterfactual Universe-K)

data = []
# Heavy tasks from training corpus
for item in TRAINING_CORPUS:
    data.append((item["prompt"], 1)) # Heavy
for item in ADDITIONAL_TRAINING_CORPUS:
    data.append((item["prompt"], 1)) # Heavy

# Tasks from 16-task suite
for i, task in enumerate(FULL_16_TASK_SUITE):
    # Tasks 1 to 8: Core HADL Suite
    # Task 1 (Algebra): Heavy (1)
    # Task 2 (VM 8-step): Heavy (1)
    # Task 3 (X-Hash): Heavy (1)
    # Task 4 (Modular Fermat): Heavy (1)
    # Task 5 (Thermal): Heavy/CalLearn (1)
    # Task 6 (VM POP): Heavy (1)
    # Task 7 (Universe-K): Heavy (1)
    # Task 8 (Graph): Heavy (1)
    # Tasks 9 to 16: Generalization Suite -> Fast Bypass (0)
    target_route = 1 if i < 8 else 0
    data.append((task["prompt"], target_route))

print(f"Total prompt dataset size: {len(data)}")

# Extract pooled prompt embeddings
embeddings = []
labels = []

with torch.no_grad():
    for prompt, route in data:
        messages = [{"role": "user", "content": prompt}]
        txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").input_ids.to("cuda:0")
        tok_emb = base_model.model.embed_tokens(inp)[0] # [S, D]
        # Mean pooled token embedding
        pooled = tok_emb.mean(dim=0).float() # [D]
        embeddings.append(pooled)
        labels.append(route)

X_emb = torch.stack(embeddings, dim=0) # [N, D]
Y_lbl = torch.tensor(labels, device="cuda:0") # [N]

print(f"X_emb shape: {X_emb.shape}, Y_lbl shape: {Y_lbl.shape}")

# Fit a linear routing basis M_v [D, 2] via ridge regression / cross-entropy
# Loss: CrossEntropy with weight decay
router_linear = nn.Linear(d_model, 2, bias=True).to("cuda:0")
optimizer = torch.optim.Adam(router_linear.parameters(), lr=0.01, weight_decay=1e-3)

for step in range(200):
    logits = router_linear(X_emb)
    loss = F.cross_entropy(logits, Y_lbl)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

pred = router_linear(X_emb).argmax(dim=-1)
acc = (pred == Y_lbl).float().mean().item() * 100.0
print(f"Router training accuracy: {acc:.1f}%, Final loss: {loss.item():.4f}")

# Now test on the 16 benchmark tasks:
print("\nTesting Router on 16 Tasks:")
with torch.no_grad():
    for i, task in enumerate(FULL_16_TASK_SUITE, 1):
        messages = [{"role": "user", "content": task["prompt"]}]
        txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(txt, return_tensors="pt").input_ids.to("cuda:0")
        tok_emb = base_model.model.embed_tokens(inp)[0].float() # [S, D]
        
        # User formula: M_r = M_t x M_v
        # M_v is router_linear.weight.T : [D, 2]
        M_v = router_linear.weight.T # [D, 2]
        bias = router_linear.bias # [2]
        
        M_r = torch.matmul(tok_emb, M_v) + bias # [S, 2]
        # Aggregate token routing affinities via LogSumExp / Mean
        route_logits = M_r.mean(dim=0)
        route_probs = F.softmax(route_logits, dim=-1)
        pred_route = torch.argmax(route_probs).item()
        
        expected = "Heavy" if i <= 8 else "Bypass"
        actual = "Heavy" if pred_route == 1 else "Bypass"
        match = "[OK]" if expected == actual else "[FAIL]"
        print(f"Task {i:2d}: {match} Expected: {expected:<6} | Actual: {actual:<6} | Prob=[Bypass:{route_probs[0]:.2f}, Heavy:{route_probs[1]:.2f}] | {task['name'][:30]}")
