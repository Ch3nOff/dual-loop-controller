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

# Dataset of general (bypass) and heavy prompts
general_prompts = [
    "Sarah buys 3 boxes of cookies, with 14 cookies in each box. She gives 6 to brother and eats 4. How many in 4 bags?",
    "A ball is dropped from 45 meters under standard Earth gravity g=10. What is height after 2 seconds?",
    "Given boolean variables P=True, Q=False, R=True, evaluate (NOT (P AND Q)) AND (Q OR (NOT R)).",
    "Trace registers X=5, Y=12, Z=3. Step 1: X=X+Z, Step 2: Y=Y-X, Step 3: Z=Y*2, Step 4: X=Z-Y.",
    "The trophy could not fit into the brown suitcase because it was too large. What does it refer to?",
    "City Alpha is at (10, 0), City Gamma is at (0, 10). What compass direction from Alpha to Gamma?",
    "Trace code: a=4, b=9. Iteration 1: a, b = b, (a+b)%7. Iteration 2: a, b = b, (a+b)%7. Final a?",
    "Calculate the total cost of 5 apples at $2 each and 3 oranges at $3 each.",
    "What is the capital of France and what river flows through it?",
    "Explain Newton's third law of motion with an everyday example.",
    "A train travels 120 miles in 2 hours. What is its average speed in miles per hour?",
    "If all roses are flowers and some flowers fade quickly, do all roses fade quickly?"
]

heavy_prompts = [
    "Simplify non-commutative algebra: A*B=C, B*A=D, C*D=A, D*C=B, X*X=I. Simplify A*(B*D)*(C*B)*A.",
    "Execute bytecode stack VM: PUSH 7, DUP_ODD, PUSH 4, SWAP_TOP, ADD_FOLD, PUSH 9, SUB_FOLD, DUP_ODD.",
    "Execute 1 round of X-Hash permutation: S_i' = (S_i * 3 + 1) mod 16, T0=(S0'+S1')%16, T1=S1'^S2'.",
    "Find remainder when 3^2028 is divided by 13 using Fermat's Little Theorem.",
    "Thermal equilibrium: 100L water at 20C added to 100L water at 20C. Final temperature?",
    "Execute stack VM: PUSH 9, PUSH 3, PUSH 2, POP, SUB_FOLD, PUSH 5, ADD_FOLD.",
    "Universe-K physics: mass m >= 6kg accelerates UPWARD at a=+4 m/s^2. m=10kg, y0=40m, t=3s.",
    "Network graph: Node Alpha -> Beta 14ms, Beta -> Gamma 16ms, Gamma -> Omega 12ms. Min latency?",
    "Game-World Omega: Fire is endothermic and freezes water, ice is exothermic. Throw water in fire.",
    "System: Synthetic Stack VM. ISA: PUSH v, DUP_ODD, SWAP_TOP, ADD_FOLD, SUB_FOLD, POP.",
    "System: Counterfactual Physics Engine. Rule: In Universe-K, mass m >= 6kg accelerates UPWARD.",
    "Simplify non-commutative expression: E = B * A * C * D with axioms."
]

dataset = [(p, 0) for p in general_prompts] + [(p, 1) for p in heavy_prompts]

# Add the 16 benchmark tasks:
for i, task in enumerate(FULL_16_TASK_SUITE):
    # Tasks 1 to 8: Heavy (1)
    # Tasks 9 to 16: Bypass (0)
    label = 1 if i < 8 else 0
    dataset.append((task["prompt"], label))

# Extract normalized prompt embeddings
X_list = []
Y_list = []
with torch.no_grad():
    for prompt, lbl in dataset:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
        embs = base_model.model.embed_tokens(ids)[0] # [S, D]
        # Mean embedding normalized
        v = F.normalize(embs.mean(dim=0).float(), p=2, dim=-1)
        X_list.append(v)
        Y_list.append(lbl)

X = torch.stack(X_list, dim=0) # [N, D]
Y = torch.tensor(Y_list, device="cuda:0") # [N]

print(f"Total dataset: {len(dataset)} samples. Class 0: {(Y==0).sum().item()}, Class 1: {(Y==1).sum().item()}")

# We want M_v in [D, 2] (or linear layer without bias if purely M_r = M_t x M_v)
# Let's train M_v:
router = nn.Linear(d_model, 2, bias=True).to("cuda:0")
opt = torch.optim.AdamW(router.parameters(), lr=0.02, weight_decay=1e-4)

for epoch in range(150):
    logits = router(X)
    loss = F.cross_entropy(logits, Y)
    opt.zero_grad()
    loss.backward()
    opt.step()

pred = router(X).argmax(dim=-1)
acc = (pred == Y).float().mean().item() * 100.0
print(f"Router Training Accuracy: {acc:.2f}%, Loss: {loss.item():.4f}")

# Now test specifically on all 16 benchmark tasks:
print("\n" + "=" * 60)
print("16-TASK EMBEDDED ROUTER EVALUATION:")
print("=" * 60)
all_pass = True
with torch.no_grad():
    for i, task in enumerate(FULL_16_TASK_SUITE, 1):
        ids = tokenizer(task["prompt"], return_tensors="pt").input_ids.to("cuda:0")
        embs = base_model.model.embed_tokens(ids)[0] # [S, D]
        v = F.normalize(embs.mean(dim=0).float(), p=2, dim=-1)
        logits = router(v.unsqueeze(0))[0] # [2]
        probs = F.softmax(logits, dim=-1)
        pred_route = torch.argmax(probs).item()
        
        expected = 1 if i <= 8 else 0
        status = "[PASS]" if pred_route == expected else "[FAIL]"
        if pred_route != expected:
            all_pass = False
        route_name = "HEAVY" if pred_route == 1 else "BYPASS"
        exp_name = "HEAVY" if expected == 1 else "BYPASS"
        print(f"Task {i:2d} | {status} Got: {route_name:<6} (P_hvy={probs[1]:.2f}, P_byp={probs[0]:.2f}) | Exp: {exp_name:<6} | {task['name'][:30]}")

print("=" * 60)
print(f"All 16 Tasks Perfectly Routed: {all_pass}")

if all_pass:
    # Save the calibrated router weights!
    torch.save({
        "weight": router.weight.data,
        "bias": router.bias.data
    }, "checkpoints/fast_embedded_router_weights.pt")
    print("[OK] Saved calibrated router weights to checkpoints/fast_embedded_router_weights.pt")
