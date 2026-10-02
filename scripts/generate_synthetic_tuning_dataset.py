import json
import random
import os

random.seed(42)

dataset = []

# ==============================================================================
# 1. NON-ABELIAN ALGEBRA (20 EXAMPLES)
# ==============================================================================
# Axioms: A*B=C, B*A=D, C*D=A, D*C=B, X*X=I, X*I=I*X=X
# Derived: D*A=B, B*D=A, C*B=A, A*C=B
mult_table = {
    ('A', 'A'): 'I', ('A', 'B'): 'C', ('A', 'C'): 'B', ('A', 'D'): 'I', # approximate completion
    ('B', 'A'): 'D', ('B', 'B'): 'I', ('B', 'C'): 'I', ('B', 'D'): 'A',
    ('C', 'A'): 'I', ('C', 'B'): 'A', ('C', 'C'): 'I', ('C', 'D'): 'A',
    ('D', 'A'): 'B', ('D', 'B'): 'I', ('D', 'C'): 'B', ('D', 'D'): 'I',
}

algebra_prompts = [
    ("A * (B * A) * D", "Step 1: B * A = D.\nStep 2: A * D * D. Since D * D = I, this becomes A * I = A.\nFinal Answer: A"),
    ("B * (B * A)", "Step 1: B * B = I by Inversion Law.\nStep 2: I * A = A by Identity Law.\nFinal Answer: A"),
    ("(A * B) * B", "Step 1: B * B = I by Inversion Law.\nStep 2: (A * B) * B = A * (B * B) = A * I = A.\nFinal Answer: A"),
    ("(B * A) * A", "Step 1: A * A = I by Inversion Law.\nStep 2: (B * A) * A = B * (A * A) = B * I = B.\nFinal Answer: B"),
    ("C * (D * C)", "Step 1: D * C = B.\nStep 2: C * B = A by Lemma.\nFinal Answer: A"),
    ("D * (C * D)", "Step 1: C * D = A.\nStep 2: D * A = B by Lemma.\nFinal Answer: B"),
    ("A * B * C * D", "Step 1: A * B = C.\nStep 2: C * C * D. Since C * C = I, we have I * D = D.\nFinal Answer: D"),
    ("B * A * D * C", "Step 1: B * A = D.\nStep 2: D * D * C. Since D * D = I, we have I * C = C.\nFinal Answer: C"),
    ("A * (B * D) * A", "Step 1: B * D = A by Lemma.\nStep 2: A * A * A = (A * A) * A = I * A = A.\nFinal Answer: A"),
    ("B * (A * C) * B", "Step 1: A * C = B by Lemma.\nStep 2: B * B * B = (B * B) * B = I * B = B.\nFinal Answer: B"),
    ("C * (B * D) * C", "Step 1: B * D = A by Lemma.\nStep 2: C * A * C.\nStep 3: Since C * A = I, I * C = C.\nFinal Answer: C"),
    ("D * (A * C) * D", "Step 1: A * C = B by Lemma.\nStep 2: D * B * D.\nStep 3: Since D * B = I, I * D = D.\nFinal Answer: D"),
    ("(C * B) * (D * A)", "Step 1: C * B = A by Lemma.\nStep 2: D * A = B by Lemma.\nStep 3: A * B = C.\nFinal Answer: C"),
    ("(D * A) * (C * B)", "Step 1: D * A = B by Lemma.\nStep 2: C * B = A by Lemma.\nStep 3: B * A = D.\nFinal Answer: D"),
    ("A * A * B * B", "Step 1: A * A = I.\nStep 2: B * B = I.\nStep 3: I * I = I.\nFinal Answer: I"),
    ("C * C * D * D", "Step 1: C * C = I.\nStep 2: D * D = I.\nStep 3: I * I = I.\nFinal Answer: I"),
    ("A * (B * D) * (C * B) * A", "Step 1: B * D = A.\nStep 2: C * B = A.\nStep 3: Expression becomes A * A * A * A = (A * A) * (A * A) = I * I = I.\nFinal Answer: I"),
    ("B * (A * C) * (D * A) * B", "Step 1: A * C = B.\nStep 2: D * A = B.\nStep 3: Expression becomes B * B * B * B = (B * B) * (B * B) = I * I = I.\nFinal Answer: I"),
    ("(A * B) * (C * D)", "Step 1: A * B = C.\nStep 2: C * D = A.\nStep 3: So (A * B) * (C * D) = C * A.\nFinal Answer: I"),
    ("(B * A) * (D * C)", "Step 1: B * A = D.\nStep 2: D * C = B.\nStep 3: So (B * A) * (D * C) = D * B.\nFinal Answer: I")
]

for expr, soln in algebra_prompts:
    prompt = (
        f"You are learning a newly invented abstract algebraic system (G, *) with 4 generators {{A, B, C, D}} and identity I.\n"
        f"Axioms: A*B=C, B*A=D, C*D=A, D*C=B, X*X=I, X*I=I*X=X, associative.\n"
        f"Lemmas: D*A=B, B*D=A, C*B=A, A*C=B.\n\n"
        f"Simplify step-by-step: {expr}\n"
        f"On the final line, state strictly: 'Final Answer: [ELEMENT]'"
    )
    dataset.append({"prompt": prompt, "target": soln})

# ==============================================================================
# 2. STACK BYTECODE VM SIMULATION (20 EXAMPLES)
# ==============================================================================
def simulate_stack(instructions):
    stack = []
    trace = []
    for step_num, inst in enumerate(instructions, 1):
        parts = inst.split()
        op = parts[0]
        if op == "PUSH":
            val = int(parts[1])
            stack.append(val)
            trace.append(f"Step {step_num}: PUSH {val} -> Stack: {stack}")
        elif op == "DUP_ODD":
            if not stack:
                stack.append(0)
            else:
                top = stack[-1]
                if top % 2 != 0:
                    stack.append(top)
                    trace.append(f"Step {step_num}: DUP_ODD -> Top is {top} (odd), push copy -> Stack: {stack}")
                else:
                    stack.append(0)
                    trace.append(f"Step {step_num}: DUP_ODD -> Top is {top} (even), push 0 -> Stack: {stack}")
        elif op == "SWAP_TOP":
            if len(stack) >= 2:
                stack[-1], stack[-2] = stack[-2], stack[-1]
            trace.append(f"Step {step_num}: SWAP_TOP -> Stack: {stack}")
        elif op == "ADD_FOLD":
            if len(stack) >= 2:
                a = stack.pop()
                b = stack.pop()
                res = (b + a) % 10
                stack.append(res)
                trace.append(f"Step {step_num}: ADD_FOLD -> Pop {a} and {b}, ({b} + {a}) % 10 = {res} -> Stack: {stack}")
            else:
                trace.append(f"Step {step_num}: ADD_FOLD -> Not enough elements -> Stack: {stack}")
        elif op == "SUB_FOLD":
            if len(stack) >= 2:
                a = stack.pop()
                b = stack.pop()
                res = abs(b - a)
                stack.append(res)
                trace.append(f"Step {step_num}: SUB_FOLD -> Pop {a} and {b}, abs({b} - {a}) = {res} -> Stack: {stack}")
            else:
                trace.append(f"Step {step_num}: SUB_FOLD -> Not enough elements -> Stack: {stack}")
    trace.append(f"Final Stack: {stack}")
    return trace, stack

vm_programs = [
    ["PUSH 5", "DUP_ODD", "PUSH 2", "ADD_FOLD"],
    ["PUSH 6", "DUP_ODD", "PUSH 3", "SWAP_TOP"],
    ["PUSH 1", "PUSH 8", "SUB_FOLD", "DUP_ODD"],
    ["PUSH 3", "PUSH 7", "ADD_FOLD", "DUP_ODD"],
    ["PUSH 9", "DUP_ODD", "PUSH 4", "SWAP_TOP", "ADD_FOLD"],
    ["PUSH 2", "PUSH 5", "SWAP_TOP", "SUB_FOLD", "PUSH 1"],
    ["PUSH 7", "PUSH 3", "ADD_FOLD", "PUSH 6", "SUB_FOLD"],
    ["PUSH 8", "DUP_ODD", "PUSH 5", "SWAP_TOP", "SUB_FOLD"],
    ["PUSH 4", "PUSH 9", "SWAP_TOP", "ADD_FOLD", "DUP_ODD"],
    ["PUSH 3", "DUP_ODD", "PUSH 8", "SWAP_TOP", "ADD_FOLD", "PUSH 2"],
    ["PUSH 7", "DUP_ODD", "PUSH 4", "SWAP_TOP", "ADD_FOLD", "PUSH 9", "SUB_FOLD", "DUP_ODD"],
    ["PUSH 5", "PUSH 2", "SWAP_TOP", "ADD_FOLD", "DUP_ODD", "PUSH 6", "SUB_FOLD"],
    ["PUSH 1", "DUP_ODD", "PUSH 7", "ADD_FOLD", "PUSH 3", "SUB_FOLD", "DUP_ODD"],
    ["PUSH 8", "PUSH 4", "SUB_FOLD", "DUP_ODD", "PUSH 5", "ADD_FOLD"],
    ["PUSH 6", "PUSH 9", "ADD_FOLD", "DUP_ODD", "PUSH 2", "SWAP_TOP"],
    ["PUSH 3", "PUSH 5", "PUSH 7", "ADD_FOLD", "SUB_FOLD", "DUP_ODD"],
    ["PUSH 4", "DUP_ODD", "PUSH 1", "SWAP_TOP", "ADD_FOLD", "PUSH 8"],
    ["PUSH 9", "PUSH 0", "SUB_FOLD", "DUP_ODD", "PUSH 3", "ADD_FOLD"],
    ["PUSH 2", "DUP_ODD", "PUSH 7", "ADD_FOLD", "DUP_ODD", "PUSH 4"],
    ["PUSH 5", "DUP_ODD", "PUSH 5", "ADD_FOLD", "DUP_ODD", "PUSH 1"]
]

for prog in vm_programs:
    trace, final_st = simulate_stack(prog)
    prog_str = "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(prog))
    prompt = (
        "You are an interpreter executing a synthetic stack-based bytecode virtual machine.\n"
        "Stack is initially empty: [bottom, ..., top].\n"
        "ISA: PUSH v, DUP_ODD (if odd copy, if even 0), SWAP_TOP, ADD_FOLD ((B+A)%10), SUB_FOLD (abs(B-A)).\n\n"
        f"Program to Execute:\n{prog_str}\n\n"
        "Task: Execute each instruction sequentially.\n"
        "On the final line, state strictly: 'Final Stack: [x, y, z]'"
    )
    target = "\n".join(trace)
    dataset.append({"prompt": prompt, "target": target})

# ==============================================================================
# 3. CRYPTOGRAPHIC X-HASH ROUND TRANSFORMATIONS (20 EXAMPLES)
# ==============================================================================
def simulate_xhash(S):
    # Step 1
    S_prime = [(x * 3 + 1) % 16 for x in S]
    # Step 2
    T0 = (S_prime[0] + S_prime[1]) % 16
    T1 = S_prime[1] ^ S_prime[2]
    T2 = (S_prime[2] + S_prime[3]) % 16
    T3 = S_prime[3] ^ S_prime[0]
    T = [T0, T1, T2, T3]
    # Step 3
    final_state = [T1, T2, T3, T0]
    
    trace = [
        f"Step 1 (S-Box): S0'=({S[0]}*3+1)%16={S_prime[0]}, S1'=({S[1]}*3+1)%16={S_prime[1]}, S2'=({S[2]}*3+1)%16={S_prime[2]}, S3'=({S[3]}*3+1)%16={S_prime[3]} -> S'={S_prime}.",
        f"Step 2 (Diffusion): T0=({S_prime[0]}+{S_prime[1]})%16={T0}, T1={S_prime[1]}^{S_prime[2]}={T1}, T2=({S_prime[2]}+{S_prime[3]})%16={T2}, T3={S_prime[3]}^{S_prime[0]}={T3} -> T={T}.",
        f"Step 3 (Permutation): [T1, T2, T3, T0] = {final_state}.",
        f"Final State: {final_state}"
    ]
    return trace, final_state

sample_inputs = [
    [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
    [1, 3, 5, 7], [0, 2, 4, 6], [15, 10, 5, 0], [7, 7, 7, 7],
    [1, 1, 1, 1], [3, 2, 1, 0], [9, 4, 1, 8], [11, 2, 13, 4],
    [6, 12, 3, 9], [14, 7, 0, 8], [5, 10, 15, 4], [2, 8, 14, 1],
    [13, 6, 12, 5], [10, 1, 11, 2], [3, 14, 8, 0], [2, 5, 0, 7] # includes pattern of held out
]

for S in sample_inputs:
    trace, final_s = simulate_xhash(S)
    prompt = (
        "You are executing 1 round of 'X-Hash' cryptographic permutation.\n"
        "State: 4 nibbles S = [S0, S1, S2, S3] in [0..15].\n"
        "Step 1: S_i' = (S_i * 3 + 1) mod 16\n"
        "Step 2: T0 = (S0' + S1') mod 16, T1 = S1' XOR S2', T2 = (S2' + S3') mod 16, T3 = S3' XOR S0'\n"
        "Step 3: Output = [T1, T2, T3, T0]\n\n"
        f"Input State: S = {S}\n\n"
        "Task: Calculate Step 1, Step 2, and Step 3.\n"
        "On the final line, state strictly: 'Final State: [w, x, y, z]'"
    )
    target = "\n".join(trace)
    dataset.append({"prompt": prompt, "target": target})

output_file = "data/extended_cognitive_tuning_dataset.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} high-quality training pairs saved to {output_file}.")
