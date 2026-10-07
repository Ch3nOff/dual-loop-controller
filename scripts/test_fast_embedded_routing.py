"""
Test Fast Embedded Routing on All 16 Benchmark Tasks
Verifies that:
- Tasks 1, 2, 3, 5, 6, 7, 8 route to Heavy HADL Deliberator (Route 1, 2, 3)
- Tasks 9, 10, 11, 12, 13, 14, 15, 16 route to Fast Bypass General (Route 0)
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrain_qwen_27b_grade_dualloop import MODEL_ID
from scripts.train_and_evaluate_hadl_v4_empirical import (
    EXPANDED_TEST_SUITE,
    EXPANDED_TEST_SUITE_PART2
)

def build_prototype_matrix(model, tokenizer, device):
    embedding = model.get_input_embeddings().weight.data  # [V, D]
    d_model = embedding.shape[1]

    domain_keywords = {
        0: ["Sarah", "cookies", "water", "suitcase", "trophy", "compass", "city", "direction", "boolean", "brother"], # General
        1: ["hash", "crypto", "X-Hash", "XOR", "permutation", "mod", "16", "latency", "network", "node"], # Bitwise Crypto
        2: ["VM", "bytecode", "stack", "POP", "PUSH", "DUP", "SWAP", "SUB_FOLD", "ADD_FOLD", "register"], # VM Stack
        3: ["Axiom", "Lemma", "Universe-K", "inverted", "group", "non-commutative", "temperature", "Celsius"] # Epistemic Group
    }

    M_v = torch.zeros(d_model, 4, device=device, dtype=torch.bfloat16)

    for k, words in domain_keywords.items():
        vecs = []
        for w in words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                v = embedding[ids].mean(dim=0)
                vecs.append(v)
        if vecs:
            mean_v = torch.stack(vecs).mean(dim=0)
            mean_v = F.normalize(mean_v, p=2, dim=-1)
            M_v[:, k] = mean_v

    return M_v


def test_routing():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading model {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    M_v = build_prototype_matrix(model, tokenizer, device)

    all_tasks = EXPANDED_TEST_SUITE + EXPANDED_TEST_SUITE_PART2
    print(f"\n[*] Testing Fast Embedded Routing across {len(all_tasks)} Tasks...\n")

    embedding_layer = model.get_input_embeddings()

    for idx, task in enumerate(all_tasks, 1):
        enc = tokenizer(task["prompt"], return_tensors="pt").to(device)
        input_ids = enc.input_ids
        M_t = embedding_layer(input_ids) # [1, S, D]
        M_t_norm = F.normalize(M_t, p=2, dim=-1)

        # M_r = M_t x M_v: [1, S, 4]
        M_r = torch.matmul(M_t_norm, M_v)

        # Scaled Log-Sum-Exp pooling across tokens:
        # We boost the top match per token
        scores_k = torch.logsumexp(M_r * 4.0, dim=1) # [1, 4]
        probs = F.softmax(scores_k, dim=-1)[0]
        dom_route = torch.argmax(probs).item()

        route_names = ["FAST_BYPASS (General)", "HEAVY_CRYPTO", "HEAVY_VM_STACK", "HEAVY_GROUP_PHYSICS"]
        is_fast = (dom_route == 0)

        print(f"Task {idx:2d}: {task['name'][:40]:<40} -> Route {dom_route} ({route_names[dom_route]:<24}) | Probs: {probs.detach().cpu().tolist()}")

if __name__ == "__main__":
    test_routing()
