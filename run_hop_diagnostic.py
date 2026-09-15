import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from run_experiment import MultiHopGraphDataset, StandardShallowTransformer, DualLoopV2_TransformerPrefix, set_seed

def run_hop_experiment():
    print("=" * 70)
    print("DIAGNOSTIC TEST: ACCURACY ACROSS HOP COMPLEXITY (H = 1, 2, 3)")
    print("Testing where Latent Recurrence works vs where it collapses")
    print("=" * 70)
    
    results_hop = {}
    for hop in [1, 2, 3]:
        set_seed(42)
        train_data = MultiHopGraphDataset(num_samples=2000, num_nodes=16, num_edges=6, hops=hop)
        test_data = MultiHopGraphDataset(num_samples=400, num_nodes=16, num_edges=6, hops=hop)
        vocab_size = train_data.vocab_size
        
        # 1. Shallow (L=1, K=0)
        set_seed(42)
        m_shallow = StandardShallowTransformer(vocab_size)
        opt_s = optim.AdamW(m_shallow.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for ep in range(25):
            m_shallow.train()
            for _ in range(20):
                x, y = train_data.get_batch(64)
                opt_s.zero_grad()
                crit(m_shallow(x), y).backward()
                opt_s.step()
        m_shallow.eval()
        with torch.no_grad():
            tx, ty = test_data.get_batch(400)
            acc_shallow = (m_shallow(tx).argmax(-1) == ty).float().mean().item() * 100
            
        # 2. Dual-Loop v2 (K=hop)
        set_seed(42)
        m_v2 = DualLoopV2_TransformerPrefix(vocab_size, k_steps=hop, num_thought_tokens=4)
        opt_v2 = optim.AdamW(m_v2.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(25):
            m_v2.train()
            for _ in range(20):
                x, y = train_data.get_batch(64)
                opt_v2.zero_grad()
                crit(m_v2(x), y).backward()
                opt_v2.step()
        m_v2.eval()
        with torch.no_grad():
            tx, ty = test_data.get_batch(400)
            acc_v2 = (m_v2(tx).argmax(-1) == ty).float().mean().item() * 100
            
        results_hop[hop] = (acc_shallow, acc_v2)
        print(f"Hop = {hop} | Shallow (L=1): {acc_shallow:5.1f}% | Dual-Loop v2 (K={hop}): {acc_v2:5.1f}%")

if __name__ == "__main__":
    run_hop_experiment()
