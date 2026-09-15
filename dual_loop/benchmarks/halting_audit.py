"""
Pareto Analysis & Dynamic Halting Audit for Dual-Loop Cognitive Controller.
=============================================================================
Audits the empirical relationship between predictive entropy threshold,
accuracy, and computational steps taken.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from dual_loop import DualLoopTransformer
from dual_loop.benchmarks import MultiHopGraphDataset

def audit_halting_pareto():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = 16
    checkpoint_path = "checkpoint_trained_dualloop.pt"

    model = DualLoopTransformer(
        vocab_size=num_nodes + 3,
        d_model=64,
        n_heads=4,
        d_ff=128,
        num_thought_tokens=4,
        num_cwm_slots=12,
        max_ponder_steps=3,
        capacity_factor=0.5
    ).to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError("Checkpoint not found.")

    model.eval()
    dataset = MultiHopGraphDataset(num_samples=500, num_nodes=num_nodes, num_edges=6, hops=3)
    x_test, y_test_all = dataset.get_batch(500)
    x_test, y_test = x_test.to(device), y_test_all[:, -1].to(device)

    # 1. Measure raw step-by-step entropy distribution
    with torch.no_grad():
        logits_k1, _ = model(x_test, k_steps=1)
        logits_k2, _ = model(x_test, k_steps=2)
        logits_k3, _ = model(x_test, k_steps=3)

        ent_k1 = -torch.sum(F.softmax(logits_k1, -1) * F.log_softmax(logits_k1, -1), -1)
        ent_k2 = -torch.sum(F.softmax(logits_k2, -1) * F.log_softmax(logits_k2, -1), -1)
        ent_k3 = -torch.sum(F.softmax(logits_k3, -1) * F.log_softmax(logits_k3, -1), -1)

    # Pareto sweep across thresholds
    thresholds = [0.50, 0.80, 1.00, 1.15, 1.25, 1.30, 1.40, 1.50, 1.80]
    pareto_data = []

    for thresh in thresholds:
        h1 = (ent_k1 <= thresh)
        h2 = (~h1) & (ent_k2 <= thresh)
        h3 = (~h1) & (~h2)

        final_preds = torch.zeros_like(y_test)
        final_preds[h1] = logits_k1[h1].argmax(dim=-1)
        final_preds[h2] = logits_k2[h2].argmax(dim=-1)
        final_preds[h3] = logits_k3[h3].argmax(dim=-1)

        acc = (final_preds == y_test).float().mean().item() * 100.0
        steps = (1.0 * h1.float() + 2.0 * h2.float() + 3.0 * h3.float()).mean().item()

        pareto_data.append({
            "threshold": thresh,
            "accuracy": acc,
            "avg_steps": steps,
            "pct_k1": h1.float().mean().item() * 100.0,
            "pct_k2": h2.float().mean().item() * 100.0,
            "pct_k3": h3.float().mean().item() * 100.0
        })

    return {
        "ent_stats": {
            "k1": {"mean": ent_k1.mean().item(), "median": torch.median(ent_k1).item(), "std": ent_k1.std().item()},
            "k2": {"mean": ent_k2.mean().item(), "median": torch.median(ent_k2).item(), "std": ent_k2.std().item()},
            "k3": {"mean": ent_k3.mean().item(), "median": torch.median(ent_k3).item(), "std": ent_k3.std().item()}
        },
        "pareto": pareto_data
    }

if __name__ == "__main__":
    res = audit_halting_pareto()
    print("Step-by-Step Predictive Entropy Stats:")
    for k, s in res["ent_stats"].items():
        print(f"  {k}: Mean={s['mean']:.3f}, Median={s['median']:.3f}, Std={s['std']:.3f}")
    print("\nPareto Frontier (Threshold vs Accuracy vs Compute):")
    print(f"{'Threshold':<10} | {'Accuracy':<10} | {'Avg Steps':<10} | {'K=1 %':<8} | {'K=2 %':<8} | {'K=3 %':<8}")
    print("-" * 62)
    for p in res["pareto"]:
        print(f"{p['threshold']:<10.2f} | {p['accuracy']:<10.1f} | {p['avg_steps']:<10.2f} | {p['pct_k1']:<8.1f} | {p['pct_k2']:<8.1f} | {p['pct_k3']:<8.1f}")
