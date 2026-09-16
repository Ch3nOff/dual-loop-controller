import os
import random
import torch
import torch.nn.functional as F
import numpy as np

from dual_loop import DualLoopTransformer, load_trained_checkpoint
from dual_loop.benchmarks import MultiHopGraphDataset

def run_halting_audit():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = 16

    # Instantiate model
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

    try:
        _, loaded_path = load_trained_checkpoint(model)
        print(f"Loaded checkpoint from '{loaded_path}'.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    model.eval()

    # Load 500 test samples deterministically
    dataset = MultiHopGraphDataset(num_samples=500, num_nodes=num_nodes, num_edges=6, hops=3, seed=42)
    x_test, y_test_all = dataset.get_batch(500, shuffle=False)
    x_test = x_test.to(device)
    y_test = y_test_all[:, -1].to(device)

    print("=" * 85)
    print("AUDIT: DISTRIBUSI ENTROPI AKTUAL DI SETIAP LANGKAH PONDER (K = 1, 2, 3)")
    print("=" * 85)

    # 1. First, inspect step-by-step entropy distribution without early stopping
    with torch.no_grad():
        _, info_full = model(x_test, k_steps=3, return_aux=True)
        # aux_logits is a list of [B, VocabSize] for steps 1, 2, 3
        entropies_per_step = []
        for step_idx, logits_step in enumerate(info_full["aux_logits"]):
            probs = F.softmax(logits_step, dim=-1)
            ent = -torch.sum(probs * F.log_softmax(logits_step, dim=-1), dim=-1) # [B]
            entropies_per_step.append(ent.cpu().numpy())
            print(f"Langkah K={step_idx+1}:")
            print(f"  Min   : {ent.min().item():.3f} nats")
            print(f"  Q25   : {np.percentile(ent.cpu().numpy(), 25):.3f} nats")
            print(f"  Median: {np.median(ent.cpu().numpy()):.3f} nats")
            print(f"  Mean  : {ent.mean().item():.3f} nats (+/- {ent.std().item():.3f})")
            print(f"  Q75   : {np.percentile(ent.cpu().numpy(), 75):.3f} nats")
            print(f"  Max   : {ent.max().item():.3f} nats")

    print("\n" + "=" * 85)
    print("AUDIT PARETO: SWEEP AMBANG BATAS ENTROPI TERHADAP TRADE-OFF AKURASI VS LANGKAH")
    print("Menguji per-sample halting (setiap sequence berhenti sendiri saat yakin)")
    print("=" * 85)

    # We test per-sample dynamic halting: evaluate each sample individually or track per-sample stop
    # Sweep thresholds from 0.4 to 2.2 nats
    thresholds = [0.5, 0.8, 1.0, 1.1, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50, 1.80]
    
    # Precompute per-sample logits and entropies at step 1, 2, 3
    with torch.no_grad():
        logits_k1, _ = model(x_test, k_steps=1)
        logits_k2, _ = model(x_test, k_steps=2)
        logits_k3, _ = model(x_test, k_steps=3)
        
        ent_k1 = -torch.sum(F.softmax(logits_k1, -1) * F.log_softmax(logits_k1, -1), -1)
        ent_k2 = -torch.sum(F.softmax(logits_k2, -1) * F.log_softmax(logits_k2, -1), -1)
        ent_k3 = -torch.sum(F.softmax(logits_k3, -1) * F.log_softmax(logits_k3, -1), -1)

    print(f"{'Threshold (nats)':<16} | {'Akurasi (%)':<12} | {'Rata2 K':<10} | {'Stop @ K=1':<12} | {'Stop @ K=2':<12} | {'Stop @ K=3':<12}")
    print("-" * 85)

    for thresh in thresholds:
        # Determine for each sample when it halts
        # If ent_k1 <= thresh: halts at K=1, prediction is logits_k1
        # Else if ent_k2 <= thresh: halts at K=2, prediction is logits_k2
        # Else: halts at K=3, prediction is logits_k3
        halt_at_1 = (ent_k1 <= thresh)
        halt_at_2 = (~halt_at_1) & (ent_k2 <= thresh)
        halt_at_3 = (~halt_at_1) & (~halt_at_2)

        # Assemble final predictions
        final_preds = torch.zeros_like(y_test)
        final_preds[halt_at_1] = logits_k1[halt_at_1].argmax(dim=-1)
        final_preds[halt_at_2] = logits_k2[halt_at_2].argmax(dim=-1)
        final_preds[halt_at_3] = logits_k3[halt_at_3].argmax(dim=-1)

        acc = (final_preds == y_test).float().mean().item() * 100.0
        
        # Calculate steps
        steps = torch.ones_like(y_test, dtype=torch.float)
        steps[halt_at_2] = 2.0
        steps[halt_at_3] = 3.0
        avg_steps = steps.mean().item()

        pct_1 = halt_at_1.float().mean().item() * 100.0
        pct_2 = halt_at_2.float().mean().item() * 100.0
        pct_3 = halt_at_3.float().mean().item() * 100.0

        print(f"{thresh:<16.2f} | {acc:<12.1f} | {avg_steps:<10.2f} | {pct_1:<12.1f}% | {pct_2:<12.1f}% | {pct_3:<12.1f}%")

    print("\n" + "=" * 85)
    print("AUDIT PERCENTILE-BASED CALIBRATION (Tinjauan terhadap klaim percentile=50)")
    print("=" * 85)
    # What does percentile=50 on step 1 do vs step 3?
    p_values = [10, 25, 40, 50, 60, 75, 90]
    for p in p_values:
        calib_thresh = np.percentile(ent_k1.cpu().numpy(), p)
        h1 = (ent_k1 <= calib_thresh)
        h2 = (~h1) & (ent_k2 <= calib_thresh)
        h3 = (~h1) & (~h2)
        final_p = torch.zeros_like(y_test)
        final_p[h1] = logits_k1[h1].argmax(dim=-1)
        final_p[h2] = logits_k2[h2].argmax(dim=-1)
        final_p[h3] = logits_k3[h3].argmax(dim=-1)
        acc_p = (final_p == y_test).float().mean().item() * 100.0
        avg_s = (1.0 * h1.float() + 2.0 * h2.float() + 3.0 * h3.float()).mean().item()
        print(f"Percentile={p:2d}% -> Threshold: {calib_thresh:.3f} nats | Akurasi: {acc_p:5.1f}% | Rata-rata Langkah: {avg_s:.2f} | Halts: K1={h1.float().mean()*100:4.1f}%, K2={h2.float().mean()*100:4.1f}%, K3={h3.float().mean()*100:4.1f}%")

if __name__ == "__main__":
    run_halting_audit()
