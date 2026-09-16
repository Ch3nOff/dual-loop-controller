import random
import numpy as np
import torch
from dual_loop import DualLoopTransformer, load_trained_checkpoint
from dual_loop.benchmarks import MultiHopGraphDataset

def test_inference_halting():
    # 1. Deterministic Seeding for 100% Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model = DualLoopTransformer(
        vocab_size=19,
        d_model=64,
        num_cwm_slots=12,
        max_ponder_steps=3,
        entropy_threshold=1.25
    )

    # 2. Resilient Checkpoint Loading (checks local & bundled package checkpoint)
    try:
        _, loaded_path = load_trained_checkpoint(model)
        print(f"[Checkpoint] Successfully loaded weights from: {loaded_path}")
    except FileNotFoundError as e:
        print("=" * 80)
        print("[!] ERROR: Checkpoint file not found.")
        print(str(e))
        print("=" * 80)
        return

    model.eval()

    # 3. Deterministic Benchmark Dataset
    dataset = MultiHopGraphDataset(num_samples=500, num_nodes=16, num_edges=6, hops=3, seed=42)
    x, y_all = dataset.get_batch(500, shuffle=False)
    y = y_all[:, -1]

    # 4. Calibrate Halting using Automated Pareto Grid-Search
    print("\n--- [CALIBRATION] Automated Pareto Grid-Search across Percentiles ---")
    calibrated_thresh = model.calibrate_halting(x, target_labels=y, verbose=True)
    print(f"Optimal Halting Threshold: {calibrated_thresh:.3f} nats\n")

    # 5. Run Forward Pass with Per-Sample Dynamic Halting
    logits, info = model(x, dynamic_halting=True)
    acc = (logits.argmax(-1) == y).float().mean().item() * 100.0
    st = info['steps_taken']

    print("=" * 60)
    print("EMPIRICAL DYNAMIC HALTING INFERENCE REPORT")
    print("=" * 60)
    print(f"Accuracy with dynamic_halting=True : {acc:.1f}%")
    print(f"Effective K (Average Ponder Steps) : {info['effective_k']:.2f} steps")
    print(f"Ponder Distribution across Batch   : K=1: {(st==1).float().mean()*100:.1f}%, K=2: {(st==2).float().mean()*100:.1f}%, K=3: {(st==3).float().mean()*100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    test_inference_halting()
