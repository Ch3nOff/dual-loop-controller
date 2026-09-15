import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

from dual_loop import DualLoopTransformer
from dual_loop.benchmarks import MultiHopGraphDataset

def main():
    parser = argparse.ArgumentParser(description="Train Dual-Loop Cognitive Controller v2.0")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hops", type=int, default=3, help="Number of relational hops in benchmark")
    parser.add_argument("--k_steps", type=int, default=3, help="Number of latent deliberation steps")
    parser.add_argument("--num_thoughts", type=int, default=4, help="Number of thought tokens")
    parser.add_argument("--d_model", type=int, default=64, help="Hidden dimension")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"Dual-Loop Cognitive Controller Training Pipeline")
    print(f"Device: {device} | Hops: {args.hops} | Ponder Steps (K): {args.k_steps}")
    print("=" * 70)

    # Initialize benchmark datasets
    train_dataset = MultiHopGraphDataset(num_samples=2500, num_nodes=20, num_edges=8, hops=args.hops)
    test_dataset = MultiHopGraphDataset(num_samples=500, num_nodes=20, num_edges=8, hops=args.hops)
    vocab_size = train_dataset.vocab_size

    # Initialize Dual-Loop Model
    model = DualLoopTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=4,
        d_ff=args.d_model * 2,
        num_thought_tokens=args.num_thoughts,
        num_cwm_slots=16,
        max_ponder_steps=args.k_steps,
        capacity_factor=0.5
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}\n")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for _ in range(25):
            inputs, hop_targets = train_dataset.get_batch(args.batch_size)
            inputs = inputs.to(device)
            final_target = hop_targets[:, -1].to(device)

            optimizer.zero_grad()
            logits, info = model(inputs, return_aux=True)

            # Main task loss on final target
            loss = criterion(logits, final_target)

            # Auxiliary training supervision on intermediate thoughts
            aux_logits = info["aux_logits"]
            for step_idx, aux_l in enumerate(aux_logits):
                if step_idx < hop_targets.shape[1]:
                    sub_target = hop_targets[:, step_idx].to(device)
                    loss = loss + 0.3 * criterion(aux_l, sub_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == final_target).sum().item()
            total += final_target.size(0)

        train_acc = correct / total * 100.0
        avg_loss = epoch_loss / 25

        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                test_in, test_tgt_all = test_dataset.get_batch(500)
                test_in = test_in.to(device)
                test_tgt = test_tgt_all[:, -1].to(device)
                test_logits, _ = model(test_in)
                test_acc = (test_logits.argmax(dim=-1) == test_tgt).float().mean().item() * 100.0

            print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:5.1f}% | Test Acc: {test_acc:5.1f}%")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s.")

    # Test-time scaling verification
    print("\n" + "-" * 70)
    print("Zero-Shot Test-Time Compute Scaling Evaluation:")
    print("-" * 70)
    model.eval()
    with torch.no_grad():
        test_in, test_tgt_all = test_dataset.get_batch(500)
        test_in = test_in.to(device)
        test_tgt = test_tgt_all[:, -1].to(device)
        for k in [0, 1, 2, 3, 5]:
            logits_k, _ = model(test_in, k_steps=k)
            acc_k = (logits_k.argmax(dim=-1) == test_tgt).float().mean().item() * 100.0
            print(f"  Ponder Steps K = {k} -> Test Accuracy: {acc_k:.1f}%")

if __name__ == "__main__":
    main()
