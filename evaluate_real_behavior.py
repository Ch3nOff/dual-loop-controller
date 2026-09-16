import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dual_loop.benchmarks import MultiHopGraphDataset
from dual_loop.decoder import DualLoopTransformer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def decode_token(tok_id, num_nodes):
    ARROW = num_nodes
    SEP = num_nodes + 1
    QUERY = num_nodes + 2
    if tok_id < num_nodes:
        return f"Node_{tok_id}"
    elif tok_id == ARROW:
        return "->"
    elif tok_id == SEP:
        return ","
    elif tok_id == QUERY:
        return "[QUERY]"
    return f"Tok_{tok_id}"

def decode_sequence(seq, num_nodes):
    return " ".join([decode_token(t.item(), num_nodes) for t in seq])

def run_training_and_deep_behavior_audit():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("REAL MODEL TRAINING & BEHAVIORAL AUDIT")
    print("Dual-Loop Cognitive Controller v2.0 (Query-Conditioned Latent Recurrence)")
    print(f"Hardware: {device}")
    print("=" * 80)

    # 1. Setup Benchmark Data (3-Hop Reasoning on 16 Nodes)
    num_nodes = 16
    hops = 3
    num_edges = 6
    train_data = MultiHopGraphDataset(num_samples=3500, num_nodes=num_nodes, num_edges=num_edges, hops=hops, split="train", seed=42)
    test_data = MultiHopGraphDataset(num_samples=500, num_nodes=num_nodes, num_edges=num_edges, hops=hops, split="test", seed=42)
    vocab_size = train_data.vocab_size

    # 2. Instantiate Small, Efficient Dual-Loop Model
    d_model = 64
    k_steps = 3
    num_thoughts = 4
    model = DualLoopTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        d_ff=128,
        num_thought_tokens=num_thoughts,
        num_cwm_slots=12,
        max_ponder_steps=k_steps,
        capacity_factor=0.5,
        entropy_threshold=0.6
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model Config] Hidden Dim: {d_model} | Params: {total_params:,} | Ponder Steps (K): {k_steps}")
    print(f"[Dataset Config] Nodes: {num_nodes} | Relational Hops: {hops} | Random Guess Chance: {100/num_nodes:.1f}%\n")

    # 3. Training Loop with Semantic Anchoring
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(">>> FASE 1: PELATIHAN MODEL (35 Epochs)...")
    start_time = time.time()
    for epoch in range(1, 36):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(30):
            x, y_hops = train_data.get_batch(batch_size=64)
            x, y_hops = x.to(device), y_hops.to(device)
            y_final = y_hops[:, -1]

            optimizer.zero_grad()
            logits, info = model(x, return_aux=True)

            # Task Loss on final target
            loss = criterion(logits, y_final)

            # Training-time Latent Anchoring on intermediate thoughts
            aux_logits = info["aux_logits"]
            for step_idx, aux_l in enumerate(aux_logits):
                if step_idx < y_hops.shape[1]:
                    loss = loss + 0.35 * criterion(aux_l, y_hops[:, step_idx])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == y_final).sum().item()
            total += y_final.size(0)

        if epoch % 5 == 0 or epoch == 35:
            model.eval()
            with torch.no_grad():
                tx, ty_hops = test_data.get_batch(batch_size=500)
                tx, ty_final = tx.to(device), ty_hops[:, -1].to(device)
                test_logits, _ = model(tx)
                test_acc = (test_logits.argmax(dim=-1) == ty_final).float().mean().item() * 100.0
            print(f"  Epoch {epoch:2d}/35 | Train Acc: {correct/total*100:5.1f}% | Loss: {total_loss/30:.4f} | Test Acc: {test_acc:5.1f}%")

    train_duration = time.time() - start_time
    print(f"\n[OK] Pelatihan selesai dalam {train_duration:.1f} detik.")
    
    # Save checkpoint
    torch.save(model.state_dict(), "checkpoint_trained_dualloop.pt")
    print("Checkpoint berhasil disimpan ke 'checkpoint_trained_dualloop.pt'.")

    # =========================================================================
    # 4. DEEP BEHAVIORAL AUDIT (Pengujian Kemampuan Ril)
    # =========================================================================
    model.eval()

    # TEST A: Test-Time Compute Scaling Curve across all 500 test samples
    print("\n" + "=" * 80)
    print("AUDIT 1: MAKRO PENSKALAAN WAKTU KOMPUTASI (Test-Time Compute Scaling)")
    print("Memverifikasi apakah pertambahan langkah laten K meningkatkan akurasi secara konsisten")
    print("=" * 80)
    with torch.no_grad():
        tx, ty_hops = test_data.get_batch(batch_size=500)
        tx, ty_final = tx.to(device), ty_hops[:, -1].to(device)
        scaling_results = {}
        for k_eval in [0, 1, 2, 3, 4, 5]:
            logits_k, info_k = model(tx, k_steps=k_eval)
            acc_k = (logits_k.argmax(dim=-1) == ty_final).float().mean().item() * 100.0
            probs = F.softmax(logits_k, dim=-1)
            entropy_k = -torch.sum(probs * F.log_softmax(logits_k, dim=-1), dim=-1).mean().item()
            scaling_results[k_eval] = (acc_k, entropy_k)
            print(f"  Ponder Step K = {k_eval:1d} | Test Accuracy: {acc_k:5.1f}% | Rata-rata Entropi: {entropy_k:.3f} nats")

    # TEST B: Micro Step-by-Step Case Trace (Concrete Graph Traversal Verification)
    print("\n" + "=" * 80)
    print("AUDIT 2: MIKRO PELACAKAN KASUS NYATA (Step-by-Step Latent Traversal)")
    print("Memeriksa apakah prediksi model bergerak dari Hop 1 -> Hop 2 -> Target di setiap langkah K")
    print("=" * 80)

    # Pick 3 specific test samples
    test_indices = [3, 7, 12]
    with torch.no_grad():
        for case_num, idx in enumerate(test_indices, 1):
            raw_seq, hop_targets = test_data.data[idx]
            input_tensor = raw_seq.unsqueeze(0).to(device)
            ground_truth_hops = [decode_token(h.item(), num_nodes) for h in hop_targets]
            query_node_id = raw_seq[-2].item()
            
            print(f"\n--- [KASUS NYATA #{case_num}] ---")
            print(f"Sequence Input : {decode_sequence(raw_seq, num_nodes)}")
            print(f"Start Query    : {decode_token(query_node_id, num_nodes)}")
            print(f"Ground Truth   : Hop 1={ground_truth_hops[0]} -> Hop 2={ground_truth_hops[1]} -> Final={ground_truth_hops[2]}")
            print("Perkembangan Penalaran Laten di setiap langkah K:")

            for k_eval in [0, 1, 2, 3]:
                logits_step, _ = model(input_tensor, k_steps=k_eval)
                probs = F.softmax(logits_step, dim=-1)[0]
                pred_id = probs.argmax().item()
                confidence = probs[pred_id].item() * 100.0
                ent = -torch.sum(probs * torch.log(probs + 1e-9)).item()

                # Get top 2 candidates
                top2_probs, top2_indices = torch.topk(probs, 2)
                cand_str = f"Top-1: {decode_token(top2_indices[0].item(), num_nodes)} ({top2_probs[0]*100:.1f}%), Top-2: {decode_token(top2_indices[1].item(), num_nodes)} ({top2_probs[1]*100:.1f}%)"

                status = "BENAR" if pred_id == hop_targets[-1].item() else "BELUM/SALAH"
                print(f"  [K={k_eval}] Prediksi Akhir: {decode_token(pred_id, num_nodes):<8} | Keyakinan: {confidence:5.1f}% | Entropi: {ent:.3f} | {cand_str} | Status: {status}")

    # TEST C: Stress-Test / Out-of-Distribution (OOD with Distractors)
    print("\n" + "=" * 80)
    print("AUDIT 3: UJI KETAHANAN GANGGUAN / STRESS-TEST (Distractor Robustness)")
    print("Menguji model pada graf dengan gangguan edge palsu (distractors) 2x lebih banyak (12 edges)")
    print("=" * 80)
    ood_dataset = MultiHopGraphDataset(num_samples=300, num_nodes=num_nodes, num_edges=12, hops=hops, split="test", seed=42)
    with torch.no_grad():
        ox, oy_hops = ood_dataset.get_batch(batch_size=300)
        ox, oy_final = ox.to(device), oy_hops[:, -1].to(device)
        for k_eval in [0, 1, 2, 3]:
            o_logits, _ = model(ox, k_steps=k_eval)
            o_acc = (o_logits.argmax(dim=-1) == oy_final).float().mean().item() * 100.0
            print(f"  OOD (12 Edges) | Ponder Step K = {k_eval} -> Accuracy: {o_acc:5.1f}%")

    # TEST D: Dynamic Halting in Action
    print("\n" + "=" * 80)
    print("AUDIT 4: PENGHENTIAN DINAMIS OTOMATIS (Dynamic Halting / Entropy Thresholding)")
    print("Melihat apakah model berhenti otomatis saat sudah yakin (menghemat langkah inferensi)")
    print("=" * 80)
    with torch.no_grad():
        tx_sub, ty_sub = tx[:100], ty_final[:100]
        logits_dyn, info_dyn = model(tx_sub, dynamic_halting=True, return_aux=True)
        acc_dyn = (logits_dyn.argmax(dim=-1) == ty_sub).float().mean().item() * 100.0
        effective_k = info_dyn["effective_k"]
        print(f"  Akurasi dengan Dynamic Halting: {acc_dyn:.1f}%")
        print(f"  Rata-rata Langkah K yang Digunakan: {effective_k:.2f} langkah (dari batas maksimal {k_steps})")

    print("\n" + "=" * 80)
    print("KESIMPULAN AUDIT PERILAKU SELESAI")
    print("=" * 80)

if __name__ == "__main__":
    run_training_and_deep_behavior_audit()
