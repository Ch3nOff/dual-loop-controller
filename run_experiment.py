import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set seeds for deterministic reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# ==========================================
# 1. Dataset: Multi-Hop Pointer Reasoning
# ==========================================
# Graph: Directed permutation / paths over V nodes.
# Context: Shuffled edges (u -> v).
# Query: Start node.
# Target: Node reached after H hops.
class MultiHopGraphDataset:
    def __init__(self, num_samples=2500, num_nodes=20, num_edges=8, hops=3):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.hops = hops
        
        self.ARROW = num_nodes
        self.SEP = num_nodes + 1
        self.QUERY = num_nodes + 2
        self.vocab_size = num_nodes + 3
        
        self.data = self._generate_data()

    def _generate_data(self):
        samples = []
        for _ in range(self.num_samples):
            nodes = list(range(self.num_nodes))
            random.shuffle(nodes)
            
            chain = nodes[:self.hops + 1]
            edges = []
            for i in range(len(chain) - 1):
                edges.append((chain[i], chain[i+1]))
            
            # Add random distractor edges until reaching exact num_edges
            while len(edges) < self.num_edges:
                u = random.choice(nodes)
                v = random.choice([n for n in nodes if n != u])
                if (u, v) not in edges:
                    edges.append((u, v))
            
            random.shuffle(edges)
            
            seq = []
            for u, v in edges:
                seq.extend([u, self.ARROW, v, self.SEP])
            
            query_node = chain[0]
            target_node = chain[self.hops]
            
            seq.extend([self.QUERY, query_node, self.ARROW])
            samples.append((torch.tensor(seq, dtype=torch.long), torch.tensor(target_node, dtype=torch.long)))
        return samples

    def get_batch(self, batch_size=64):
        indices = random.sample(range(self.num_samples), batch_size)
        seqs = [self.data[i][0] for i in indices]
        targets = torch.stack([self.data[i][1] for i in indices])
        inputs = torch.stack(seqs)
        return inputs, targets

# ==========================================
# 2. Architectures Under Honest Evaluation
# ==========================================

# Baseline A: Standard Shallow Transformer (L=1)
class StandardShallowTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, S = x.shape
        h = self.embedding(x) + self.pos_emb[:, :S, :]
        h = self.encoder(h)
        logits = self.head(h[:, -1, :])
        return logits

# Baseline B: Deep Parameter-Matched Transformer (L=4)
class DeepTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=128, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, S = x.shape
        h = self.embedding(x) + self.pos_emb[:, :S, :]
        h = self.encoder(h)
        logits = self.head(h[:, -1, :])
        return logits

# Flawed Architecture v1: Naive GRU Controller with 1D Vector Bottleneck
class DualLoopV1_GRU(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=128, k_steps=3):
        super().__init__()
        self.k_steps = k_steps
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.gru = nn.GRUCell(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.gate = nn.Linear(d_model, 1)
        
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, S = x.shape
        h = self.embedding(x) + self.pos_emb[:, :S, :]
        ctx = self.encoder(h)
        
        # 1D Vector state initialization
        h_k = ctx.mean(dim=1)
        
        for _ in range(self.k_steps):
            g = torch.sigmoid(self.gate(h_k))
            q = h_k.unsqueeze(1)
            attn_out, _ = self.cross_attn(q, ctx, ctx)
            thought_input = g * attn_out.squeeze(1) + (1.0 - g) * self.mlp(h_k)
            h_k = self.gru(thought_input, h_k)
            
        logits = self.head(h_k)
        return logits

# Proposed Architecture v2: Recurrent Latent Transformer with Prefix Thoughts
class DualLoopV2_TransformerPrefix(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=128, k_steps=3, num_thought_tokens=4):
        super().__init__()
        self.k_steps = k_steps
        self.num_thought_tokens = num_thought_tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.shallow_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.thought_seeds = nn.Parameter(torch.randn(1, num_thought_tokens, d_model) * 0.02)
        self.latent_self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.latent_cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.latent_mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.readout = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, k_override=None):
        B, S = x.shape
        h = self.embedding(x) + self.pos_emb[:, :S, :]
        ctx = self.shallow_encoder(h)
        
        k = self.k_steps if k_override is None else k_override
        H = self.thought_seeds.expand(B, -1, -1)
        
        for _ in range(k):
            attn1, _ = self.latent_self_attn(H, H, H)
            H = self.norm1(H + attn1)
            attn2, _ = self.latent_cross_attn(H, ctx, ctx)
            H = self.norm2(H + attn2)
            H = self.norm3(H + self.latent_mlp(H))
            
        fused = torch.cat([H, ctx], dim=1)
        out = self.readout(fused)
        logits = self.head(out[:, -1, :])
        return logits

# ==========================================
# 3. Training & Evaluation Engine
# ==========================================
def train_and_eval(model, train_data, test_data, epochs=30, lr=1e-3, model_name=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n>>> [{model_name}] Total Parameters: {total_params:,}")
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        grad_norms = []
        
        for _ in range(25):
            inputs, targets = train_data.get_batch(batch_size=64)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2).item()
            grad_norms.append(total_norm)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        train_acc = correct / total * 100.0
        avg_loss = total_loss / 25
        avg_grad_norm = np.mean(grad_norms)
        
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_inputs, test_targets = test_data.get_batch(batch_size=200)
                test_logits = model(test_inputs)
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == test_targets).sum().item() / test_targets.size(0) * 100.0
            print(f"  Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:5.1f}% | Test Acc: {test_acc:5.1f}% | GradNorm: {avg_grad_norm:.3f}")
            
    elapsed = time.time() - start_time
    
    model.eval()
    with torch.no_grad():
        test_inputs, test_targets = test_data.get_batch(batch_size=500)
        test_logits = model(test_inputs)
        test_preds = test_logits.argmax(dim=-1)
        final_acc = (test_preds == test_targets).sum().item() / test_targets.size(0) * 100.0
        
    return {
        "final_acc": final_acc,
        "elapsed_s": elapsed,
        "params": total_params,
        "final_grad_norm": avg_grad_norm
    }

# ==========================================
# 4. Main Experiment Runner
# ==========================================
if __name__ == "__main__":
    print("=" * 75)
    print("HONEST EMPIRICAL TEST: DUAL-LOOP COGNITIVE ARCHITECTURE")
    print("Task: Multi-Hop Graph Traversal (Hops = 3)")
    print("=" * 75)
    
    train_dataset = MultiHopGraphDataset(num_samples=2500, num_nodes=20, num_edges=8, hops=3)
    test_dataset = MultiHopGraphDataset(num_samples=500, num_nodes=20, num_edges=8, hops=3)
    vocab_size = train_dataset.vocab_size
    
    results = {}
    
    # 1. Baseline: Shallow Transformer (L=1)
    set_seed(42)
    m_shallow = StandardShallowTransformer(vocab_size)
    results["Standard Shallow (L=1)"] = train_and_eval(m_shallow, train_dataset, test_dataset, epochs=30, model_name="Standard Shallow Transformer (L=1, No Ponder)")
    
    # 2. Dual-Loop v1 (GRU Controller + 1D Bottleneck)
    set_seed(42)
    m_v1 = DualLoopV1_GRU(vocab_size, k_steps=3)
    results["Dual-Loop v1 (GRU Bottleneck)"] = train_and_eval(m_v1, train_dataset, test_dataset, epochs=30, model_name="Dual-Loop v1 (GRU + 1D State Vector)")
    
    # 3. Dual-Loop v2 (Weight-Tied Transformer + Prefix Thoughts)
    set_seed(42)
    m_v2 = DualLoopV2_TransformerPrefix(vocab_size, k_steps=3, num_thought_tokens=4)
    results["Dual-Loop v2 (Prefix Thoughts K=3)"] = train_and_eval(m_v2, train_dataset, test_dataset, epochs=30, model_name="Dual-Loop v2 (Recurrent Latent + Prefix Thoughts K=3)")
    
    # 4. Deep Transformer Baseline (L=4)
    set_seed(42)
    m_deep = DeepTransformer(vocab_size, num_layers=4)
    results["Deep Transformer (L=4)"] = train_and_eval(m_deep, train_dataset, test_dataset, epochs=30, model_name="Deep Transformer (L=4 Stacked Feedforward)")
    
    # 5. Pondering Step Zero-Shot Generalization Test on v2:
    print("\n" + "=" * 75)
    print("TEST-TIME COMPUTE SCALING TEST (Dual-Loop v2 with Dynamic K)")
    print("=" * 75)
    m_v2.eval()
    test_inputs, test_targets = test_dataset.get_batch(batch_size=500)
    with torch.no_grad():
        for k_eval in [0, 1, 2, 3, 5]:
            logits_k = m_v2(test_inputs, k_override=k_eval)
            acc_k = (logits_k.argmax(dim=-1) == test_targets).sum().item() / test_targets.size(0) * 100.0
            print(f"  Test-Time Ponder Step K = {k_eval} -> Accuracy: {acc_k:.1f}%")

    print("\n" + "=" * 75)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Model Architecture':<38} | {'Params':<8} | {'Accuracy':<9} | {'GradNorm':<9} | {'Time (s)':<8}")
    print("-" * 75)
    for name, r in results.items():
        print(f"{name:<38} | {r['params']:<8} | {r['final_acc']:5.1f}%   | {r['final_grad_norm']:<9.3f} | {r['elapsed_s']:<8.1f}")
