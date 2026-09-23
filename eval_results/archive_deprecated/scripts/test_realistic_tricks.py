import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# Dataset that provides intermediate hop targets (FOR TRAINING ONLY)
class MultiHopDatasetWithHops:
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
            
            chain = nodes[:self.hops + 1] # [chain[0], chain[1], chain[2], chain[3]]
            edges = []
            for i in range(len(chain) - 1):
                edges.append((chain[i], chain[i+1]))
            
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
            # chain[1..hops] are the intermediate and final hop targets
            hop_targets = chain[1:self.hops + 1] # length = hops
            
            seq.extend([self.QUERY, query_node, self.ARROW])
            samples.append((
                torch.tensor(seq, dtype=torch.long),
                torch.tensor(hop_targets, dtype=torch.long) # [hops]
            ))
        return samples

    def get_batch(self, batch_size=64):
        indices = random.sample(range(self.num_samples), batch_size)
        seqs = [self.data[i][0] for i in indices]
        targets = torch.stack([self.data[i][1] for i in indices])
        inputs = torch.stack(seqs)
        return inputs, targets # targets: [B, hops]

# ==========================================
# Dual-Loop v2.1 with Realistic Tricks:
# Trick 1: Latent Semantic Anchoring (Training-time probe)
# Trick 2: Query-Anchored Residual Stream (No prompt forgetting)
# Trick 3: Norm Constrained Latent Space
# ==========================================
class DualLoopV2_Enhanced(nn.Module):
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
        
        # Auxiliary Training-Time Probe (Zero-cost at test time)
        self.aux_probe = nn.Linear(d_model, vocab_size)
        
        self.readout = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, k_override=None, return_aux=False):
        B, S = x.shape
        h = self.embedding(x) + self.pos_emb[:, :S, :]
        ctx = self.shallow_encoder(h)
        
        k = self.k_steps if k_override is None else k_override
        H = self.thought_seeds.expand(B, -1, -1)
        H_initial = H.clone() # Trick 2: Anchor
        
        aux_logits = []
        for step in range(k):
            attn1, _ = self.latent_self_attn(H, H, H)
            H = self.norm1(H + attn1)
            attn2, _ = self.latent_cross_attn(H, ctx, ctx)
            H = self.norm2(H + attn2 + 0.2 * H_initial) # Residual anchor
            H = self.norm3(H + self.latent_mlp(H))
            
            if return_aux:
                # Probe token 0 of thoughts
                aux_logits.append(self.aux_probe(H[:, 0, :]))
            
        fused = torch.cat([H, ctx], dim=1)
        out = self.readout(fused)
        final_logits = self.head(out[:, -1, :])
        
        if return_aux:
            return final_logits, aux_logits
        return final_logits

def run_experiment():
    print("=" * 70)
    print("TESTING REALISTIC TRICKS FOR DUAL-LOOP COGNITIVE CONTROLLER")
    print("=" * 70)
    
    train_data = MultiHopDatasetWithHops(num_samples=2500, num_nodes=20, num_edges=8, hops=3)
    test_data = MultiHopDatasetWithHops(num_samples=500, num_nodes=20, num_edges=8, hops=3)
    vocab_size = train_data.vocab_size
    
    # 1. Baseline v2: Raw End-to-End (No tricks)
    set_seed(42)
    m_raw = DualLoopV2_Enhanced(vocab_size, k_steps=3)
    opt_raw = optim.AdamW(m_raw.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    print("\nTraining Dual-Loop v2 RAW (Tanpa Trick, End-to-End saja)...")
    for ep in range(1, 31):
        m_raw.train()
        for _ in range(25):
            x, y_all = train_data.get_batch(64)
            y_final = y_all[:, -1]
            opt_raw.zero_grad()
            logits = m_raw(x, return_aux=False)
            loss = crit(logits, y_final)
            loss.backward()
            opt_raw.step()
            
    m_raw.eval()
    with torch.no_grad():
        tx, ty_all = test_data.get_batch(500)
        ty_final = ty_all[:, -1]
        acc_raw = (m_raw(tx).argmax(-1) == ty_final).float().mean().item() * 100
    print(f"-> Test Acc Dual-Loop v2 RAW: {acc_raw:.1f}%")

    # 2. Enhanced v2: With Realistic Tricks (Training-time Latent Anchoring)
    set_seed(42)
    m_enhanced = DualLoopV2_Enhanced(vocab_size, k_steps=3)
    opt_enh = optim.AdamW(m_enhanced.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\nTraining Dual-Loop v2 ENHANCED (Dengan Semantic Anchoring + Residual Stream)...")
    for ep in range(1, 31):
        m_enhanced.train()
        for _ in range(25):
            x, y_all = train_data.get_batch(64)
            y_final = y_all[:, -1]
            opt_enh.zero_grad()
            logits, aux_list = m_enhanced(x, return_aux=True)
            
            # Final loss
            loss = crit(logits, y_final)
            # Auxiliary hop anchor loss (hanya saat training!)
            for step_idx, aux_l in enumerate(aux_list):
                if step_idx < y_all.shape[1]:
                    loss = loss + 0.4 * crit(aux_l, y_all[:, step_idx])
                    
            loss.backward()
            opt_enh.step()
            
    m_enhanced.eval()
    with torch.no_grad():
        # Uji coba inferensi 100% laten (tanpa token bantu dan tanpa aux!)
        acc_enh = (m_enhanced(tx, return_aux=False).argmax(-1) == ty_final).float().mean().item() * 100
    print(f"-> Test Acc Dual-Loop v2 ENHANCED (Zero-cost latent inference): {acc_enh:.1f}%")

    # 3. Test-time compute scaling on ENHANCED
    print("\nTest-Time Compute Scaling pada Model ENHANCED (Evaluasi Dinamis K):")
    with torch.no_grad():
        for k_val in [0, 1, 2, 3, 4]:
            acc_k = (m_enhanced(tx, k_override=k_val).argmax(-1) == ty_final).float().mean().item() * 100
            print(f"  Ponder Steps K = {k_val} -> Accuracy: {acc_k:.1f}%")

if __name__ == "__main__":
    run_experiment()
