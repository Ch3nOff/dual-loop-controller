import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from test_realistic_tricks import MultiHopDatasetWithHops, set_seed

class DualLoopV3_QueryConditioned(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=128, k_steps=3, num_thought_tokens=2):
        super().__init__()
        self.k_steps = k_steps
        self.num_thought_tokens = num_thought_tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.shallow_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Recurrent thought transitions
        self.latent_self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.latent_cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.latent_mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.thought_proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, k_override=None):
        B, S = x.shape
        h = self.embedding(x) + self.pos_emb[:, :S, :]
        ctx = self.shallow_encoder(h)
        
        k = self.k_steps if k_override is None else k_override
        
        # TRICK: Inisialisasi H BUKAN dari vektor acak konstan,
        # melainkan dari representasi token Query di ujung prompt!
        query_repr = ctx[:, -2, :].unsqueeze(1) # [B, 1, D]
        H = self.thought_proj(query_repr).repeat(1, self.num_thought_tokens, 1) # [B, L_thought, D]
        
        for step in range(k):
            # 1. Self-Attention antar thought tokens
            attn1, _ = self.latent_self_attn(H, H, H)
            H = self.norm1(H + attn1)
            # 2. Cross-Attention ke edges di context
            attn2, _ = self.latent_cross_attn(H, ctx, ctx)
            H = self.norm2(H + attn2)
            # 3. Transition MLP
            H = self.norm3(H + self.latent_mlp(H))
            
        # Prediksi langsung dari status laten pemikiran akhir
        logits = self.head(H[:, 0, :])
        return logits

def test_query_conditioned():
    set_seed(42)
    train_data = MultiHopDatasetWithHops(num_samples=3000, num_nodes=16, num_edges=6, hops=3)
    test_data = MultiHopDatasetWithHops(num_samples=500, num_nodes=16, num_edges=6, hops=3)
    vocab_size = train_data.vocab_size
    
    model = DualLoopV3_QueryConditioned(vocab_size, k_steps=3, num_thought_tokens=2)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    print("Training Query-Conditioned Dual-Loop (30 epochs)...")
    for ep in range(1, 31):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for _ in range(30):
            x, y_all = train_data.get_batch(64)
            y_final = y_all[:, -1]
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y_final)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += (logits.argmax(-1) == y_final).sum().item()
            total += y_final.size(0)
        if ep % 10 == 0:
            print(f"Epoch {ep:2d} | Train Acc: {correct/total*100:.1f}% | Loss: {total_loss/30:.4f}")
            
    model.eval()
    with torch.no_grad():
        tx, ty_all = test_data.get_batch(500)
        ty_final = ty_all[:, -1]
        acc_test = (model(tx).argmax(-1) == ty_final).float().mean().item() * 100
        print(f"\n-> FINAL TEST ACCURACY (3-Hop): {acc_test:.1f}% (Random baseline: {100/16:.1f}%)")
        
        print("\nTest-Time Compute Scaling dengan Query Conditioning:")
        for k_val in [0, 1, 2, 3, 4, 5]:
            acc_k = (model(tx, k_override=k_val).argmax(-1) == ty_final).float().mean().item() * 100
            print(f"  Ponder Step K = {k_val} -> Accuracy: {acc_k:.1f}%")

if __name__ == "__main__":
    test_query_conditioned()
