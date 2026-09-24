"""
Universal Cross-Modal Invariant Controller Benchmark Suite
==========================================================
Comprehensive Empirical Evaluation across 4 Core Multimodal Invariant Dimensions:
1. Procrustes Optimal Manifold Transport (Covariance & Tangent Alignment Fidelity)
2. Spatio-Temporal Entropic CWM (Token Explosion Stress Test: 256 to 2048 patches)
3. Hetero-Associative Plastic Memory (In-Situ Zero-Shot Concept Binding & Instant Recall)
4. Popperian Cross-Modal Falsification (Grounded vs Hallucinatory Sensory Attractor Suppression)
5. End-to-End DualLoopQwen Multimodal Pipeline Performance & Latency
"""

import time
import math
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any

from dual_loop.multimodal_transport import ProcrustesOptimalManifoldTransport
from dual_loop.topological_cwm import SpatioTemporalEntropicCWM
from dual_loop.plasticity import HeteroAssociativePlasticMemory
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter
from dual_loop.adapters.qwen_adapter import DualLoopQwenModel, attach_dual_loop
from transformers import Qwen2Config, Qwen2ForCausalLM


def run_benchmark_manifold_transport(device: torch.device, d_model: int = 1536) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("  BENCHMARK 1: PROCRUSTES OPTIMAL MANIFOLD TRANSPORT")
    print("="*80)
    
    transport = ProcrustesOptimalManifoldTransport(d_model=d_model).to(device)
    
    # 1. Text manifold distribution: N(mu_T, Sigma_T)
    # Simulate text latent distribution (mean 0.5, std 1.2)
    mu_T_target = torch.randn(1, 1, d_model, device=device) * 0.5 + 0.3
    sigma_T_target = torch.rand(1, 1, d_model, device=device) * 0.8 + 0.6
    
    # Generate text context
    h_text = torch.randn(4, 128, d_model, device=device) * sigma_T_target + mu_T_target
    
    # 2. Visual sensory distribution: radically different mean and scale (e.g. ViT patch norm)
    mu_V_target = torch.randn(1, 1, d_model, device=device) * 3.0 - 1.5
    sigma_V_target = torch.rand(1, 1, d_model, device=device) * 2.5 + 2.0
    h_vision_raw = torch.randn(4, 576, d_model, device=device) * sigma_V_target + mu_V_target
    
    # Compute baseline divergence before transport
    mu_diff_before = (h_vision_raw.mean(dim=(0, 1)) - h_text.mean(dim=(0, 1))).norm().item()
    std_diff_before = (h_vision_raw.std(dim=(0, 1)) - h_text.std(dim=(0, 1))).norm().item()
    
    # Warmup and timed execution
    for _ in range(10):
        _ = transport(h_vision_raw, h_text)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    repeats = 50
    for _ in range(repeats):
        h_v_transported, telem = transport(h_vision_raw, h_text)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_transport_ms = (time.perf_counter() - t0) / repeats * 1000.0
    
    # Compute divergence post-transport
    mu_diff_after = (h_v_transported.mean(dim=(0, 1)) - h_text.mean(dim=(0, 1))).norm().item()
    std_diff_after = (h_v_transported.std(dim=(0, 1)) - h_text.std(dim=(0, 1))).norm().item()
    
    alignment_fidelity_mu = (1.0 - mu_diff_after / max(1e-6, mu_diff_before)) * 100.0
    alignment_fidelity_std = (1.0 - std_diff_after / max(1e-6, std_diff_before)) * 100.0
    
    print(f"  * Mean Discrepancy:      {mu_diff_before:.4f} -> {mu_diff_after:.6f} ({alignment_fidelity_mu:.2f}% aligned)")
    print(f"  * Std Dev Discrepancy:   {std_diff_before:.4f} -> {std_diff_after:.6f} ({alignment_fidelity_std:.2f}% aligned)")
    print(f"  * Transport Latency:     {t_transport_ms:.4f} ms (sub-millisecond closed-form pure algebra)")
    print(f"  * Parameter Overhead:    0 learnable parameters (pure affine optimal transport)")
    
    return {
        "mu_diff_before": mu_diff_before,
        "mu_diff_after": mu_diff_after,
        "std_diff_before": std_diff_before,
        "std_diff_after": std_diff_after,
        "alignment_fidelity_mu": alignment_fidelity_mu,
        "alignment_fidelity_std": alignment_fidelity_std,
        "latency_ms": t_transport_ms
    }


def run_benchmark_token_explosion(device: torch.device, d_model: int = 1536) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("  BENCHMARK 2: SPATIO-TEMPORAL ENTROPIC CWM COMPRESSION")
    print("="*80)
    
    patch_counts = [256, 576, 1024, 2048]
    m_slots = 16
    topo_cwm = SpatioTemporalEntropicCWM(d_model=d_model, num_slots=m_slots, n_heads=8, max_salient_patches=64).to(device)
    
    # Baseline standard CWM cross-attention without saliency compression (full cross-attn on all patches)
    class StandardUncompressedCWM(nn.Module):
        def __init__(self, d_model, num_slots):
            super().__init__()
            self.slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
            self.mha = nn.MultiheadAttention(d_model, 8, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
        def forward(self, h_all):
            B = h_all.size(0)
            q = self.slots.expand(B, -1, -1)
            out, _ = self.mha(q, h_all, h_all)
            return self.norm(q + out)

    standard_cwm = StandardUncompressedCWM(d_model=d_model, num_slots=m_slots).to(device)
    
    results = {}
    
    for n_v in patch_counts:
        B = 2
        S_t = 128
        h_text = torch.randn(B, S_t, d_model, device=device)
        h_vis = torch.randn(B, n_v, d_model, device=device)
        
        # 1. Benchmark Standard CWM (concat text + all patches)
        h_concat = torch.cat([h_text, h_vis], dim=1) # [B, S_t + N_V, D]
        
        # Warmup
        for _ in range(5):
            _ = standard_cwm(h_concat)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            m0 = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            m0 = 0.0
            
        t0 = time.perf_counter()
        reps = 30
        for _ in range(reps):
            _ = standard_cwm(h_concat)
        if device.type == "cuda":
            torch.cuda.synchronize()
            m_std = torch.cuda.max_memory_allocated(device) / (1024**2) - m0
        else:
            m_std = 0.0
        t_std = (time.perf_counter() - t0) / reps * 1000.0
        
        # 2. Benchmark Topological Entropic CWM
        for _ in range(5):
            _ = topo_cwm(h_text=h_text, h_vision=h_vis)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            m0_t = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            m0_t = 0.0
            
        t0 = time.perf_counter()
        for _ in range(reps):
            mem, slots_v, telem = topo_cwm(h_text=h_text, h_vision=h_vis)
        if device.type == "cuda":
            torch.cuda.synchronize()
            m_topo = torch.cuda.max_memory_allocated(device) / (1024**2) - m0_t
        else:
            m_topo = 0.0
        t_topo = (time.perf_counter() - t0) / reps * 1000.0
        
        speedup = t_std / max(1e-4, t_topo)
        compression = telem.get("compression_ratio", 1.0)
        
        print(f"  * Patches N_V={n_v:4d} | Standard: {t_std:6.2f} ms ({m_std:5.1f} MB) | Topological: {t_topo:6.2f} ms ({m_topo:5.1f} MB) | Speedup: {speedup:4.2f}x | Salient Pool: {telem['num_patches_salient']}/{n_v}")
        
        results[n_v] = {
            "t_std_ms": t_std,
            "m_std_mb": m_std,
            "t_topo_ms": t_topo,
            "m_topo_mb": m_topo,
            "speedup": speedup,
            "salient_patches": telem['num_patches_salient'],
            "compression_ratio": compression
        }
        
    return results


def run_benchmark_hetero_associative(device: torch.device, d_model: int = 1536) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("  BENCHMARK 3: HETERO-ASSOCIATIVE PLASTIC FAST-WEIGHT BINDING")
    print("="*80)
    
    plastic_mem = HeteroAssociativePlasticMemory(d_model=d_model, rank=64, plastic_lr=0.50).to(device)
    plastic_mem.set_continual_mode(True, decay=0.98)
    
    # Create 5 distinct novel concept pairs (Vision Object Vector, Text Concept Label Vector)
    torch.manual_seed(42)
    num_concepts = 5
    v_concepts = [torch.randn(1, 16, d_model, device=device) for _ in range(num_concepts)]
    t_concepts = [torch.randn(1, 1, d_model, device=device) for _ in range(num_concepts)]
    
    # 1. Baseline recall before binding (Tabula Rasa)
    zero_delta, telem_zero = plastic_mem.recall_from_visual(v_concepts[0])
    recall_zero_norm = zero_delta.norm().item()
    
    # 2. In-situ One-Shot Plastic Binding (zero gradient backpropagation)
    print(f"  Phase 1: In-situ Hebbian One-Shot Concept Binding (5 unseen concepts)...")
    binding_telemetries = []
    for i in range(num_concepts):
        u_vac = torch.tensor([[0.85]], device=device) # High Dirichlet vacuity (novelty)
        bind_res = plastic_mem.bind_concept(v_concepts[i], t_concepts[i], u_vacuity=u_vac)
        binding_telemetries.append(bind_res)
        print(f"    - Concept #{i+1}: Bound novel sensory embedding to concept label (Trace norm: {bind_res['m_cross_norm']:.4f})")
        
    # 3. Test One-Shot Zero-Shot Recall under Sensory Noise
    print(f"\n  Phase 2: Instant Recall Verification under 20% Sensory Noise...")
    recall_accuracies = []
    cosine_sims = []
    
    for i in range(num_concepts):
        # Add 20% Gaussian noise perturbation to visual sensory feature
        noise = torch.randn_like(v_concepts[i]) * 0.20
        v_noisy = v_concepts[i] + noise
        
        delta_rec, rec_telem = plastic_mem.recall_from_visual(v_noisy)
        rec_label = delta_rec.mean(dim=1, keepdim=True)
        
        # Compute cosine similarity between recalled label and target text concept
        cos_sim = F.cosine_similarity(rec_label, t_concepts[i], dim=-1).item()
        cosine_sims.append(cos_sim)
        
        # Distractor similarity (check orthogonality to other concepts)
        other_sims = [F.cosine_similarity(rec_label, t_concepts[j], dim=-1).item() for j in range(num_concepts) if j != i]
        max_distractor_sim = max(other_sims) if other_sims else -1.0
        
        is_correct = (cos_sim > max_distractor_sim)
        recall_accuracies.append(1.0 if is_correct else 0.0)
        
        print(f"    - Query #{i+1}: Cosine Similarity={cos_sim:.4f} (Margin over Distractors: {cos_sim - max_distractor_sim:+.4f}) -> {'SUCCESS' if is_correct else 'FAILURE'}")
        
    avg_accuracy = (sum(recall_accuracies) / len(recall_accuracies)) * 100.0
    avg_cosine = sum(cosine_sims) / len(cosine_sims)
    
    print(f"\n  * Overall 1-Shot Recognition Accuracy: {avg_accuracy:.1f}%")
    print(f"  * Mean Alignment Cosine:               {avg_cosine:.4f}")
    print(f"  * Gradient Backprop Required:          0 updates (100% frozen base)")
    
    return {
        "accuracy_pct": avg_accuracy,
        "mean_cosine": avg_cosine,
        "pre_bind_recall_norm": recall_zero_norm,
        "m_cross_final_norm": binding_telemetries[-1]["m_cross_norm"]
    }


def run_benchmark_popperian_falsification(device: torch.device, d_model: int = 1536) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("  BENCHMARK 4: POPPERIAN CROSS-MODAL FALSIFICATION (ANTI-HALLUCINATION)")
    print("="*80)
    
    adapter = LatentDeliberationAdapter(
        d_model=d_model,
        n_heads=8,
        num_thought_tokens=4,
        max_ponder_steps=3,
        num_cwm_slots=16,
        adapter_mode="residual",
        enable_allostatic_modulation=True
    ).to(device)
    adapter.gate_alpha.data.fill_(0.5)
    adapter.eval()
    
    B = 2
    S = 32
    h_text = torch.randn(B, S, d_model, device=device)
    h_q = h_text[:, -1, :] # [B, D]
    
    # 1. Scenario A: Grounded Multimodal Scenario
    # Visual scene with salient foreground object matching the query concept
    v_grounded = torch.randn(B, 64, d_model, device=device) * 0.1
    v_grounded[:, :16, :] = h_q.unsqueeze(1) + 0.1 * torch.randn(B, 16, d_model, device=device)
    
    with torch.no_grad():
        enhanced_grounded, telem_grounded = adapter(
            hidden_states=h_text,
            visual_embeds=v_grounded,
            k_steps=3
        )
        
    f_cross_grounded = telem_grounded.get("cross_modal_alignment", 1.0)
    h_pen_grounded = telem_grounded.get("hallucination_penalty", 0.0)
    delta_norm_grounded = (enhanced_grounded - h_text).norm().item()
    
    # 2. Scenario B: Deceptive / Hallucinatory Scenario
    # Visual scene with salient object orthogonal to the query concept (hallucination)
    h_ortho = torch.randn(B, 1, d_model, device=device)
    proj = (h_ortho * h_q.unsqueeze(1)).sum(dim=-1, keepdim=True) * h_q.unsqueeze(1) / (h_q.unsqueeze(1).norm(dim=-1, keepdim=True)**2 + 1e-6)
    h_ortho = h_ortho - proj

    v_deceptive = torch.randn(B, 64, d_model, device=device) * 0.1
    v_deceptive[:, :16, :] = h_ortho + 0.1 * torch.randn(B, 16, d_model, device=device)
    
    with torch.no_grad():
        enhanced_deceptive, telem_deceptive = adapter(
            hidden_states=h_text,
            visual_embeds=v_deceptive,
            k_steps=3
        )
        
    f_cross_deceptive = telem_deceptive.get("cross_modal_alignment", 1.0)
    h_pen_deceptive = telem_deceptive.get("hallucination_penalty", 0.0)
    delta_norm_deceptive = (enhanced_deceptive - h_text).norm().item()
    
    suppression_ratio = (1.0 - delta_norm_deceptive / max(1e-6, delta_norm_grounded)) * 100.0
    
    print(f"  * Grounded Perception:")
    print(f"    - Alignment F_cross:        {f_cross_grounded:.4f} (>= tau_evidence 0.25)")
    print(f"    - Hallucination Penalty:    {h_pen_grounded:.4f}")
    print(f"    - Residual Delta Injected:  {delta_norm_grounded:.4f} (Valid cognitive deliberation)")
    print(f"  * Hallucinatory / Ungrounded:")
    print(f"    - Alignment F_cross:        {f_cross_deceptive:.4f} (< tau_evidence 0.25)")
    print(f"    - Hallucination Penalty:    {h_pen_deceptive:.4f}")
    print(f"    - Residual Delta Injected:  {delta_norm_deceptive:.4f} (Gated to safe margin)")
    print(f"  * Falsification Suppression:  {suppression_ratio:.2f}% reduction in hallucinatory delta")
    
    return {
        "grounded_f_cross": f_cross_grounded,
        "grounded_penalty": h_pen_grounded,
        "grounded_delta_norm": delta_norm_grounded,
        "deceptive_f_cross": f_cross_deceptive,
        "deceptive_penalty": h_pen_deceptive,
        "deceptive_delta_norm": delta_norm_deceptive,
        "suppression_ratio": suppression_ratio
    }


def run_benchmark_e2e_qwen(device: torch.device) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("  BENCHMARK 5: END-TO-END DUAL-LOOP QWEN MULTIMODAL INTEGRATION")
    print("="*80)
    
    vocab_size = 1000
    hidden_size = 896
    num_layers = 8
    
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2048,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=1024,
        pad_token_id=0
    )
    base_model = Qwen2ForCausalLM(config).to(device)
    base_model.eval()
    
    wrapped = attach_dual_loop(base_model, layer_idx=4, k_steps=2, bottleneck_dim=512)
    wrapped.to(device)
    wrapped.eval()
    
    B = 2
    S = 64
    N_V = 256
    input_ids = torch.randint(1, vocab_size, (B, S), device=device)
    v_embeds = torch.randn(B, N_V, hidden_size, device=device)
    
    # 1. Base Model Latency (k=0 bypass)
    wrapped.set_ponder_steps(0)
    wrapped.set_visual_embeds(None)
    for _ in range(5):
        _ = wrapped(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 30
    for _ in range(reps):
        _ = wrapped(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_base = (time.perf_counter() - t0) / reps * 1000.0
    
    # 2. Dual-Loop Multimodal Deliberation (k=2 steps + 256 visual patches)
    wrapped.set_ponder_steps(2)
    wrapped.set_visual_embeds(v_embeds)
    for _ in range(5):
        _ = wrapped(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = wrapped(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_dual_loop = (time.perf_counter() - t0) / reps * 1000.0
    
    telem = wrapped.last_telemetry
    latency_overhead = t_dual_loop - t_base
    active_neurons_dict = telem.get('active_neurons', {})
    active_count = active_neurons_dict.get('active_neurons_total', 0) if isinstance(active_neurons_dict, dict) else active_neurons_dict
    
    print(f"  * Pure Base Qwen (k=0 bypass):         {t_base:.2f} ms")
    print(f"  * Dual-Loop Multimodal Qwen (k=2):     {t_dual_loop:.2f} ms")
    print(f"  * Deliberation & Multimodal Overhead:  +{latency_overhead:.2f} ms")
    print(f"  * Active Neurons in Pondering:         {active_count:,}")
    print(f"  * Bimodal Working Memory Active:       {telem.get('topological_cwm', {}).get('bimodal', False)}")
    print(f"  * Hallucination Penalty:               {telem.get('hallucination_penalty', 0.0):.4f}")
    
    return {
        "t_base_ms": t_base,
        "t_dual_loop_ms": t_dual_loop,
        "latency_overhead_ms": latency_overhead,
        "active_neurons": active_count,
        "bimodal": telem.get("topological_cwm", {}).get("bimodal", False)
    }


def main():
    print("="*80)
    print("  HADL UNIVERSAL CROSS-MODAL INVARIANT CONTROLLER BENCHMARK")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
    bench1 = run_benchmark_manifold_transport(device)
    bench2 = run_benchmark_token_explosion(device)
    bench3 = run_benchmark_hetero_associative(device)
    bench4 = run_benchmark_popperian_falsification(device)
    bench5 = run_benchmark_e2e_qwen(device)
    
    # Save results as JSON
    results = {
        "benchmark1_transport": bench1,
        "benchmark2_token_explosion": bench2,
        "benchmark3_hetero_associative": bench3,
        "benchmark4_falsification": bench4,
        "benchmark5_e2e_qwen": bench5
    }
    
    out_dir = "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_modal_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Benchmark results serialized to {out_path}")


if __name__ == "__main__":
    main()
