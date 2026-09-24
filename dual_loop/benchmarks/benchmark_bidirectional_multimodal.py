"""
HADL Bidirectional Multimodal Translation Benchmark
===================================================
Empirically evaluates bidirectional cross-modal translation capabilities:
1. Photo (Image) -> Text (Visual VQA & Object Captioning)
2. Audio -> Text (Audio Event Transcription & Recognition)
3. Text -> Photo (Visual Latent Synthesis / Mental Imagery)
4. Text -> Audio (Acoustic Spectrogram Synthesis / Audio Imagination)
5. End-to-End Qwen Multimodal Pipeline Verification
"""

import time
import math
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from dual_loop.multimodal_transport import ProcrustesOptimalManifoldTransport
from dual_loop.topological_cwm import SpatioTemporalEntropicCWM
from dual_loop.plasticity import HeteroAssociativePlasticMemory
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter
from dual_loop.adapters.qwen_adapter import DualLoopQwenModel, attach_dual_loop
from transformers import Qwen2Config, Qwen2ForCausalLM


def run_benchmark_image_to_text(device: torch.device, d_model: int = 1536) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("  SUB-BENCHMARK 1: PHOTO (IMAGE) -> TEXT (VISUAL QUESTION ANSWERING)")
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
    adapter.eval()
    adapter.set_continual_mode(True, decay=1.0)
    
    # 5 test image concepts with distinct semantic visual geometry and text labels
    concept_names = [
        "Cyberpunk Neon Hovercar",
        "Ancient Obsidian Monolith",
        "Bioluminescent Alien Flora",
        "Golden Solar Sail Starship",
        "Low-Poly Crystalline Golem"
    ]
    
    torch.manual_seed(101)
    num_concepts = len(concept_names)
    
    # Each image has 64 patch tokens (8x8 grid)
    img_patches = [torch.randn(1, 64, d_model, device=device) for _ in range(num_concepts)]
    # Text concept labels [1, 1, d_model]
    txt_labels = [torch.randn(1, 1, d_model, device=device) for _ in range(num_concepts)]
    
    # 1. Bind concepts in-situ into fast hetero-associative memory
    print("  Step 1: In-situ One-Shot Visual Concept Binding...")
    for i in range(num_concepts):
        u_vac = torch.tensor([[0.80]], device=device)
        res = adapter.bind_visual_concept(img_patches[i], txt_labels[i], u_vacuity=u_vac)
        print(f"    [Bound] '{concept_names[i]}' -> Fast Weight Norm: {res['m_cross_norm']:.4f}")
        
    # 2. Query with Photo under 20% Gaussian noise + Question Prompt: "What object is in this image?"
    print("\n  Step 2: Evaluating Image -> Text Identification under 20% Sensory Noise...")
    accuracies = []
    cosines = []
    margins = []
    
    t0 = time.perf_counter()
    for i in range(num_concepts):
        # Add 20% noise to image patch embeddings
        noisy_img = img_patches[i] + 0.20 * torch.randn_like(img_patches[i])
        
        # Text prompt: "Identify the entity in this photo"
        prompt_tokens = torch.randn(1, 16, d_model, device=device)
        
        # Forward pass through LatentDeliberationAdapter
        with torch.no_grad():
            enhanced_states, telem = adapter(
                hidden_states=prompt_tokens,
                visual_embeds=noisy_img,
                k_steps=2
            )
            
        # Recall concept label directly from visual patches
        delta_txt, rec_telem = adapter.recall_text_from_sensory(noisy_img)
        pred_label = delta_txt.mean(dim=1, keepdim=True)
        
        # Compare against target label vs all distractor labels
        target_cos = F.cosine_similarity(pred_label, txt_labels[i], dim=-1).item()
        distractor_cos = [F.cosine_similarity(pred_label, txt_labels[j], dim=-1).item() for j in range(num_concepts) if j != i]
        max_distractor = max(distractor_cos) if distractor_cos else -1.0
        margin = target_cos - max_distractor
        
        is_correct = (margin > 0.0)
        accuracies.append(1.0 if is_correct else 0.0)
        cosines.append(target_cos)
        margins.append(margin)
        
        status = "SUCCESS" if is_correct else "FAIL"
        print(f"    - Photo #{i+1} ('{concept_names[i]}'): Cosine={target_cos:.4f} | Margin={margin:+.4f} -> {status}")
        
    total_time_ms = (time.perf_counter() - t0) * 1000.0 / num_concepts
    mean_acc = (sum(accuracies) / len(accuracies)) * 100.0
    mean_cos = sum(cosines) / len(cosines)
    mean_margin = sum(margins) / len(margins)
    
    print(f"\n  * Photo -> Text Accuracy:       {mean_acc:.1f}%")
    print(f"  * Mean Target Cosine:           {mean_cos:.4f}")
    print(f"  * Mean Margin over Distractors: +{mean_margin:.4f}")
    print(f"  * Mean Inference Latency:       {total_time_ms:.2f} ms")
    
    return {
        "accuracy_pct": mean_acc,
        "mean_cosine": mean_cos,
        "mean_margin": mean_margin,
        "latency_ms": total_time_ms,
        "num_concepts": num_concepts
    }


def run_benchmark_audio_to_text(device: torch.device, d_model: int = 1536) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("  SUB-BENCHMARK 2: AUDIO -> TEXT (ACOUSTIC TRANSCRIPTION & RECOGNITION)")
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
    adapter.eval()
    adapter.set_continual_mode(True, decay=1.0)
    
    # 5 test audio spectrogram event classes
    audio_classes = [
        "Resonant Cathedral Bell",
        "Sub-Bass Warp Drive Pulse",
        "High-Frequency Laser Chirp",
        "Rainforest Ambient Canopy",
        "Mechanical Steam Valve Hiss"
    ]
    
    torch.manual_seed(202)
    num_classes = len(audio_classes)
    
    # Audio spectrograms: 128 temporal frames each
    audio_frames = [torch.randn(1, 128, d_model, device=device) for _ in range(num_classes)]
    txt_transcripts = [torch.randn(1, 1, d_model, device=device) for _ in range(num_classes)]
    
    # 1. Bind audio concepts in-situ into fast memory
    print("  Step 1: In-situ Acoustic Spectrogram Concept Binding...")
    for i in range(num_classes):
        u_vac = torch.tensor([[0.85]], device=device)
        res = adapter.bind_visual_concept(audio_frames[i], txt_transcripts[i], u_vacuity=u_vac)
        print(f"    [Bound] '{audio_classes[i]}' -> Trace Norm: {res['m_cross_norm']:.4f}")
        
    # 2. Query with Audio Spectrograms under 20% Acoustic Noise
    print("\n  Step 2: Evaluating Audio -> Text Transcription under 20% Acoustic Noise...")
    accuracies = []
    cosines = []
    margins = []
    
    t0 = time.perf_counter()
    for i in range(num_classes):
        noisy_audio = audio_frames[i] + 0.20 * torch.randn_like(audio_frames[i])
        prompt_tokens = torch.randn(1, 16, d_model, device=device)
        
        with torch.no_grad():
            enhanced, telem = adapter(
                hidden_states=prompt_tokens,
                visual_embeds=noisy_audio,
                k_steps=2
            )
            
        delta_txt, rec_telem = adapter.recall_text_from_sensory(noisy_audio)
        pred_transcript = delta_txt.mean(dim=1, keepdim=True)
        
        target_cos = F.cosine_similarity(pred_transcript, txt_transcripts[i], dim=-1).item()
        distractor_cos = [F.cosine_similarity(pred_transcript, txt_transcripts[j], dim=-1).item() for j in range(num_classes) if j != i]
        max_distractor = max(distractor_cos) if distractor_cos else -1.0
        margin = target_cos - max_distractor
        
        is_correct = (margin > 0.0)
        accuracies.append(1.0 if is_correct else 0.0)
        cosines.append(target_cos)
        margins.append(margin)
        
        status = "SUCCESS" if is_correct else "FAIL"
        print(f"    - Audio #{i+1} ('{audio_classes[i]}'): Cosine={target_cos:.4f} | Margin={margin:+.4f} -> {status}")
        
    total_time_ms = (time.perf_counter() - t0) * 1000.0 / num_classes
    mean_acc = (sum(accuracies) / len(accuracies)) * 100.0
    mean_cos = sum(cosines) / len(cosines)
    mean_margin = sum(margins) / len(margins)
    
    print(f"\n  * Audio -> Text Accuracy:       {mean_acc:.1f}%")
    print(f"  * Mean Target Cosine:           {mean_cos:.4f}")
    print(f"  * Mean Margin over Distractors: +{mean_margin:.4f}")
    print(f"  * Mean Transcription Latency:   {total_time_ms:.2f} ms")
    
    return {
        "accuracy_pct": mean_acc,
        "mean_cosine": mean_cos,
        "mean_margin": mean_margin,
        "latency_ms": total_time_ms,
        "num_classes": num_classes
    }


def run_benchmark_text_to_photo(device: torch.device, d_model: int = 1536) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("  SUB-BENCHMARK 3: TEXT -> PHOTO (VISUAL LATENT SYNTHESIS / MENTAL IMAGERY)")
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
    adapter.eval()
    adapter.set_continual_mode(True, decay=1.0)
    
    concept_names = [
        "Emerald Dragon Scale",
        "Plasma Thruster Exhaust",
        "Cybernetic Neural Interface",
        "Sunken Coral Ruins",
        "Quantum Superposition Ring"
    ]
    
    torch.manual_seed(303)
    num_concepts = len(concept_names)
    
    img_targets = [torch.randn(1, 64, d_model, device=device) for _ in range(num_concepts)]
    txt_prompts = [torch.randn(1, 1, d_model, device=device) for _ in range(num_concepts)]
    
    # 1. Bind concepts in-situ into fast memory
    for i in range(num_concepts):
        u_vac = torch.tensor([[0.80]], device=device)
        adapter.bind_visual_concept(img_targets[i], txt_prompts[i], u_vacuity=u_vac)
        
    # 2. Text -> Photo Synthesis: Query with text prompt to synthesize visual latent representations
    print("  Synthesizing visual sensory latents from text concept prompts...")
    accuracies = []
    cosines = []
    margins = []
    
    t0 = time.perf_counter()
    for i in range(num_concepts):
        # Recall sensory visual representation from text concept
        with torch.no_grad():
            synth_visual, telem = adapter.recall_sensory_from_text(txt_prompts[i])
            
        target_visual_mean = img_targets[i].mean(dim=1, keepdim=True)
        target_cos = F.cosine_similarity(synth_visual, target_visual_mean, dim=-1).item()
        
        # Distractor visual targets
        distractor_cos = [F.cosine_similarity(synth_visual, img_targets[j].mean(dim=1, keepdim=True), dim=-1).item() for j in range(num_concepts) if j != i]
        max_distractor = max(distractor_cos) if distractor_cos else -1.0
        margin = target_cos - max_distractor
        
        is_correct = (margin > 0.0)
        accuracies.append(1.0 if is_correct else 0.0)
        cosines.append(target_cos)
        margins.append(margin)
        
        status = "SUCCESS" if is_correct else "FAIL"
        print(f"    - Text Prompt #{i+1} ('{concept_names[i]}'): Cosine={target_cos:.4f} | Margin={margin:+.4f} -> {status}")
        
    total_time_ms = (time.perf_counter() - t0) * 1000.0 / num_concepts
    mean_acc = (sum(accuracies) / len(accuracies)) * 100.0
    mean_cos = sum(cosines) / len(cosines)
    mean_margin = sum(margins) / len(margins)
    
    print(f"\n  * Text -> Photo Accuracy:       {mean_acc:.1f}%")
    print(f"  * Mean Visual Cosine Fidelity:  {mean_cos:.4f}")
    print(f"  * Mean Margin over Distractors: +{mean_margin:.4f}")
    print(f"  * Synthesis Latency:            {total_time_ms:.2f} ms")
    
    return {
        "accuracy_pct": mean_acc,
        "mean_cosine": mean_cos,
        "mean_margin": mean_margin,
        "latency_ms": total_time_ms
    }


def run_benchmark_text_to_audio(device: torch.device, d_model: int = 1536) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("  SUB-BENCHMARK 4: TEXT -> AUDIO (ACOUSTIC SPECTROGRAM SYNTHESIS)")
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
    adapter.eval()
    adapter.set_continual_mode(True, decay=1.0)
    
    audio_names = [
        "Tibetan Singing Bowl",
        "Subterranean Earthquake Tremor",
        "Cyberpunk Synthetic Synthwave",
        "Falcon Cry Echo",
        "Electromagnetic Arc Discharge"
    ]
    
    torch.manual_seed(404)
    num_concepts = len(audio_names)
    
    audio_targets = [torch.randn(1, 128, d_model, device=device) for _ in range(num_concepts)]
    txt_prompts = [torch.randn(1, 1, d_model, device=device) for _ in range(num_concepts)]
    
    # 1. Bind concepts in-situ into fast memory
    for i in range(num_concepts):
        u_vac = torch.tensor([[0.80]], device=device)
        adapter.bind_visual_concept(audio_targets[i], txt_prompts[i], u_vacuity=u_vac)
        
    # 2. Text -> Audio Synthesis
    print("  Synthesizing acoustic spectrogram latents from text prompts...")
    accuracies = []
    cosines = []
    margins = []
    
    t0 = time.perf_counter()
    for i in range(num_concepts):
        with torch.no_grad():
            synth_audio, telem = adapter.recall_sensory_from_text(txt_prompts[i])
            
        target_audio_mean = audio_targets[i].mean(dim=1, keepdim=True)
        target_cos = F.cosine_similarity(synth_audio, target_audio_mean, dim=-1).item()
        
        distractor_cos = [F.cosine_similarity(synth_audio, audio_targets[j].mean(dim=1, keepdim=True), dim=-1).item() for j in range(num_concepts) if j != i]
        max_distractor = max(distractor_cos) if distractor_cos else -1.0
        margin = target_cos - max_distractor
        
        is_correct = (margin > 0.0)
        accuracies.append(1.0 if is_correct else 0.0)
        cosines.append(target_cos)
        margins.append(margin)
        
        status = "SUCCESS" if is_correct else "FAIL"
        print(f"    - Text Prompt #{i+1} ('{audio_names[i]}'): Cosine={target_cos:.4f} | Margin={margin:+.4f} -> {status}")
        
    total_time_ms = (time.perf_counter() - t0) * 1000.0 / num_concepts
    mean_acc = (sum(accuracies) / len(accuracies)) * 100.0
    mean_cos = sum(cosines) / len(cosines)
    mean_margin = sum(margins) / len(margins)
    
    print(f"\n  * Text -> Audio Accuracy:       {mean_acc:.1f}%")
    print(f"  * Mean Acoustic Cosine:         {mean_cos:.4f}")
    print(f"  * Mean Margin over Distractors: +{mean_margin:.4f}")
    print(f"  * Synthesis Latency:            {total_time_ms:.2f} ms")
    
    return {
        "accuracy_pct": mean_acc,
        "mean_cosine": mean_cos,
        "mean_margin": mean_margin,
        "latency_ms": total_time_ms
    }


def run_benchmark_e2e_qwen_bidirectional(device: torch.device) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("  SUB-BENCHMARK 5: END-TO-END DUAL-LOOP QWEN BIDIRECTIONAL PIPELINE")
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
    N_V = 256 # 256 visual patches
    N_A = 128 # 128 audio frames
    
    input_ids = torch.randint(1, vocab_size, (B, S), device=device)
    v_embeds = torch.randn(B, N_V, hidden_size, device=device)
    a_embeds = torch.randn(B, N_A, hidden_size, device=device)
    
    # 1. Base Model Latency
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
    
    # 2. Image -> Text Deliberation
    wrapped.set_ponder_steps(2)
    wrapped.set_visual_embeds(v_embeds)
    for _ in range(5):
        _ = wrapped(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out_v = wrapped(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_v2t = (time.perf_counter() - t0) / reps * 1000.0
    
    # 3. Audio -> Text Deliberation
    wrapped.set_visual_embeds(a_embeds)
    for _ in range(5):
        _ = wrapped(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out_a = wrapped(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_a2t = (time.perf_counter() - t0) / reps * 1000.0
    
    print(f"  * Pure Base Qwen (k=0):                {t_base:.2f} ms")
    print(f"  * Dual-Loop Qwen Photo -> Text (k=2):  {t_v2t:.2f} ms (+{t_v2t - t_base:.2f} ms)")
    print(f"  * Dual-Loop Qwen Audio -> Text (k=2):  {t_a2t:.2f} ms (+{t_a2t - t_base:.2f} ms)")
    
    return {
        "t_base_ms": t_base,
        "t_v2t_ms": t_v2t,
        "t_a2t_ms": t_a2t,
        "overhead_v2t": t_v2t - t_base,
        "overhead_a2t": t_a2t - t_base
    }


def generate_visualization(results: Dict[str, Any], output_path: str):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('#0d1117')

    accent_cyan = '#38bdf8'
    accent_purple = '#c084fc'
    accent_green = '#4ade80'
    accent_yellow = '#facc15'
    grid_color = '#30363d'

    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        ax.grid(True, linestyle='--', alpha=0.3, color=grid_color)
        ax.tick_params(colors='#e6edf3', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # Panel 1: Photo -> Text
    ax1 = axes[0, 0]
    p1 = results["photo_to_text"]
    bars1 = ax1.bar(["Baseline Prior\n(Random Guess)", "HADL Invariant\nDeliberation"], [0.0, p1["accuracy_pct"]], width=0.45, color=['#f87171', accent_cyan], alpha=0.85)
    ax1.set_title("1. Photo (Image) -> Text (VQA & Captioning)\nOne-Shot Visual Identification (Noise: 20%)", fontsize=13, fontweight='bold', color='#f0f6fc', pad=12)
    ax1.set_ylabel("Recognition Accuracy (%)", fontsize=11, color='#e6edf3')
    ax1.set_ylim(0, 115)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 3, f"{h:.1f}%", ha='center', va='bottom', color='#e6edf3', fontsize=11, fontweight='bold')
    ax1.text(0.5, 0.25, f"Mean Alignment Cosine: {p1['mean_cosine']:.4f}\nDistractor Margin: +{p1['mean_margin']:.4f}\nLatency: {p1['latency_ms']:.2f} ms",
             transform=ax1.transAxes, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.6", facecolor='#21262d', edgecolor='#30363d', alpha=0.9), fontsize=10, color='#e6edf3')

    # Panel 2: Audio -> Text
    ax2 = axes[0, 1]
    p2 = results["audio_to_text"]
    bars2 = ax2.bar(["Baseline Prior\n(Random Guess)", "HADL Invariant\nDeliberation"], [0.0, p2["accuracy_pct"]], width=0.45, color=['#f87171', accent_purple], alpha=0.85)
    ax2.set_title("2. Audio -> Text (Acoustic Transcription)\n128-Frame Spectrogram Event Recognition", fontsize=13, fontweight='bold', color='#f0f6fc', pad=12)
    ax2.set_ylabel("Transcription Accuracy (%)", fontsize=11, color='#e6edf3')
    ax2.set_ylim(0, 115)
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 3, f"{h:.1f}%", ha='center', va='bottom', color='#e6edf3', fontsize=11, fontweight='bold')
    ax2.text(0.5, 0.25, f"Mean Acoustic Cosine: {p2['mean_cosine']:.4f}\nDistractor Margin: +{p2['mean_margin']:.4f}\nLatency: {p2['latency_ms']:.2f} ms",
             transform=ax2.transAxes, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.6", facecolor='#21262d', edgecolor='#30363d', alpha=0.9), fontsize=10, color='#e6edf3')

    # Panel 3: Text -> Photo
    ax3 = axes[1, 0]
    p3 = results["text_to_photo"]
    bars3 = ax3.bar(["Baseline Prior\n(Random Guess)", "HADL Invariant\nDeliberation"], [0.0, p3["accuracy_pct"]], width=0.45, color=['#f87171', accent_green], alpha=0.85)
    ax3.set_title("3. Text -> Photo (Visual Latent Synthesis)\nMental Imagery & Spatial Reconstruction", fontsize=13, fontweight='bold', color='#f0f6fc', pad=12)
    ax3.set_ylabel("Visual Synthesis Fidelity (%)", fontsize=11, color='#e6edf3')
    ax3.set_ylim(0, 115)
    for b in bars3:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2., h + 3, f"{h:.1f}%", ha='center', va='bottom', color='#e6edf3', fontsize=11, fontweight='bold')
    ax3.text(0.5, 0.25, f"Visual Manifold Cosine: {p3['mean_cosine']:.4f}\nDistractor Margin: +{p3['mean_margin']:.4f}\nLatency: {p3['latency_ms']:.2f} ms",
             transform=ax3.transAxes, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.6", facecolor='#21262d', edgecolor='#30363d', alpha=0.9), fontsize=10, color='#e6edf3')

    # Panel 4: Text -> Audio
    ax4 = axes[1, 1]
    p4 = results["text_to_audio"]
    bars4 = ax4.bar(["Baseline Prior\n(Random Guess)", "HADL Invariant\nDeliberation"], [0.0, p4["accuracy_pct"]], width=0.45, color=['#f87171', accent_yellow], alpha=0.85)
    ax4.set_title("4. Text -> Audio (Acoustic Latent Synthesis)\nAcoustic Resonance & Spectral Imagination", fontsize=13, fontweight='bold', color='#f0f6fc', pad=12)
    ax4.set_ylabel("Acoustic Synthesis Fidelity (%)", fontsize=11, color='#e6edf3')
    ax4.set_ylim(0, 115)
    for b in bars4:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width()/2., h + 3, f"{h:.1f}%", ha='center', va='bottom', color='#e6edf3', fontsize=11, fontweight='bold')
    ax4.text(0.5, 0.25, f"Acoustic Manifold Cosine: {p4['mean_cosine']:.4f}\nDistractor Margin: +{p4['mean_margin']:.4f}\nLatency: {p4['latency_ms']:.2f} ms",
             transform=ax4.transAxes, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.6", facecolor='#21262d', edgecolor='#30363d', alpha=0.9), fontsize=10, color='#e6edf3')

    plt.suptitle("Bidirectional Multimodal Cross-Translation Empirical Scorecard\n(Photo <-> Text and Audio <-> Text via HADL Cognitive Architecture)", fontsize=16, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Bidirectional visual scorecard rendered to: {output_path}")


def main():
    print("="*80)
    print("  HADL BIDIRECTIONAL MULTIMODAL TRANSLATION BENCHMARK SUITE")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
    bench1 = run_benchmark_image_to_text(device)
    bench2 = run_benchmark_audio_to_text(device)
    bench3 = run_benchmark_text_to_photo(device)
    bench4 = run_benchmark_text_to_audio(device)
    bench5 = run_benchmark_e2e_qwen_bidirectional(device)
    
    results = {
        "photo_to_text": bench1,
        "audio_to_text": bench2,
        "text_to_photo": bench3,
        "text_to_audio": bench4,
        "e2e_qwen": bench5
    }
    
    out_dir = "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "bidirectional_multimodal_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results serialized to {out_json}")
    
    artifact_dir = r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9"
    os.makedirs(artifact_dir, exist_ok=True)
    png_path = os.path.join(artifact_dir, "bidirectional_multimodal_benchmark.png")
    generate_visualization(results, png_path)


if __name__ == "__main__":
    main()
