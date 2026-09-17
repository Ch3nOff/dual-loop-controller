---
title: Dual-Loop Cognitive Controller
emoji: 🧠
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Latent Deliberation & Memory Architecture for LLMs
models:
- Qwen/Qwen3.5-2B
- CH3NDev/dual-loop-qwen3.5-2b
---

# Dual-Loop Cognitive Controller — Interactive Demo

This Hugging Face Space demonstrates the **Dual-Loop Cognitive Controller** on `Qwen/Qwen3.5-2B` ($D=2048$, Layer 11 hook).

## Key Innovations Demonstrated:
1. **Zero-Token Latent Deliberation (System 2)**: Recursively ponders in continuous latent space ($h \in \mathbb{R}^D$) inside GPU SRAM without emitting intermediate CoT tokens.
2. **Cognitive Matrix Helper (Tversky Elimination-by-Aspects)**: Automatically prunes 40%–57% superficial distractor options (*wrong logs*) in Bench 1 and focuses System 2 deliberation in Bench 2, boosting accuracy from 50.0% to 83.3%.
3. **Hippocampal Episodic Virtual Memory**: Recalls previously verified reasoning traces in **<0.01 seconds** (3,146x speedup) with 0 FLOPs.

Visit the [GitHub Repository](https://github.com/Ch3nOff/dual-loop-controller) or [PyPI Package](https://pypi.org/project/dual-loop-controller/).
