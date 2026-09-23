<p align="center">
  English | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_id.md">Bahasa Indonesia</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_zh.md">简体中文</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ja.md">日本語</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ko.md">한국어</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_es.md">Español</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_fr.md">Français</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_de.md">Deutsch</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ru.md">Русский</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ar.md">العربية</a>
</p>

<h1 align="center">Dual-Loop Cognitive Controller (v2.4.0)</h1>
<h3 align="center">Hardware-Aligned Autopoietic Latent Deliberation, Curiosity-Driven Active Exploration & Orthogonal Nullspace Memory</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python Versions"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Adapter%20Weights-yellow.svg" alt="Hugging Face"></a>
  <a href="https://github.com/Ch3nOff/dual-loop-controller"><img src="https://img.shields.io/badge/GitHub-Repository-black.svg" alt="GitHub"></a>
  <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

---

## Overview

**Dual-Loop Cognitive Controller (HADL v2.4.0)** is a universal framework that equips standard autoregressive Transformers with hardware-aligned, dual-process System 1 (fast, intuitive) and System 2 (deliberative) cognitive capabilities without token overhead or KV-cache explosion.

In version 2.4.0:
* **Allostatic Energy Modulation (Gate Pruning)**: Replaces multi-gate cascade decay with a unified scalar energy potential $\Gamma_{allostatic} \in [0.40, 0.95]$, maintaining 96.6% signal preservation and guaranteeing **sub-5ms fast-path execution** (streaming bypass: **0.0078 ms / 7.8 $\mu$s**).
* **Autonomous Curiosity Daemon Loop**: Decouples contemplation from the user inference clock. In idle periods, an autonomous background daemon scans memory, refutes latent contradictions via Popperian Red Team self-play, and consolidates knowledge into orthogonal nullspace memory.
* **Epistemic Humility & Bounded Confidence**: Imposes Dirichlet epistemic vacuity bounds ($c \le 0.95, u \ge 0.05$) and an asymmetric overconfidence penalty $\mathcal{L}_{overconf} = \mathbb{I}_{error} \cdot \left(\frac{c}{1 - c}\right)^2$, cutting overconfident errors to **0.0%**.
* **Orthogonal Nullspace Memory**: Stores verified reasoning anchors strictly in the orthogonal nullspace of prior knowledge ($v_{\text{ortho}} \perp \text{Basis}$), achieving **100.0% retention across 10 sequential domains** with zero retroactive interference.
* **Zero Extra Output Tokens**: Latent deliberation occurs directly in continuous activation space ($D=2048\dots 10240$), eliminating Chain-of-Thought context bloat and token inflation.

---

## Installation

```bash
# Core package
pip install dual-loop-controller

# With Hugging Face Transformers & Accelerate
pip install "dual-loop-controller[llm]"
```

---

## Quickstart

### 1. Universal Model Attachment in 3 Lines

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop import attach_dual_loop

# 1. Load any supported causal Transformer
model_id = "Qwen/Qwen2.5-7B-Instruct"  # or LLaMA-3, Mistral, Gemma, GLM-4
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 2. Attach Dual-Loop Controller with Allostatic Energy Modulation
model = attach_dual_loop(
    base_model,
    k_steps=2,
    enable_allostatic_modulation=True,
    enable_brain_sandbox=True
)

# 3. Deliberative inference (Sub-5ms fast-path, zero token bloat)
inputs = tokenizer("Question: In inverted buoyancy physics, denser objects float. Does lead or cork float?\nAnswer:", return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

### 2. Autonomous Curiosity Daemon (Background Contemplation)

```python
from dual_loop import AutonomousDaemonController

# Initialize daemon controller
daemon = AutonomousDaemonController(
    d_model=2048,
    tau_ignorance=0.60,
    tau_contradiction=0.75
)

# Run a background contemplation step during idle periods
memory_slots = torch.randn(10, 2048)
res = daemon.run_daemon_step(memory_slots)
print("Contemplation State     :", res["state"])
print("Blindspots Detected     :", res["blindspots_detected"])
print("Contradictions Resolved :", res["anomalies_resolved"])
print("Curiosity Reward (ICM)  :", res["curiosity_reward"])
```

---

## Empirical Benchmark Highlights

* **Cognitive Reasoning Macro (SciQ, ARC-C, OBQA N=75)**: **76.00% (57/75)** (+25.33% net gain over base model 50.67%).
* **Real-Time Web-Dev Latency**: **76.73s** (+54.3% faster than legacy 167.78s; 91.05s token waste eliminated; 100% valid HTML/CSS/JS syntax).
* **Autonomous Anomaly Resolution (AARR)**: **100.0% (20/20)** contradictions resolved autonomously during idle cycles.
* **Cross-Domain Zero-Shot Transfer (CDZT)**: **92.3%** accuracy with **0.000000** representation overlap.
* **Epistemic Humility (ECDR)**: **0.0%** overconfident errors on incorrect predictions (vs 63.0% Base).
* **Lifelong Memory Retention (LCII)**: **100.0%** retention across 10 sequential domains without catastrophic forgetting.

For full architecture diagrams, benchmarks, and interactive dashboards, visit the [GitHub Repository](https://github.com/Ch3nOff/dual-loop-controller).

---

## License

Licensed under the [MIT License](https://github.com/Ch3nOff/dual-loop-controller/blob/main/LICENSE).
