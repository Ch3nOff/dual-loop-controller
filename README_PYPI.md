<p align="center">
  English | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_id.md">Bahasa Indonesia</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_zh.md">简体中文</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ja.md">日本語</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ko.md">한국어</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_es.md">Español</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_fr.md">Français</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_de.md">Deutsch</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ru.md">Русский</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ar.md">العربية</a>
</p>

<h1 align="center">Dual-Loop Cognitive Controller (HADL v3.2.0)</h1>
<h3 align="center">Unified Cognitive OS: SquareCloud Simplex, Dynamic Moving Points, Fast-Slow Surprisal Routing & Unitary Isometry</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python Versions"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Adapter%20Weights-yellow.svg" alt="Hugging Face"></a>
  <a href="https://github.com/Ch3nOff/dual-loop-controller"><img src="https://img.shields.io/badge/GitHub-Repository-black.svg" alt="GitHub"></a>
  <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/Ch3nOff/dual-loop-controller/tree/main/tests"><img src="https://img.shields.io/badge/tests-144%20passed%20(100%25)-brightgreen.svg" alt="Unit Tests"></a>
</p>

---

## Overview

**Dual-Loop Cognitive Controller (HADL v3.2.0 Unified Cognitive OS)** is an open-source, model-agnostic cognitive framework that upgrades ANY autoregressive Transformer into an autonomous dual-process cognitive operating system with 5 Computational Brain Organs and the SquareCloud Dynamic Cognitive Engine:

1. **Universal Model-Agnostic Deliberation (Organ 1)**: Dynamic Runtime Graph Introspection (`DynamicGraphIntrospector`) discovers layer containers and hooks automatically across Qwen, LLaMA, Mistral, Gemma, and GLM-4. Standardizes latent deliberation onto a **Canonical Manifold** ($\mathbb{R}^{D_{native}} \to \mathbb{R}^{1024} \to \mathbb{R}^{D_{native}}$) with mathematical ReZero Identity Preservation ($\Delta_{init} \equiv 0$).
2. **Hardware-Aligned Dynamic VRAM Auto-Tuning**: Automatically profiles hardware and selects optimal execution regimes (BF16, INT8, NF4) with guaranteed memory headroom to prevent Out-Of-Memory (OOM) errors on consumer GPUs.
3. **SquareCloud Dynamic Engine (NextGen)**: Selective Identity Matrix Router ($\mathbf{M}_{\text{select}}$), Bounded Simplex Density Cloud, Dynamic Moving Particle Points ($[V \odot K]$), and Unitary Givens Trigonometric Rotations ($\|h'\|_2 \equiv \|h\|_2$).
4. **Allostatic Energy Modulator & Friston Policy Router (Organ 2)**: Dynamically routes execution between Fast-Path Streaming Bypass ($7.8\ \mu\text{s}$), Fast Evidential Checking, and Recurrent Deliberation, eliminating dead neurons and logit space attenuation.
5. **Sleep-Phase Consolidation Engine (Organ 4)**: Offline memory replay translating waking Hebbian fast weights ($M_{fast}$) into permanent LoRA parameters via truncated SVD low-rank distillation and QR nullspace orthogonalization ($0.000000$ knowledge interference leakage).
6. **Sheaf-Theoretic Invariant Firewall (Organ 5)**: Sub-0.05ms ($42.5\ \mu\text{s}$) prefrontal executive filter enforcing Bounded Norm ($\|h\| \le \gamma$), Directional Stability, Dirichlet Vacuity ($u \ge 0.05, c \le 0.95$), and Code Execution Integrity ($\Delta_{test} = \emptyset$).

---

## Installation

```bash
# Core package (PyPI v3.2.0 - instant install, immune to Windows MAX_PATH limits)
pip install dual-loop-controller

# For NVIDIA GPU Acceleration (Recommended: installs PyTorch with CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "dual-loop-controller[llm]"

# With Custom Inference Server (vLLM / Ollama alternative)
pip install "dual-loop-controller[serve]"
```

### 🛠️ Hardware Diagnostic & Setup
```bash
hadl setup
# or: dual-loop setup
```

> **Windows Users & PyTorch `[Errno 2]` MAX_PATH Notice**:
> If Windows blocks PyTorch header file extraction with `[Errno 2] No such file or directory`:
> 1. Run `fix_windows_longpaths.bat` (or in Admin CMD: `reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d 1 /f`).
> 2. Or install inside a virtual environment: `python -m venv .venv` -> `.venv\Scripts\activate` -> `pip install dual-loop-controller`.
> 3. Or double-click the turnkey installer: `INSTALL_DUAL_LOOP.bat`.

---

## Quickstart

### 1. Universal Model Attachment (3 Lines of Code)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop import attach

# 1. Load any supported causal Transformer
model_id = "Qwen/Qwen2.5-7B-Instruct"  # or LLaMA-3, Mistral, Gemma, GLM-4
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 2. Attach Dual-Loop Controller (Zero retraining, 100% frozen base model)
model = attach(base_model, k_steps=2)

# 3. Standard text inference with latent deliberation (zero token bloat)
inputs = tokenizer("Question: In inverted buoyancy physics, denser objects float. Does lead or cork float?\nAnswer:", return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

### 2. High-Throughput OpenAI-Compatible Server

Launch an OpenAI-compatible API server with dynamic VRAM auto-tuning:

```bash
dual-loop serve --model Qwen/Qwen2.5-7B-Instruct --port 8000 --regime nf4
```

Or consume it via standard OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Explain active inference."}]
)
print(response.choices[0].message.content)
```

---

### 3. Running Offline Sleep-Phase Consolidation

```python
import torch
from dual_loop import SleepPhaseConsolidationEngine, SleepMemoryEpisode

engine = SleepPhaseConsolidationEngine(d_model=1024, lora_rank=16)

# Record waking episodes into sleep buffer
for _ in range(10):
    episode = SleepMemoryEpisode(
        pre_states=torch.randn(8, 1024),
        post_states=torch.randn(8, 1024),
        delta_loss=0.45,
        domain_tag="domain_algebra"
    )
    engine.record_episode(episode)

# Run offline sleep consolidation replay
report = engine.run_consolidation_cycle(svd_rank=16, orthogonalize_nullspace=True)
print("Consolidated Episodes :", report["episodes_processed"])
print("LoRA Rank             :", report["lora_rank"])
print("Nullspace Overlap     :", report["nullspace_leakage_overlap"])  # 0.000000
```

---

### 4. Command-Line Interface (CLI)

```bash
# Display environment and registered brain organ telemetry
hadl info

# Run standardized benchmark suite
hadl benchmark --suite multimodal

# Validate mathematical and structural consistency of benchmark logs
hadl validate-benchmark eval_results

# Execute single autonomous background contemplation cycle
hadl daemon-step --slots 16 --d-model 1024

# Execute offline sleep-phase consolidation replay and SVD distillation
hadl sleep-cycle --episodes 10 --rank 16

# Execute AST-hardened Popperian sandbox verification
hadl verify-sandbox "math.sqrt(16) + 2"
```

---

## Empirical Benchmark Highlights (NVIDIA RTX 5060 GPU)

All benchmarks are 100% reproducible and physically measured on an NVIDIA GeForce RTX 5060 Laptop GPU evaluating `Qwen/Qwen3.5-2B` (bfloat16):

* **SquareCloud Dynamic Reasoning Accuracy**: **66.7% (2/3)** (+100.0% relative improvement over unaugmented baseline 33.3%).
* **Fail-Safe Executive Veto**: 100% protection against catastrophic divergence via the 50% capacity Latent 1-Bit Judge ($v_{\text{gate}} = 0.0$ on ambiguous states).
* **Length-Preserving Isometry**: **Strictly 0.000000 isometry error** across all tokens and sequences ($\|h'\|_2 \equiv \|h\|_2$) via Unitary Givens rotations.
* **Knowledge Syringe Fact Injection**: Quasi-orthogonal concept binding via FFT circular convolution ($\langle \text{Syringe}, \text{Key} \rangle = -0.0163$, $\langle \text{Syringe}, \text{Val} \rangle = +0.0395$, $\|\text{Syringe}\| = 1.0000$).
* **Sub-Millisecond Forward Latency**: Overhead is **< 1.5 ms / forward pass**, delivering real-time 15.4–17.5 tok/s generation throughput.
* **Security & Audit Compliance**: 100% resolution of Issue #45 audit findings (zero-gradient fix, key->value recall, causal prefix isolation).

For full architecture diagrams, benchmarks, and interactive dashboards, visit the [GitHub Repository](https://github.com/Ch3nOff/dual-loop-controller).

---

## License

Licensed under the [MIT License](https://github.com/Ch3nOff/dual-loop-controller/blob/main/LICENSE).
