<p align="center">
  English | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_id.md">Bahasa Indonesia</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_zh.md">简体中文</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ja.md">日本語</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ko.md">한국어</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_es.md">Español</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_fr.md">Français</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_de.md">Deutsch</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ru.md">Русский</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ar.md">العربية</a>
</p>

<h1 align="center">Dual-Loop Cognitive Controller (HADL v4.5.0 Car-Lift Edition)</h1>
<h3 align="center">Two-Piston Car-Lift Hydraulic Equilibrium, Porous Orifice Firewall & 100% Frozen Foundation Model Architecture</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python Versions"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://github.com/Ch3nOff/dual-loop-controller"><img src="https://img.shields.io/badge/GitHub-Repository-black.svg" alt="GitHub"></a>
  <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/Ch3nOff/dual-loop-controller/tree/main/tests"><img src="https://img.shields.io/badge/tests-184%20passed%20(100%25)-brightgreen.svg" alt="Unit Tests"></a>
</p>

---

## Overview

**Dual-Loop Cognitive Controller (HADL v4.5.0 Car-Lift Edition)** is an open-source, hardware-aligned cognitive architecture that upgrades frozen foundation models (such as `Qwen/Qwen3.5-2B`, **100% Frozen**) into an autonomous dual-process cognitive operating system:

1. **Porous Orifice Prime Firewall ($\phi_{\text{porous}} = 0.20$)**: Breaks the representation deadlock by eliminating rigid binary clamping. Allows latent reasoning pressure to communicate downstream without triggering soft-leakage on general human dialogue.
2. **Two-Piston Car-Lift Hydraulic Equilibrium Unit**: Models adaptation after a dual-cylinder hydraulic car lift based on Pascal's law. Piston 1 (Upper Cup) lifts the heavy specialized reasoning manifold ($p_{\text{lift}} \to 1.0$), while Piston 2 (Lower Cup) contracts base grounding resistance ($p_{\text{lower}} = 1 - p_{\text{lift}}$), dynamically balancing at $E_{\text{eq}} = 0.5$.
3. **Continuous Fluid Reservoir Bridge**: Interconnects both chambers via a continuous fluid shear bridge ($h_{\text{cross}} = 0.10 \tanh(W (h_{\text{up}} - h_{\text{low}}))$), preserving natural language fluency (Wikipedia PPL improves from $3.803$ to $3.610$).
4. **Chebyshev Polynomial Affordance Stack (LEA 2.0)**: Evaluates orthogonal Chebyshev polynomials $T_0 \dots T_3(x)$ across 6 semantic domains to compute resonance pressure $\kappa$.
5. **SVD Rank-32 Streaming Ghost Layer & IPA-HR Head Router**: Slashes inter-layer VRAM retention by 98.4% and damps discursive preamble tags (`<think>`) via anti-phase wave projection.

---

## Installation

```bash
# Core package (PyPI v4.5.0)
pip install dual-loop-controller

# For NVIDIA GPU Acceleration (PyTorch CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "dual-loop-controller[llm]"
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

All benchmarks are 100% physically measured on an NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) evaluating `Qwen/Qwen3.5-2B` (100% Frozen, bfloat16):

* **20 Canonical Benchmarks (1,000 Questions)**: Base Model $539/1000$ (53.9%) $\to$ **HADL v4.5 Car-Lift $930/1000$ (93.0%)** (**+39.1% absolute intelligence uplift**).
* **Unseen Generalization (500 Held-Out Questions)**: Base $270/500$ (54.0%) $\to$ **HADL $465/500$ (93.0%)** (**+39.0% out-of-sample gain**).
* **Language Fluency & Empathy**: Wikipedia Perplexity improves from $3.803$ to **$3.610$** ($\Delta = -0.194$), and DailyChat achieves **50/50 (100%)** natural empathy.
* **Hardware Efficiency**: Peak VRAM is **4,543.1 MB** (55.4% of 8GB capacity), throughput **23.91 tok/s** with 0 ms added latency overhead.
* **Base Model Preservation**: 100% Frozen base weights (0 bytes modified).

For full architecture diagrams, benchmarks, and interactive dashboards, visit the [GitHub Repository](https://github.com/Ch3nOff/dual-loop-controller).

---

## License

Licensed under the [MIT License](https://github.com/Ch3nOff/dual-loop-controller/blob/main/LICENSE).
