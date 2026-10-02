# Welcome to the Dual-Loop Cognitive Controller Wiki

Welcome to the official documentation wiki for **Dual-Loop Cognitive Controller (HADL v3.1.0)** — an open-source, model-agnostic **Cognitive Operating System (Cognitive OS)** for Large Language Models (LLMs) and Vision-Language Models (VLMs).

---

## 🧭 Wiki Navigation

| Topic | Description | Link |
| :--- | :--- | :--- |
| **Cognitive OS Architecture** | Deep dive into the 5 Computational Brain Organs | [[Architecture: Cognitive OS|01-Architecture-Cognitive-OS]] |
| **Benchmarks & Validation** | HA-COGBENCH 5-Module Cognitive Suite, Web Dev Latency | [[Benchmarks & Validation|03-Benchmarks-and-Empirical-Validation]] |
| **Python SDK & API Reference** | Complete class, method, and function documentation | [[API Reference & SDK|04-API-Reference-and-SDK]] |
| **Security & Sandboxing** | Compliance matrix SEC-01 to SEC-11, AST sandboxing | [[Security & Sandboxing|05-Security-and-Sandboxing]] |
| **CLI & Production Deployment** | Command-line guide, Windows `.bat` launchers, vLLM & CUDA graphs | [[CLI & Production Deployment|06-CLI-and-Production-Deployment]] |
| **Mathematical Formulations** | Theorems, Lyapunov proofs, and active inference equations | [[Mathematical Formulations|07-Mathematical-Formulations]] |

---

## 🏛️ What is HADL?

**HADL (Hardware-Aligned Autopoietic Latent Deliberation)** bridges the gap between reactive autoregressive next-token prediction and autonomous human-grade cognition.

Unlike Chain-of-Thought (CoT) prompting which pollutes the context window with thousands of discrete scratchpad tokens, HADL operates **inside continuous latent manifolds**:
* **Zero Output Token Waste**: 0 extra tokens generated during System 2 reasoning.
* **Continuous State Space**: Deliberation occurs over latent vectors $\mathbb{R}^{D}$.
* **Zero Interference**: Memories are consolidated into orthogonal nullspaces ($P_{null} = I - V V^T$) guaranteeing $0.000000$ catastrophic forgetting.
* **Hardware Alignment**: Dynamic VRAM Auto-Tuning ensures seamless execution across BF16, INT8, and NF4 without OOM errors.

---

## 📦 Quick Installation

```bash
# Core package from PyPI
pip install dual-loop-controller==3.1.0

# With Hugging Face Transformers & Accelerate
pip install "dual-loop-controller[llm]==3.1.0"
```

---

## 💡 Quick Start in 3 Lines

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop import attach

# Load any model (Qwen, Gemma, LLaMA, Mistral, GLM-4)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Attach Dual-Loop Cognitive Controller
model = attach(base_model, k_steps=2, enable_allostatic_modulation=True)

# Generate with deliberative System 2 reasoning
output = model.generate(**inputs, max_new_tokens=64)
```

---

## 🤝 Community & Support

- **GitHub Repository**: [Ch3nOff/dual-loop-controller](https://github.com/Ch3nOff/dual-loop-controller)
- **PyPI Package**: [dual-loop-controller on PyPI](https://pypi.org/project/dual-loop-controller/)
- **Live Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo)
- **Issue Tracker**: [GitHub Issues](https://github.com/Ch3nOff/dual-loop-controller/issues)
