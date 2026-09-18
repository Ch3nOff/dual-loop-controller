<p align="center">
  English | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_id.md">Bahasa Indonesia</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_zh.md">简体中文</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ja.md">日本語</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ko.md">한국어</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_es.md">Español</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_fr.md">Français</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_de.md">Deutsch</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ru.md">Русский</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ar.md">العربية</a>
</p>

<h1 align="center">Dual-Loop Cognitive Controller</h1>
<h3 align="center">Hardware-Aligned Latent Deliberation, Context Directional Routing & Memory Architecture for Any Transformer</h3>

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

**Dual-Loop Cognitive Controller** is a universal framework that equips standard autoregressive Transformers with dual-process **System 1 (fast, intuitive)** and **System 2 (deliberative)** cognitive capabilities.

Instead of generating hundreds or thousands of expensive Chain-of-Thought (CoT) text tokens, Dual-Loop deliberates recursively in **continuous latent vector space** ($D=2048\dots 10240$) inside GPU SRAM/L2 cache:

* **Zero Output Token Waste**: Millisecond latent deliberation without KV-cache explosion or context bloat (0 extra text tokens).
* **Context Directional Bipolar Router**: Projects tasks into a directional manifold ($\rho_{\text{direction}}$): Scientific inquiry routes upwards to deep System 2 deliberation, while everyday reality routes downwards to common-sense grounding.
* **Compact Common-Sense Reservoir ($f \circ g$)**: Stores foundational physical reality axioms in a micro-prototype matrix ($< 50\text{ KB}$ in RAM), eliminating associative overthinking.
* **Probabilistic Soft Belief Revision & 2x-Think Gating**: Replaces brittle hard-locks with soft penalties, enabling adaptive belief updates upon overwhelming deliberative evidence ($76.00\%$ Macro Accuracy on standard N=75 suite).
* **Zero Negative Drift**: Directional Safety Projection ensures confident intuitive answers are never degraded.
* **Universal Compatibility**: Attaches to **any** causal Transformer (LLaMA, Mistral, Qwen, Gemma, DeepSeek, Phi) and scales from 1B to 120B+ models with multi-GPU sharding and 4-bit quantization.

> 📖 **Full Documentation, Empirical Scoreboards & Architectural Comparisons**:  
> For the complete benchmark report (75-item standard benchmark suite, token overload analysis, and system comparison graphs), please visit our **[GitHub Repository](https://github.com/Ch3nOff/dual-loop-controller)**.

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

Attach the controller to any standard Hugging Face model (`Llama`, `Mistral`, `Qwen`, `Gemma`, etc.):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop import attach_dual_loop

# 1. Load your model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # or "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 2. Attach Dual-Loop Controller (automatically attaches to optimal middle layer)
model = attach_dual_loop(base_model, k_steps=2)

# 3. Deliberative inference in latent space (Zero Extra Text Tokens)
prompt = "Question: In inverted buoyancy physics, denser objects float. Does lead or cork float?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

### 2. Directional Router & Probabilistic Cognitive Judge

```python
from dual_loop import ProbabilisticCognitiveJudge

# Initialize Cognitive Judge with Directional Manifold & Common-Sense Reservoir (f o g)
judge = ProbabilisticCognitiveJudge(
    cs_margin_threshold=0.35,
    base_lambda=0.85,
    intuitive_lambda=0.20,
    soft_penalty_weight=4.5,
    allow_belief_revision=True,
    use_directional_reservoir=True
)

prompt = "Which requires energy to move?"
choices = ["weasel", "willow", "mango", "poison ivy"]
labels = ["A", "B", "C", "D"]

scores_base = [-8.40759, -8.40907, -14.929, -5.713]
scores_delib = [-7.5420, -5.9615, -13.826, -5.317]

# Evaluates candidates with directional routing and soft belief revision
decision = judge.judge_and_fuse(
    scores_base=scores_base,
    scores_delib=scores_delib,
    labels=labels,
    banned_labels=["D"],  # Previously logged wrong choice
    prompt=prompt,
    choices=choices
)

print("Predicted Choice :", decision["pred_label"])   # -> 'A' (weasel - CORRECT)
print("Manifold Vector  :", decision["direction"])    # -> 'DOWN_COMMONSENSE'
print("Grounding Delta  :", decision["cs_deltas"])   # -> [+2.2, -0.8, -0.8, -0.8]
```

---

### 3. Large Models (27B, 70B, 120B+) with 4-Bit Quantization

Scale to massive models without 30–60 second CoT latency or VRAM exhaustion:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dual_loop import attach_dual_loop

# 4-bit NF4 quantization for large parameters
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "Qwen/Qwen2.5-27B-Instruct"  # or "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"  # Shards across available GPUs
)

# Automatically matches quantized layer device & precision
model = attach_dual_loop(base_model, k_steps=2)

inputs = tokenizer("Analyze Byzantine fault tolerance in decentralized state machines:\nAnswer:", return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Supported Architectures

| Family | Architectures | Scales |
| :--- | :--- | :--- |
| **Meta LLaMA** | LLaMA-2, LLaMA-3, LLaMA-3.1, LLaMA-3.2 | 1B, 3B, 8B, 70B+ |
| **Mistral AI** | Mistral-7B, Mixtral-8x7B, Mixtral-8x22B, Mistral Large | 7B to 8x22B |
| **Qwen** | Qwen-1.5, Qwen-2, Qwen-2.5, Qwen-3.5 | 0.5B, 7B, 27B, 72B |
| **Google Gemma** | Gemma, Gemma-2 | 2B, 9B, 27B |
| **DeepSeek** | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1-Distill | 1.5B to 70B |
| **Microsoft Phi** | Phi-2, Phi-3, Phi-3.5 | 3.8B to 14B |
| **Generic** | Any causal Hugging Face `PreTrainedModel` | Up to 120B+ |

---

## Links & Community

* **GitHub Repository**: [https://github.com/Ch3nOff/dual-loop-controller](https://github.com/Ch3nOff/dual-loop-controller)
* **Full Benchmark Suite & Empirical Graphs**: [https://github.com/Ch3nOff/dual-loop-controller#decisive-empirical-benchmark-n75-authentic-standard-benchmark-suite](https://github.com/Ch3nOff/dual-loop-controller)
* **Pretrained Weights**: [Hugging Face Hub](https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b)
* **Interactive Web Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo)
* **Bug Reports & Issues**: [GitHub Issues](https://github.com/Ch3nOff/dual-loop-controller/issues)

## License

MIT License. See [LICENSE](https://github.com/Ch3nOff/dual-loop-controller/blob/main/LICENSE) for details.
