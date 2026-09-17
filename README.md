<p align="center">
  English | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_id.md">Bahasa Indonesia</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_zh.md">简体中文</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ja.md">日本語</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ko.md">한국어</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_es.md">Español</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_fr.md">Français</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_de.md">Deutsch</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ru.md">Русский</a> | <a href="https://github.com/Ch3nOff/dual-loop-controller/blob/main/docs/README_ar.md">العربية</a>
</p>

<h1 align="center">Dual-Loop Cognitive Controller</h1>
<h3 align="center">State-of-the-Art Latent Deliberation & Cognitive Reasoning Framework for Any Transformer Model</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python Versions"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Adapter%20Weights-yellow.svg" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/tests-73%20passed-brightgreen.svg" alt="Unit Tests"></a>
</p>

---

## What is Dual-Loop Cognitive Controller?

Standard autoregressive Transformers perform uniform $O(1)$ computation per token regardless of task complexity. While Chain-of-Thought (CoT) prompting enables multi-step reasoning, it consumes heavy output token bandwidth, creates severe serial latency, and exposes models to prompt distraction. Conversely, naive recurrent pondering suffers from **overthinking** (corrupting commonsense intuition) and **the unsupervised falsification trap** (second-guessing correct initial predictions).

**Dual-Loop Cognitive Controller** is a universal model-enhancement framework that equips **any Transformer architecture** with dual-process System 1 (intuitive) and System 2 (deliberative) reasoning:

* **Outer Loop (System 2 / Latent Deliberation)**: Executes recursive mental simulation in continuous latent space without emitting intermediate discrete tokens.
* **Inner Loop (System 1 / Generation)**: Decodes high-fidelity tokens conditioned on converged thought vectors.
* **Cognitive Matrix Helper (Tversky Elimination-by-Aspects)**: Screens candidate options in Bench 1, eliminates distractor *wrong logs*, and focuses deliberation strictly on surviving contenders in Bench 2.
* **Hippocampal Episodic Virtual Memory**: Stores verified reasoning traces as Settled Anchors, enabling instant ($<0.01\text{s}$) zero-compute shortcut recall.
* **Directional Safety Projection**: Mathematically shields confident predictions from degradation, guaranteeing **Zero Negative Drift**.

---

## Universal Compatibility: Works with Any Transformer

`dual-loop-controller` attaches seamlessly via non-invasive PyTorch forward hooks to any standard causal language model. No modifications to your underlying model weights are required:

| Model Family | Supported Architectures & Parameter Scales | Example Checkpoints |
| :--- | :--- | :--- |
| **Meta LLaMA** | LLaMA-2, LLaMA-3, LLaMA-3.1, LLaMA-3.2 (1B, 3B, 8B, 70B+) | `meta-llama/Meta-Llama-3-70B-Instruct`, `Llama-3.2-3B` |
| **Mistral AI** | Mistral-7B, Mixtral-8x7B, Mixtral-8x22B, Mistral Large | `mistralai/Mistral-7B-Instruct-v0.3`, `Mixtral-8x7B` |
| **Qwen** | Qwen-1.5, Qwen-2, Qwen-2.5, Qwen-3.5 (0.5B, 7B, 27B, 72B) | `Qwen/Qwen2.5-27B-Instruct`, `Qwen/Qwen2.5-72B`, `Qwen3.5-2B` |
| **Google Gemma** | Gemma, Gemma-2 (2B, 9B, 27B) | `google/gemma-2-27b-it`, `google/gemma-2-9b-it` |
| **DeepSeek** | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1-Distill (1.5B to 70B) | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` |
| **Microsoft Phi**| Phi-2, Phi-3, Phi-3.5 | `microsoft/Phi-3-mini-4k-instruct` |
| **Large Open-Weight (100B+)** | GPT-OSS-120B, Falcon-180B, DBRX, Bloom-176B | Any 70B–120B+ Transformer with multi-GPU sharding |

---

## Installation

Works with Python 3.9+ and [PyTorch](https://pytorch.org/get-started/locally/) 2.0+.

### With `pip`:
```bash
pip install dual-loop-controller
```

### With `uv`:
```bash
uv pip install dual-loop-controller
```

### Install with LLM dependencies (Transformers & Accelerate):
```bash
pip install "dual-loop-controller[llm]"
```

### Install from Source:
```bash
git clone https://github.com/Ch3nOff/dual-loop-controller.git
cd dual-loop-controller
pip install -e .
```

---

## Quickstart

### 1. Attach Dual-Loop to ANY Hugging Face Model in 3 Lines

You can attach the controller to **any** model family (`Llama`, `Mistral`, `Qwen`, `Gemma`, etc.) using the universal `attach_dual_loop` factory:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop import attach_dual_loop

# Step 1: Load your favorite Hugging Face model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # or "mistralai/Mistral-7B-v0.3", "Qwen/Qwen2.5-7B", "google/gemma-2-9b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Step 2: Attach Dual-Loop Cognitive Controller
# layer_idx defaults to the optimal midpoint layer automatically
model = attach_dual_loop(base_model, k_steps=2)

# Step 3: Run inference with latent System 2 pondering
prompt = "Question: Under an inverted buoyancy physics law, denser objects float. If lead and cork drop in water, which floats?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

# Model deliberates in latent space before generating output tokens
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

### 2. Multi-Choice Solving with 2-Bench Cognitive Matrix Helper

For challenging multiple-choice tasks (medical diagnosis, science QA, legal entailment), use `CognitiveMatrixHelper` to eliminate distractor options (*wrong logs*) and focus System 2 attention on surviving contenders:

```python
import numpy as np
from dual_loop import CognitiveMatrixHelper

# Initialize helper with adaptive distractor cutoff
matrix_helper = CognitiveMatrixHelper(elimination_threshold=0.12, min_survivors=2)

# Bench 1: Raw candidate scores from base model
scores_bench1 = [-9.1488, -9.2891, -9.5007, -11.0977, -10.9492]
labels = ["D", "E", "F", "A", "B"]

# Step 1: Populate Cognitive Evidence Matrix & prune distractors
matrix = matrix_helper.build_evidence_matrix(scores_bench1, labels=labels)
print("Pruned Distractor Logs :", matrix["eliminated_labels"])  # -> ['A', 'B'] (Noise eliminated)
print("Surviving Contenders    :", matrix["survivor_labels"])    # -> ['D', 'E', 'F'] (Viable dilemma)

# Bench 2: System 2 deliberates strictly on surviving candidates [D, E, F]
scores_delib_survivors = [-6.9465, -5.8747, -4.4858]

# Step 2: Fuse scores (eliminated distractors are locked to -infinity)
final_scores = matrix_helper.fuse_scores(
    scores_base=scores_bench1,
    scores_delib_survivors=scores_delib_survivors,
    survivor_indices=matrix["survivors"],
    lambda_delib=0.85
)

best_idx = np.argmax(final_scores)
print("Final Rescued Decision :", labels[best_idx])  # -> 'F' (Correct Answer!)
```

---

### 3. Accelerated Reasoning with Hippocampal Virtual Memory

Enable human-like memory consolidation where familiar queries bypass deliberation with **instant $<0.01\text{s}$ retrieval (3,146x speedup)**:

```python
import torch
from dual_loop import CognitiveWorkingMemory
from dual_loop.memory import EpisodicMemoryBuffer

# Initialize continuous key-value memory bank
memory = EpisodicMemoryBuffer(d_model=2048, capacity=512, sim_threshold=0.95)

# Store verified reasoning trace
query_vector = torch.randn(1, 2048)
thought_vector = torch.randn(1, 2048)

memory.store(
    key=query_vector,
    thought=thought_vector,
    margin=0.45,
    meta={"answer": "F", "task": "colored_objects"},
    is_settled=True
)

# Recall instantly on subsequent encounters (Zero FLOPs, Zero Token Waste)
match = memory.recall_settled(query_vector, sim_threshold=0.95)
if match:
    print("Instant Memory Recall:", match["metadata"]["answer"])
```

---

### 4. Scaling to Large Models (27B, 70B, 120B+) with Multi-GPU & 4-bit Quantization

On large models (such as **Qwen 27B**, **LLaMA-3 70B**, or **120B+ open-weight models**), standard Chain-of-Thought (CoT) prompting generates 1,000–3,000 discrete tokens, incurring 30–60 seconds of latency and massive KV-cache VRAM consumption. 

**Dual-Loop Controller** solves this by performing System 2 deliberation in continuous latent space ($D=5120\dots 10240$), finishing in milliseconds with zero output token bloat. It is fully compatible with **Multi-GPU Sharding (`device_map="auto"`)** and **BitsAndBytes 4-bit / 8-bit Quantization**:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dual_loop import attach_dual_loop

# Example: Running on Qwen-27B, LLaMA-70B, or 120B model with 4-bit quantization
model_id = "Qwen/Qwen2.5-27B-Instruct"  # or "meta-llama/Meta-Llama-3-70B-Instruct"

# 1. Configure 4-bit NF4 quantization to fit on consumer/prosumer GPUs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"  # Automatically shards across GPU 0, 1, etc.
)

# 2. Attach Dual-Loop Controller (Automatically matches layer device and precision)
model = attach_dual_loop(base_model, k_steps=2)

# 3. High-efficiency inference without CoT token overhead
prompt = "Question: Analyze the fault tolerance of this distributed Byzantine consensus protocol:\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

# Model deliberates in latent vectors (SRAM cache) rather than emitting 1000s of CoT tokens
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Why Should I Use Dual-Loop Controller?

* **Universal Compatibility**: Works with Llama, Mistral, Qwen, Gemma, DeepSeek, and any causal LM.
* **Zero Output Token Waste**: Deliberates in continuous latent thought space instead of generating hundreds of CoT scratchpad tokens.
* **Distractor Elimination (Amos Tversky EBA)**: Solves multi-choice attention dilution by filtering superficial distractor options.
* **Zero Negative Drift Guarantee**: Directional Safety Projection ensures confident intuitive answers are never degraded.
* **Hardware-Aligned & Cache-Friendly**: Cognitive Working Memory (CWM) fits directly inside GPU SRAM / L2 cache, eliminating redundant KV-cache lookups.
* **100% Offline & Private**: Runs entirely locally on your machine or server. Zero external API bills, zero data leakage.

---

## When Shouldn't I Use Dual-Loop Controller?

* **Pure Embedding Models**: Dual-Loop is designed for generative causal autoregressive decoders, not encoder-only models (like BERT) without generation heads.
* **Ultra-Low Latency Sub-5ms Audio Streams**: Latent pondering adds a small computational budget ($K$ iterations) at an intermediate layer, suited for high-accuracy reasoning rather than hard real-time streaming audio.

---

## Benchmark Suite & Empirical Research

For comprehensive benchmarks (including ARC-Challenge, Big-Bench Hard, 20-Task Macro Suites, and procedural stress tests), please consult [`BENCHMARKS.md`](BENCHMARKS.md) and [`eval_results/`](eval_results/).

---

## Citation

If you use `dual-loop-controller` in your research or production systems, please cite:

```bibtex
@software{chen2026dualloop,
  author = {Matthew Chen and Contributors},
  title = {Dual-Loop Cognitive Controller: Hardware-Aligned Latent Deliberation & Memory Architecture for Transformers},
  year = {2026},
  publisher = {PyPI},
  version = {2.2.1},
  url = {https://github.com/Ch3nOff/dual-loop-controller}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
