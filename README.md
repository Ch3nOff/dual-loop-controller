# Dual-Loop Cognitive Controller v2.0
> **A Hardware-Aligned Latent Deliberation Framework for Transformers: Architecture & Empirical Analysis**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-empirical--audit-orange.svg)](#empirical-findings)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Standard Autoregressive Transformers perform uniform $O(1)$ layer computation per token regardless of task complexity. While Chain-of-Thought (CoT) prompting allows multi-step reasoning, it expends significant output token bandwidth and introduces serial generation latency.

The **Dual-Loop Cognitive Controller** investigates decoupling deliberation from token generation into two loops:
1. **Outer Loop (Executive Deliberation / System 2)**: Runs recursive state transitions in a continuous latent space without emitting intermediate tokens.
2. **Inner Loop (Language Generation / System 1)**: Reads the matured latent thoughts ($H_{\text{thought}}$) as a soft prefix to decode final text responses.

---

## Empirical Findings & Negative Results (The Unvarnished Truth)

To maintain strict scientific integrity, this repository reports **the actual, measured behavior of the model trained end-to-end (225,959 parameters, 35 epochs, 3,500 samples, 16 nodes, chance baseline = 6.25%)**, rather than idealized projections.

### 1. The Model Learns Real Relational Signals
* **Final Test Accuracy (3-Hop Graph Reasoning)**: **29.4%** vs. random chance **6.25%** (~4.7x better than random guessing).
* This confirms that the weight-tied recurrent Transformer and CWM buffer are capable of gradient propagation and multi-step pattern learning.

### 2. The Absence of Monotonic Test-Time Compute Scaling
A central theoretical hypothesis of recurrent latent pondering is that increasing inference steps ($K$) will progressively improve answer accuracy. **On this 225K parameter implementation, this claim does not hold**:

```text
========================================================================================
EMPIRICAL TEST-TIME COMPUTE EVALUATION (Checkpoint: checkpoint_trained_dualloop.pt)
========================================================================================
Ponder Steps (K) | Test Accuracy (500 samples) | Mean Predictive Entropy (nats)
----------------------------------------------------------------------------------------
K = 0 (No Ponder)| 27.4% - 30.6%               | 1.332 - 1.362 nats
K = 1            | 28.2%                       | 1.370 nats
K = 2            | 30.6%                       | 1.307 nats
K = 3 (Trained)  | 30.4%                       | 1.268 nats
K = 4            | 30.0%                       | 1.268 nats
K = 5            | 31.6%                       | 1.275 nats
========================================================================================
```

**Scientific Diagnosis**:
* **Flat/Noisy Trajectory**: $K=0$ (bypassing the Outer Loop entirely) performs at parity with or slightly exceeds intermediate $K$ values.
* **Representational Drift**: Tracing individual predictions step-by-step reveals that while some cases improve with pondering, others degrade (e.g. correct at $K=0..1$, but diverging to incorrect candidates at $K=2..3$ due to distractor pull).
* **Scale Artifact vs. Fundamental Limit**: At 225K parameters, the latent space lacks the geometric capacity to preserve stable multi-step deductions without explicit discrete token anchors. Pondering without token-level supervision introduces noise as much as refinement.

### 3. Degradation Under Context Distractors (Stress Test)
When distractor edge count increases on 3-hop graphs, performance decays steadily:
* **6 Edges**: 31.0%
* **8 Edges**: 21.0%
* **12 Edges**: 13.7%
* **16 Edges**: 10.3%

### 4. Dynamic Halting Requires Empirical Calibration
* The default `entropy_threshold = 0.5` nats from naive theory is **miscalibrated**: the model's actual predictive entropy operates in the **1.25 – 1.40 nats** regime.
* With threshold = 0.5, dynamic halting acts as a static loop using $K=K_{\max}$ for 100% of samples.
* To make dynamic halting functional, the `EntropyHaltingUnit` now includes `calibrate_threshold(percentile=50.0)` to tune the cutoff against the model's actual validation distribution.

---

## Architectural Implementation

Despite the scaling limits at small model regimes, the repository provides clean, production-grade PyTorch implementations of the core modules:

* **Cognitive Working Memory (`dual_loop/memory.py`)**: Compresses context into $M \ll N$ slots in GPU SRAM/L2 cache to avoid HBM memory bandwidth roundtrips.
* **Top-K Capacity Routing (`dual_loop/controller.py`)**: Enforces static tensor shapes $[B, K_{\text{cap}}, D]$ to eliminate CUDA warp divergence (MoD-style).
* **Calibrated Entropy Halting (`dual_loop/halting.py`)**: Adaptive stopping based on predictive uncertainty and convergence delta.
* **Latent Deliberation Adapter (`dual_loop/adapters/latent_adapter.py`)**: A plug-and-play mid-network adapter for pretrained LLMs (e.g., Llama, Qwen).

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/Ch3nOff/dual-loop-controller.git
cd dual-loop-controller
python -m pip install torch numpy
```

### 2. Running Component Tests (Verifying Shapes & Gradients)
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Running the Honest Benchmark Suite (Live Tensor Computations)
```bash
python -m dual_loop.benchmarks.comprehensive_suite
```

### 4. Re-Training from Scratch
```bash
python train.py --epochs 35 --hops 3 --k_steps 3 --d_model 64
```

For the complete technical paper and theoretical post-mortem, see [WHITEPAPER.md](WHITEPAPER.md).
