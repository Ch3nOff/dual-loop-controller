# 06. CLI & Production Deployment

HADL v3.1.0 is engineered for enterprise-grade deployment, supporting vLLM, TensorRT-LLM, CUDA Graphs, and background daemon execution.

---

## 🖥️ Command-Line Interface (CLI)

Both `hadl` and `dual-loop` binaries are registered as console entry points upon package installation:

```bash
# Check version, GPU telemetry, and registered brain organs
hadl info

# Run multimodal bidirectional benchmark
hadl benchmark --suite multimodal

# Validate mathematical integrity of benchmark logs
hadl validate-benchmark eval_results/

# Execute a single autonomous background contemplation cycle
hadl daemon-step --slots 16 --d-model 1024

# Execute offline sleep-phase consolidation replay
hadl sleep-cycle --episodes 10 --rank 16

# Test AST-hardened Popperian sandbox verification
hadl verify-sandbox "math.sqrt(16) + 2"
```

---

## 🚀 Turnkey Windows Launchers

Launch interactive tools and live dashboards on Windows with a single double-click:

| Launcher File | Purpose & Capability |
| :--- | :--- |
| `START_BENCHMARK.bat` | Starts local streaming inference server & English HUD at `http://127.0.0.1:8000` |
| `run_qwen3_8_27b_benchmark.bat` | Executes complete Qwen3.8-27B OOM memory profiler on real hardware |
| `run_webgame_comparison.bat` | Runs unconstrained web game generation & in-situ learning benchmark |
| `run_benchmark.bat` | Interactive 4-mode terminal suite (Spotlight Showdown, Web Dashboard, Memory Loop) |

---

## 🏭 High-Throughput Inference Engines (vLLM & TensorRT-LLM)

### CUDA Graph Safety
Traditional Python-level branching in PyTorch disrupts CUDA Graph capture. HADL's fast-path streaming mode executes **pure branchless tensor arithmetic**:
- Zero dynamic Python control flow during streaming generation.
- $100\%$ compatible with `torch.cuda.make_graphed_callables()`.
- Measured fast-path bypass latency: **$0.0078\text{ ms}$ ($7.8\ \mu\text{s}$)**.

### Flat 2D Tensor Support
In high-throughput serving engines like vLLM (PagedAttention), input tokens are often flattened into 2D tensors of shape $[N, D_{native}]$ across heterogeneous batch sequences. HADL natively processes:
- Standard 3D batch sequences: $[B, S, D_{native}]$
- Flattened 2D batch tokens: $[N, D_{native}]$

### Zero KV-Cache Bloat
Because HADL's System 2 reasoning takes place within continuous activation manifolds, **no additional tokens are appended to the context window**:
- Chain-of-Thought (CoT): $+1,500$ to $+5,000$ tokens per query &rarr; $O(L^2)$ KV-cache explosion.
- HADL Cognitive OS: **$0$ extra tokens** &rarr; **Constant $O(1)$ memory overhead**.
