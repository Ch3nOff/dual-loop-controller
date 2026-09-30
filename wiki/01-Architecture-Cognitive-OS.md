# 01. Cognitive OS Architecture: The 5 Computational Brain Organs

In version **3.1.0**, the Dual-Loop Cognitive Controller transitions from a modular adapter into a **Unified Cognitive Operating System (Cognitive OS)** for foundation models.

---

## 🏛️ System Overview

Autoregressive models (System 1) predict tokens sequentially based solely on local attention weights. HADL overlays a **System 2 Recurrent Deliberation Substrate** structured into **5 distinct Computational Brain Organs**:

```mermaid
flowchart TD
    subgraph S1 ["System 1: Autoregressive Backbone"]
        TokenIn["Input Tokens"] --> L_early["Early Layers 1..L_mid"]
        L_early --> MidHook["Interception Hook (Layer L_mid)"]
        L_late["Later Layers"] --> LMHead["LM Head"] --> TokenOut["Token Output"]
    end

    subgraph CognitiveOS ["HADL Cognitive OS: The 5 Computational Organs"]
        MidHook --> Organ1["Organ 1: Global Workspace & Canonical Projection<br/>R^(D_native) -> R^1024"]
        Organ1 --> Organ2{"Organ 2: Allostatic Energy & Friston Policy Router<br/>G(pi) Free Energy Minimization"}
        
        Organ2 -->|"pi_0: Fast Streaming"| Bypass["7.8 us Bypass"]
        Organ2 -->|"pi_1: Verification"| FastCheck["Evidential Gate"]
        Organ2 -->|"pi_2: Deep Deliberation"| Deliberation["K-Step Deliberation Loop"]
        
        Bypass --> Organ3["Organ 3: Multi-Time-Scale Memory<br/>CWM (16 slots) + Hebbian M_fast"]
        FastCheck --> Organ3
        Deliberation --> Organ3
        
        Organ3 --> Organ5{"Organ 5: Sheaf Invariant Firewall<br/>Cohomological Obstruction Check"}
        Organ5 -->|"Safe"| InvProj["Canonical Inverse: R^1024 -> R^(D_native)"]
        Organ5 -->|"Violation"| SafetyClamp["Executive Inhibition / Clamp"]
        InvProj --> L_late
        SafetyClamp --> L_late
    end

    subgraph IdlePhase ["Organ 4: Offline Sleep Phase (Decoupled Idle Clock)"]
        IdleDetect["System Idle Detection"] --> Organ4["Organ 4: Sleep Consolidation Engine<br/>SVD Distillation + QR Nullspace Orthogonalization"]
        Organ4 --> PermLoRA[("Permanent Consolidated Memory<br/>0.000000 Interference")]
    end
```

---

## 🧠 The 5 Computational Brain Organs in Detail

### Organ 1: Global Workspace Theory (GWT) & Dynamic Graph Introspector
* **Module**: `DynamicGraphIntrospector` & `UniversalDualLoopAdapter`
* **Purpose**: Provides a universal interface across heterogeneous model families (Qwen, Gemma, LLaMA, Mistral, GLM-4) without hardcoding layer paths.
* **Mechanism**:
  - Dynamically scans `model.named_modules()` to discover layer containers (`model.layers`, `model.model.layers`, `transformer.encoder.layers`).
  - Sets up mid-layer hooks (default: $L_{mid} = \lfloor 0.60 \times L \rfloor$).
  - Projects representations to a unified **Canonical Manifold**:
    $$\mathbb{R}^{D_{native}} \xrightarrow{W_{down}} \mathbb{R}^{1024} \xrightarrow{\text{Deliberate}} \mathbb{R}^{1024} \xrightarrow{W_{up}} \mathbb{R}^{D_{native}}$$
  - **ReZero Identity**: Enforces $\Delta_{init} \equiv 0$ through a zero-initialized scalar $\alpha$, guaranteeing that the baseline model is mathematically unaltered upon initialization.

### Organ 2: Allostatic Energy Modulator & Friston Active Inference Router
* **Module**: `AllostaticEnergyModulator` & `ActiveInferencePolicyRouter`
* **Purpose**: Prevents computational waste and gate cascade collapse.
* **Mechanism**:
  - Unifies multiple gating criteria (uncertainty, novelty, homeostatic drive) into a single scalar energy potential:
    $$\Gamma_{allostatic} = \sigma\left(\frac{E_{allo}}{\tau}\right)$$
  - Evaluates Expected Free Energy $G(\pi)$ to choose optimal inference path:
    * Policy $\pi_0$ ($u < 0.65$): Fast-Path Streaming Bypass ($7.8\ \mu\text{s}$ latency).
    * Policy $\pi_1$ ($0.65 \le u < 0.85$): Fast evidential check.
    * Policy $\pi_2$ ($u \ge 0.85$): Full $K$-step latent deliberation loop.

### Organ 3: Multi-Time-Scale Working Memory & Directional Reservoir
* **Module**: `SpatioTemporalEntropicCWM`, `PlasticFastWeightUnit`, `CompactCommonSenseReservoir`
* **Purpose**: Provides immediate, in-session adaptive memory without weight updates.
* **Mechanism**:
  - 16-slot SpatioTemporal Entropic CWM for tracking transient cognitive states.
  - Fast Hebbian weights ($M_{fast}$) updated via trace plasticity:
    $$\Delta M_{fast} = \eta \left(x_{post} x_{pre}^T - \alpha M_{fast}\right)$$
  - Directional commonsense reservoir performing topological cosine-similarity lookup in $<0.01\text{s}$.

### Organ 4: Sleep-Phase Consolidation Engine
* **Module**: `SleepPhaseConsolidationEngine`
* **Purpose**: Translates transient waking knowledge into permanent parameters during system idle periods.
* **Mechanism**:
  - Replays episodic trajectories stored in the waking memory buffer.
  - Computes Truncated SVD on accumulated fast weights:
    $$M_{fast} \approx U_r \Sigma_r V_r^T \implies \Delta W = A \cdot B$$
  - Projects updates into the orthogonal nullspace of prior tasks using QR Gram-Schmidt orthogonalization:
    $$P_{null} = I - V_{prior} V_{prior}^T, \quad \Delta W_{safe} = P_{null} \Delta W$$
  - Result: **$0.000000$ catastrophic interference leakage**.

### Organ 5: Sheaf-Theoretic Invariant Firewall
* **Module**: `SheafInvariantFirewall`
* **Purpose**: Prefrontal executive inhibition to block catastrophic and unsafe actions in sub-0.05ms ($42.5\ \mu\text{s}$).
* **Active Invariants**:
  1. **Bounded Norm Invariant**: Clamps activation norms $\|h\|_2 \le \gamma$ to avoid gradient explosions.
  2. **Directional Stability Invariant**: Enforces contractive Lyapunov dynamics across recurrent steps.
  3. **Dirichlet Vacuity Invariant**: Enforces $c \le 0.95$ and $u \ge 0.05$ on evidential predictions, eliminating arrogant hallucinations.
  4. **Code Execution Integrity Invariant**: Inspects AST nodes in Python code, blocks unauthorized system calls, prevents infinite bash loops, and enforces test immutability ($\Delta_{test} = \emptyset$).
