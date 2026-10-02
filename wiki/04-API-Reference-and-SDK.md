# 04. Python SDK & API Reference

HADL v3.1.0 exposes a clean, modular Python API that integrates seamlessly with PyTorch and Hugging Face Transformers.

---

## 📦 Core Exports

```python
from dual_loop import (
    attach,                           # Universal Model Attachment (Canonical Adapter)
    attach_dual_loop,                 # Alias to attach
    DynamicGraphIntrospector,         # Layer Topology Introspection Engine
    UniversalDualLoopAdapter,         # Canonical Latent Adapter (D_native -> 1024 -> D_native)
    SleepPhaseConsolidationEngine,    # Offline Sleep Memory Consolidation
    SleepMemoryEpisode,               # Data container for waking memory trajectories
    SheafInvariantFirewall,           # Sub-0.05ms Prefrontal Invariant Gate
    PopperianSelfPlayEngine,          # AST-Hardened Hypothesis Verification Sandbox
    AutonomousDaemonController,       # Background Idle Contemplation Daemon
)
```

---

## 🔧 Function & Class Reference

### 1. `attach(model, ...)` / `attach_dual_loop(model, ...)`
Attaches the Dual-Loop Cognitive Controller to any supported causal Transformer model.

```python
def attach(
    base_model: torch.nn.Module,
    k_steps: int = 2,
    canonical_dim: int = 1024,
    enable_allostatic_modulation: bool = True,
    enable_brain_sandbox: bool = True,
    enable_cross_modal: bool = False,
    mid_layer_ratio: float = 0.60
) -> UniversalDualLoopModelWrapper:
```

- **Parameters**:
  * `base_model` (`nn.Module`): Frozen autoregressive model.
  * `k_steps` (`int`): Maximum recurrent deliberation hops (default: `2`).
  * `canonical_dim` (`int`): Latent space dimension (default: `1024`).
  * `enable_allostatic_modulation` (`bool`): Active inference energy routing (default: `True`).
  * `enable_brain_sandbox` (`bool`): Enables System 2 deliberation (default: `True`).
  * `enable_cross_modal` (`bool`): Enables bidirectional image/audio translation (default: `False`).
  * `mid_layer_ratio` (`float`): Layer depth fraction for deliberation hook (default: `0.60`).
- **Returns**: `UniversalDualLoopModelWrapper` wrapping the base model with `.generate()` and standard forward pass.

---

### 2. `SleepPhaseConsolidationEngine`
Performs offline memory distillation and QR nullspace orthogonalization.

```python
class SleepPhaseConsolidationEngine:
    def __init__(self, d_model: int = 1024, lora_rank: int = 16): ...
    
    def record_episode(self, episode: SleepMemoryEpisode) -> None:
        """Stores a waking trajectory into the offline consolidation buffer."""
        
    def run_consolidation_cycle(
        self,
        svd_rank: int = 16,
        orthogonalize_nullspace: bool = True
    ) -> Dict[str, Any]:
        """
        Executes SVD distillation and projects weights into the nullspace of prior tasks.
        Guarantees nullspace_leakage_overlap == 0.000000.
        """
```

---

### 4. `SheafInvariantFirewall`
Sub-0.05ms prefrontal executive inhibitor.

```python
class SheafInvariantFirewall(nn.Module):
    def __init__(
        self,
        max_norm: float = 50.0,
        max_confidence: float = 0.95,
        min_vacuity: float = 0.05
    ): ...
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        action_candidate: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns (filtered_hidden_states, firewall_telemetry).
        Clamps explosive norms, enforces Dirichlet vacuity, and blocks unsafe actions.
        """
```
