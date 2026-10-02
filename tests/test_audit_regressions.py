"""
Regression Test Suite for Audit Findings in Issue #45 (v3.1.1 Audit)
===================================================================
Covers:
1. Universal Adapter Identity Preservation & Non-Zero Gradient Flow
2. Sleep-Phase Engine Associative Memory Key-Value Recall & Fresh State Reload
3. CWM Causal Prefix Isolation & Suffix Invariance
4. Benchmark Zero-Output Ablation Sensitivity
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch

import sys
sys.path.append('.')

from dual_loop import (
    UniversalDualLoopAdapter,
    SleepPhaseConsolidationEngine,
)
from dual_loop.adapters.latent_adapter import LatentDeliberationAdapter
from scripts import run_ha_cogbench as bench


def test_universal_adapter_identity_and_nonzero_gradient():
    """
    Issue 1: Default universal adapter must preserve base-model identity
    while ensuring non-zero gradient flow back from task loss.
    """
    torch.manual_seed(42)
    adapter = UniversalDualLoopAdapter(d_native=64).eval()
    
    x = torch.randn(2, 8, 64)
    target = torch.randn_like(x)
    
    # Check identity preservation across multiple K steps
    for k in (0, 1, 3):
        with torch.no_grad():
            y, telem = adapter(x, k_steps=k)
            diff = (y - x).abs().max().item()
            assert diff == 0.0, f"Identity violated at k={k}, max diff: {diff}"
            assert telem["delta_norm"] == 0.0
            
    # Verify non-zero gradient flows to ReZero alpha on backpropagation
    adapter.train()
    out, _ = adapter(x, k_steps=3)
    loss = F.mse_loss(out, target)
    loss.backward()
    
    assert adapter.alpha.grad is not None, "alpha.grad is None"
    alpha_grad = adapter.alpha.grad.abs().max().item()
    assert alpha_grad > 0.0, f"alpha gradient must be non-zero, got {alpha_grad}"


def test_sleep_engine_key_value_recall_and_fresh_reload():
    """
    Issue 5: Sleep engine must correctly map key -> value (not reverse),
    and support reloading saved state_dict into a fresh instance.
    """
    d_model = 32
    rank = 4
    engine = SleepPhaseConsolidationEngine(d_model=d_model, rank=rank)
    
    key = torch.zeros(1, d_model)
    value = torch.zeros(1, d_model)
    key[0, 0] = 1.0
    value[0, 1] = 1.0
    
    engine.record_episode(key, value)
    telem = engine.consolidate()
    assert telem["status"] == "CONSOLIDATED_SUCCESS"
    
    # 1. Check Key -> Value Recall
    recalled = engine(key)
    cos_sim = F.cosine_similarity(recalled, value).item()
    assert cos_sim > 0.95, f"Expected key-to-value cosine > 0.95, got {cos_sim}"
    
    # 2. Check Fresh Instance Reload
    fresh = SleepPhaseConsolidationEngine(d_model=d_model, rank=rank)
    # This must not raise size mismatch for historical_basis
    fresh.load_state_dict(engine.state_dict())
    
    recalled_fresh = fresh(key)
    fresh_sim = F.cosine_similarity(recalled_fresh, value).item()
    assert fresh_sim > 0.95, f"Fresh reload failed recall check, sim={fresh_sim}"


def test_causal_prefix_isolation_suffix_invariance():
    """
    Issue 6: Suffix tokens in a candidate sequence must NOT leak into
    the prefix query anchor representations (strict causal isolation).
    """
    torch.manual_seed(42)
    adapter = LatentDeliberationAdapter(
        d_model=64,
        n_heads=4,
        num_thought_tokens=2,
        max_ponder_steps=2,
        enable_plasticity=False,
        use_evidential_gate=False,
        use_open_concept=False,
        use_surprise_gate=False,
        use_contrastive_evidence=False
    ).eval()
    
    prefix_len = 16
    suffix_len = 24
    
    prefix = torch.randn(1, prefix_len, 64)
    # Generate two sequences with the exact same prefix but totally different suffixes
    seq1 = torch.cat([prefix, torch.randn(1, suffix_len, 64)], dim=1)
    seq2 = torch.cat([prefix, torch.randn(1, suffix_len, 64)], dim=1)
    
    query_anchor = prefix_len - 1
    with torch.no_grad():
        out1, _ = adapter(seq1, query_idx=query_anchor, k_steps=2)
        out2, _ = adapter(seq2, query_idx=query_anchor, k_steps=2)
        
    diff = (out1[:, query_anchor, :] - out2[:, query_anchor, :]).abs().max().item()
    assert diff == 0.0, f"Future suffix tokens leaked into prefix anchor! Diff: {diff}"


def test_benchmark_zero_output_ablation_dependence():
    """
    Issue 2: Zero-output ablations must reduce benchmark scores to zero,
    proving benchmark scores depend on actual module computation.
    """
    device = torch.device("cpu")
    
    # Test Context Flood with zero CWM
    def zero_cwm(self, x, *args, **kwargs):
        return torch.zeros(x.shape[0], 16, x.shape[-1], device=x.device)
        
    with patch.object(bench.CognitiveWorkingMemory, "forward", zero_cwm):
        ablated_cf, _ = bench.run_modul_3_context_flood(device)
        assert ablated_cf["hadl_constraint_retention_accuracy_pct"] == 0.0, (
            "Ablated CWM must produce 0.0% CRA"
        )
        
    # Test Overnight Awakening with zero sleep output
    with patch.object(bench.SleepPhaseConsolidationEngine, "forward", lambda self, x: torch.zeros_like(x)):
        summary, _ = bench.run_modul_5_overnight_awakening(device)
        assert summary["hadl_zero_shot_sessional_recall_zsr_pct"] == 0.0, (
            "Ablated sleep engine must produce 0.0% ZSR"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING AUDIT REGRESSION TEST SUITE")
    print("=" * 70)
    
    print("\n[1/4] Testing Universal Adapter Identity & Nonzero Gradient...")
    test_universal_adapter_identity_and_nonzero_gradient()
    print("  -> PASSED!")
    
    print("\n[2/4] Testing Sleep Engine Recall & Checkpoint Reload...")
    test_sleep_engine_key_value_recall_and_fresh_reload()
    print("  -> PASSED!")
    
    print("\n[3/4] Testing CWM Causal Prefix Isolation & Suffix Invariance...")
    test_causal_prefix_isolation_suffix_invariance()
    print("  -> PASSED!")
    
    print("\n[4/4] Testing Benchmark Zero-Output Ablation Sensitivity...")
    test_benchmark_zero_output_ablation_dependence()
    print("  -> PASSED!")
    
    print("\n" + "=" * 70)
    print("ALL 4 AUDIT REGRESSION TESTS PASSED PERFECTLY!")
    print("=" * 70)
