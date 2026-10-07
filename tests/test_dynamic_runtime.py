"""
Unit Tests for HADL Dynamic Recurrent Runtime Engine & Context Threads
======================================================================
"""

import unittest
import torch
import torch.nn as nn

from dual_loop.runtime.context_threads import LatentContextThreadEngine, DualStoreMemory
from dual_loop.runtime.dynamic_runtime import DynamicHydraulicPiston, HADLDynamicRuntime


class DummyBlock(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        return (self.linear(x),)


class DummyModel(nn.Module):
    def __init__(self, num_layers=6, d_model=64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyBlock(d_model) for _ in range(num_layers)])
        
    def forward(self, x):
        h = x
        for layer in self.model.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return h


class TestDynamicRuntime(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.B = 2
        self.L = 32
        
    def test_latent_context_thread_engine(self):
        engine = LatentContextThreadEngine(d_model=self.d_model, chunk_size=8, expansion_factor=2)
        x = torch.randn(self.B, self.L, self.d_model)
        
        threads, residual = engine.chunk_and_compress(x)
        expected_chunks = self.L // 8
        
        self.assertEqual(threads.shape, (self.B, expected_chunks, self.d_model))
        self.assertEqual(residual.shape, (self.B, self.L % 8, self.d_model))
        
    def test_dual_store_memory(self):
        memory = DualStoreMemory(d_model=self.d_model, short_term_capacity=32, rank=8, base_lr=0.1, base_decay=0.05)
        h = torch.randn(self.B, 10, self.d_model)
        
        delta, norm1 = memory(query=h, current_state=h)
        self.assertEqual(delta.shape, (self.B, 10, self.d_model))
        self.assertGreater(norm1.item(), 0.0)
        
        # Test adaptive decay over long horizon
        for _ in range(10):
            delta, norm_seq = memory(query=h, current_state=h)
            
        # Memory norm must remain bounded (no eigenvalue explosion)
        self.assertLess(norm_seq.item(), 50.0)
        
        # Test reset
        memory.reset_state()
        self.assertIsNone(memory.m_long)
        self.assertEqual(memory.step_counter, 0)

    def test_dynamic_hydraulic_piston(self):
        piston = DynamicHydraulicPiston(d_model=self.d_model, porous_factor=0.20)
        h = torch.randn(self.B, 10, self.d_model)
        
        delta, p = piston(h)
        self.assertEqual(delta.shape, (self.B, 10, self.d_model))
        self.assertEqual(p.shape, (self.B, 10, 1))
        self.assertTrue((p >= 0.0).all() and (p <= 1.0).all())

    def test_dynamic_runtime_attach_detach(self):
        base_model = DummyModel(num_layers=6, d_model=self.d_model)
        runtime = HADLDynamicRuntime(base_model=base_model, device="cpu")
        
        self.assertEqual(runtime.mid_idx, 2)
        self.assertEqual(runtime.ghost_idx, 4)
        
        runtime.attach()
        self.assertTrue(runtime._active)
        self.assertEqual(len(runtime._hooks), 2)
        
        # Forward pass through base model with dynamic hooks attached
        x = torch.randn(self.B, 8, self.d_model)
        out = base_model(x)
        self.assertEqual(out.shape, (self.B, 8, self.d_model))
        
        runtime.detach()
        self.assertFalse(runtime._active)
        self.assertEqual(len(runtime._hooks), 0)


if __name__ == "__main__":
    unittest.main()
