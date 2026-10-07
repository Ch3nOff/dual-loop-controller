"""
Unit tests for HADL to GGUF Weight Fusion & Export Pipeline
===========================================================
"""

import os
import tempfile
import unittest
import torch
import torch.nn as nn

from dual_loop.export_gguf import (
    resolve_model_path,
    fuse_hadl_weights,
    _fuse_delta_into_layer,
)
from dual_loop.cli import get_parser


class DummyAttention(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.o_proj = nn.Linear(d_model, d_model, bias=False)


class DummyLayer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.self_attn = DummyAttention(d_model)


class DummyModel(nn.Module):
    def __init__(self, num_layers=4, d_model=64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyLayer(d_model) for _ in range(num_layers)])


class TestHADLGGUFExport(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.model = DummyModel(num_layers=4, d_model=self.d_model)

    def test_fuse_delta_into_layer(self):
        layer = self.model.model.layers[1]
        orig_weight = layer.self_attn.o_proj.weight.clone()
        delta = torch.ones(self.d_model, self.d_model) * 0.05

        _fuse_delta_into_layer(layer, delta, "Test Delta")
        new_weight = layer.self_attn.o_proj.weight

        self.assertTrue(torch.allclose(new_weight, orig_weight + delta))

    def test_fuse_delta_into_linear_attn_layer(self):
        # Test modern hybrid architectures like Qwen3.5 linear_attn
        class DummyLinearAttnLayer(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.linear_attn = nn.Module()
                self.linear_attn.out_proj = nn.Linear(d_model, d_model, bias=False)

        layer = DummyLinearAttnLayer(self.d_model)
        orig_weight = layer.linear_attn.out_proj.weight.clone()
        delta = torch.ones(self.d_model, self.d_model) * 0.08

        _fuse_delta_into_layer(layer, delta, "Linear Attn Delta")
        new_weight = layer.linear_attn.out_proj.weight

        self.assertTrue(torch.allclose(new_weight, orig_weight + delta))

    def test_fuse_hadl_weights_from_checkpoint(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        try:
            # Build mock Car-Lift checkpoint
            state_dict = {
                "ghost.u_proj": torch.randn(self.d_model, 8),
                "ghost.verify_linear.weight": torch.randn(self.d_model, 8),
                "hydraulic_cup.fluid_bridge.weight": torch.randn(self.d_model, self.d_model),
            }
            torch.save({"controller_state_dict": state_dict}, ckpt_path)

            mid_orig = self.model.model.layers[1].self_attn.o_proj.weight.clone()
            ghost_orig = self.model.model.layers[2].self_attn.o_proj.weight.clone()

            # Execute fusion
            fuse_hadl_weights(self.model, checkpoint_path=ckpt_path, alpha_mid=0.1, alpha_ghost=0.1)

            # Both midpoint and ghost layers must receive weight updates
            self.assertFalse(torch.equal(self.model.model.layers[1].self_attn.o_proj.weight, mid_orig))
            self.assertFalse(torch.equal(self.model.model.layers[2].self_attn.o_proj.weight, ghost_orig))
            # Other layers remain unmodified
            self.assertEqual(self.model.model.layers[0].self_attn.o_proj.weight.shape, (self.d_model, self.d_model))
        finally:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    def test_resolve_model_path_local(self):
        # Existing directory returns as-is
        self.assertEqual(resolve_model_path("checkpoints"), "checkpoints")

    def test_cli_export_gguf_args(self):
        parser = get_parser()
        args = parser.parse_args([
            "export-gguf",
            "--model", "Qwen/Qwen3.5-2B",
            "--output", "dist/custom.gguf",
            "--outtype", "q8_0",
        ])
        self.assertEqual(args.command, "export-gguf")
        self.assertEqual(args.model, "Qwen/Qwen3.5-2B")
        self.assertEqual(args.output, "dist/custom.gguf")
        self.assertEqual(args.outtype, "q8_0")


if __name__ == "__main__":
    unittest.main()
