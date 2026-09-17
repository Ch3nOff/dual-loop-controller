import unittest
import os
import tempfile
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from dual_loop import DualLoopQwenModel, attach_dual_loop_to_qwen


class TestDualLoopQwenAdapter(unittest.TestCase):
    """
    Unit and integration tests for DualLoopQwenModel and attach_dual_loop_to_qwen.
    Uses lightweight Qwen2Config to run rapidly on local CPU/GPU without downloading multi-gigabyte checkpoints.
    """
    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 1000
        self.hidden_size = 256
        self.num_layers = 4
        self.config = Qwen2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=512,
            num_hidden_layers=self.num_layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
            pad_token_id=0
        )
        self.base_model = Qwen2ForCausalLM(self.config)
        self.base_model.eval()

    def test_attachment_and_layer_targeting(self):
        """Verify target layer defaults to L // 2 and hook handles attach correctly."""
        model = attach_dual_loop_to_qwen(self.base_model, k_steps=2)
        self.assertIsInstance(model, DualLoopQwenModel)
        self.assertEqual(model.layer_idx, self.num_layers // 2)
        self.assertEqual(model.hidden_size, self.hidden_size)
        
        summary = model.get_parameter_summary()
        self.assertIn("trainable_ratio_pct", summary)
        self.assertGreater(summary["adapter_parameters"], 0)
        self.assertGreater(summary["total_parameters"], summary["adapter_parameters"])
        model.remove_hook()

    def test_universal_attach_dual_loop(self):
        """Verify universal attach_dual_loop factory operates correctly."""
        from dual_loop import attach_dual_loop, attach_dual_loop_to_model
        model1 = attach_dual_loop(self.base_model, k_steps=3)
        self.assertIsInstance(model1, DualLoopQwenModel)
        self.assertEqual(model1.k_steps, 3)
        model1.remove_hook()

        model2 = attach_dual_loop_to_model(self.base_model, k_steps=1)
        self.assertIsInstance(model2, DualLoopQwenModel)
        self.assertEqual(model2.k_steps, 1)
        model2.remove_hook()

    def test_freeze_backbone_peft(self):
        """Verify freeze_backbone keeps only adapter parameters trainable."""
        model = attach_dual_loop_to_qwen(self.base_model, k_steps=2)
        model.freeze_backbone()
        
        summary = model.get_parameter_summary()
        self.assertEqual(summary["trainable_parameters"], summary["adapter_parameters"])
        self.assertLess(summary["trainable_ratio_pct"], 50.0)
        
        # Verify base layers have requires_grad == False
        for param in model.qwen.parameters():
            self.assertFalse(param.requires_grad)
            
        # Verify adapter layers have requires_grad == True
        for param in model.adapter.parameters():
            self.assertTrue(param.requires_grad)
        model.remove_hook()

    def test_k0_strict_identity_equivalence(self):
        """Verify K=0 produces exact bitwise identical outputs to unaugmented base model."""
        input_ids = torch.randint(1, self.vocab_size, (2, 8))
        
        # Base model forward pass without adapter
        with torch.no_grad():
            base_out = self.base_model(input_ids)
            
        # Wrapped model forward pass with k_steps=0
        model = attach_dual_loop_to_qwen(self.base_model, k_steps=0)
        with torch.no_grad():
            k0_out = model(input_ids)
            
        self.assertTrue(torch.allclose(base_out.logits, k0_out.logits, atol=1e-6))
        self.assertTrue(model.last_telemetry.get("bypassed", False))
        model.remove_hook()

    def test_k_greater_than_zero_residual_deliberation(self):
        """Verify K>0 alters representation while preserving exact sequence length and shape."""
        input_ids = torch.randint(1, self.vocab_size, (2, 8))
        model = attach_dual_loop_to_qwen(self.base_model, k_steps=2)
        
        with torch.no_grad():
            out_k2 = model(input_ids)
            
        self.assertEqual(out_k2.logits.shape, (2, 8, self.vocab_size))
        self.assertEqual(model.last_telemetry["ponder_steps"], 2)
        self.assertFalse(model.last_telemetry.get("bypassed", False))
        
        # Switch to k_steps=3
        model.set_ponder_steps(3)
        with torch.no_grad():
            out_k3 = model(input_ids)
        self.assertEqual(model.last_telemetry["ponder_steps"], 3)
        
        # Differing ponder steps should yield differing refined logits
        diff = (out_k3.logits - out_k2.logits).abs().sum().item()
        self.assertGreater(diff, 1e-4)
        model.remove_hook()

    def test_peft_gradient_backward(self):
        """Verify gradients flow backward through causal LM loss into the adapter only."""
        # query_idx=-2 targets the prompt/query token before the final label in shift_logits
        model = attach_dual_loop_to_qwen(self.base_model, k_steps=2, query_idx=-2)
        model.freeze_backbone()
        model.train()
        
        input_ids = torch.randint(1, self.vocab_size, (2, 8))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        self.assertIsNotNone(loss)
        loss.backward()
        
        # Base model parameters must have no grad
        for name, param in model.qwen.named_parameters():
            self.assertIsNone(param.grad, f"Base parameter {name} has grad despite being frozen!")
            
        # Adapter parameters must have valid grad
        has_adapter_grad = any(p.grad is not None and p.grad.norm().item() > 0 for p in model.adapter.parameters())
        self.assertTrue(has_adapter_grad, "Adapter parameters failed to receive gradients!")
        model.remove_hook()

    def test_autoregressive_generation_compatibility(self):
        """Verify generate() runs without crashing or dimension mismatch."""
        model = attach_dual_loop_to_qwen(self.base_model, k_steps=2)
        model.eval()
        
        input_ids = torch.tensor([[10, 25, 42]], dtype=torch.long)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=4,
                pad_token_id=self.config.pad_token_id,
                do_sample=False
            )
            
        self.assertEqual(generated.shape, (1, 7))
        model.remove_hook()

    def test_save_and_load_adapter(self):
        """Verify lightweight adapter weights can be saved and reloaded."""
        model = attach_dual_loop_to_qwen(self.base_model, k_steps=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "dual_loop_adapter.pt")
            model.save_adapter(save_path)
            self.assertTrue(os.path.exists(save_path))
            
            # Recreate new model and load
            model2 = attach_dual_loop_to_qwen(self.base_model, k_steps=2)
            model2.load_adapter(save_path)
            
            # Check parameter equality
            for p1, p2 in zip(model.adapter.parameters(), model2.adapter.parameters()):
                self.assertTrue(torch.equal(p1, p2))
                
            model.remove_hook()
            model2.remove_hook()


if __name__ == "__main__":
    unittest.main()
