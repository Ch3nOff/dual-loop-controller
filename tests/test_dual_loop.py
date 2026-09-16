import unittest
import torch
from dual_loop.memory import CognitiveWorkingMemory
from dual_loop.halting import EntropyHaltingUnit
from dual_loop.controller import TopKCapacityCrossAttention, RecurrentLatentController
from dual_loop.decoder import DualLoopTransformer

class TestDualLoopComponents(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 4
        self.N = 32
        self.D = 64
        self.M = 8
        self.L_thought = 4
        self.vocab_size = 30

    def test_cwm_compression(self):
        cwm = CognitiveWorkingMemory(d_model=self.D, num_slots=self.M)
        context = torch.randn(self.B, self.N, self.D)
        compressed = cwm(context)
        self.assertEqual(compressed.shape, (self.B, self.M, self.D))

    def test_topk_capacity_cross_attention(self):
        cap_attn = TopKCapacityCrossAttention(d_model=self.D, n_heads=4, capacity_factor=0.5)
        thoughts = torch.randn(self.B, self.L_thought, self.D)
        memory = torch.randn(self.B, self.M, self.D)
        out = cap_attn(thoughts, memory)
        self.assertEqual(out.shape, (self.B, self.L_thought, self.D))

    def test_entropy_halting_unit(self):
        halting = EntropyHaltingUnit(entropy_threshold=0.5)
        # Uniform distribution (high entropy)
        uniform_logits = torch.zeros(self.B, self.vocab_size)
        halt_mask, ent = halting.should_halt(uniform_logits)
        self.assertFalse(halt_mask.any())

        # Highly peaked distribution (low entropy)
        peaked_logits = torch.zeros(self.B, self.vocab_size)
        peaked_logits[:, 0] = 50.0
        halt_mask_peaked, ent_peaked = halting.should_halt(peaked_logits)
        self.assertTrue(halt_mask_peaked.all())

    def test_recurrent_controller(self):
        controller = RecurrentLatentController(
            d_model=self.D,
            n_heads=4,
            d_ff=128,
            num_thought_tokens=self.L_thought,
            max_ponder_steps=3,
            vocab_size=self.vocab_size
        )
        query = torch.randn(self.B, self.D)
        memory = torch.randn(self.B, self.M, self.D)
        
        H, aux, entropies = controller(query, memory, return_aux=True)
        self.assertEqual(H.shape, (self.B, self.L_thought, self.D))
        self.assertEqual(len(aux), 3)

    def test_dual_loop_transformer_forward(self):
        model = DualLoopTransformer(
            vocab_size=self.vocab_size,
            d_model=self.D,
            n_heads=4,
            d_ff=128,
            num_thought_tokens=self.L_thought,
            num_cwm_slots=self.M,
            max_ponder_steps=3
        )
        input_ids = torch.randint(0, self.vocab_size, (self.B, self.N))
        logits, info = model(input_ids, return_aux=True)
        self.assertEqual(logits.shape, (self.B, self.vocab_size))
        self.assertEqual(info["num_thoughts"], self.L_thought)

    def test_per_sample_dynamic_halting_forward(self):
        model = DualLoopTransformer(
            vocab_size=self.vocab_size,
            d_model=self.D,
            n_heads=4,
            d_ff=128,
            max_ponder_steps=3,
            entropy_threshold=1.30
        )
        input_ids = torch.randint(0, self.vocab_size, (8, self.N))
        logits, info = model(input_ids, dynamic_halting=True)
        self.assertEqual(logits.shape, (8, self.vocab_size))
        self.assertIn("steps_taken", info)
        self.assertEqual(info["steps_taken"].shape, (8,))
        # Check that effective_k is a valid float within [1.0, 3.0]
        self.assertTrue(1.0 <= info["effective_k"] <= 3.0)

    def test_gradient_flow(self):
        model = DualLoopTransformer(
            vocab_size=self.vocab_size,
            d_model=self.D,
            max_ponder_steps=3
        )
        input_ids = torch.randint(0, self.vocab_size, (self.B, self.N))
        target = torch.randint(0, self.vocab_size, (self.B,))
        
        logits, _ = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        
        # Check that gradients flow to thought projector and recurrent layers
        self.assertIsNotNone(model.outer_loop.query_projector[0].weight.grad)
        self.assertGreater(model.outer_loop.query_projector[0].weight.grad.norm().item(), 0.0)

    def test_calibrate_halting_default_and_search(self):
        model = DualLoopTransformer(
            vocab_size=self.vocab_size,
            d_model=self.D,
            max_ponder_steps=3
        )
        sample_inputs = torch.randint(0, self.vocab_size, (16, self.N))
        
        # Test 1: Default percentile (75.0)
        calib_thresh = model.calibrate_halting(sample_inputs)
        self.assertIsInstance(calib_thresh, float)
        self.assertEqual(model.outer_loop.halting_unit.entropy_threshold, calib_thresh)

        # Test 2: Automated Grid-Search with target_labels
        target_labels = torch.randint(0, self.vocab_size, (16,))
        best_thresh = model.calibrate_halting(sample_inputs, target_labels=target_labels)
        self.assertIsInstance(best_thresh, float)
        self.assertEqual(model.outer_loop.halting_unit.entropy_threshold, best_thresh)

    def test_load_trained_checkpoint(self):
        from dual_loop import load_trained_checkpoint
        model = DualLoopTransformer(vocab_size=19, d_model=64, num_cwm_slots=12, max_ponder_steps=3)
        loaded_model, loaded_path = load_trained_checkpoint(model)
        self.assertIsNotNone(loaded_model)
        self.assertTrue(len(loaded_path) > 0)

    def test_multihop_graph_dataset_reproducibility(self):
        from dual_loop.benchmarks import MultiHopGraphDataset
        ds1 = MultiHopGraphDataset(num_samples=50, num_nodes=16, num_edges=6, hops=3, seed=123)
        ds2 = MultiHopGraphDataset(num_samples=50, num_nodes=16, num_edges=6, hops=3, seed=123)
        
        # Check first sequence and targets match exactly
        self.assertTrue(torch.equal(ds1.data[0][0], ds2.data[0][0]))
        self.assertTrue(torch.equal(ds1.data[0][1], ds2.data[0][1]))

        # Check get_batch matches
        b1_x, b1_y = ds1.get_batch(20, shuffle=False)
        b2_x, b2_y = ds2.get_batch(20, shuffle=False)
        self.assertTrue(torch.equal(b1_x, b2_x))
        self.assertTrue(torch.equal(b1_y, b2_y))

if __name__ == "__main__":
    unittest.main()
