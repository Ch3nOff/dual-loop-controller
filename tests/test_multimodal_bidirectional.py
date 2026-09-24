import unittest
import torch
import torch.nn.functional as F

from dual_loop import (
    attach,
    ProcrustesOptimalManifoldTransport,
    SpatioTemporalEntropicCWM,
    HeteroAssociativePlasticMemory,
    LatentDeliberationAdapter
)


class TestMultimodalBidirectional(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.d_model = 128
        self.rank = 32

    def test_procrustes_optimal_manifold_transport(self):
        transport = ProcrustesOptimalManifoldTransport(d_model=self.d_model).to(self.device)
        h_vision = torch.randn(2, 64, self.d_model, device=self.device) * 3.0 + 2.0
        h_text = torch.randn(2, 16, self.d_model, device=self.device) * 0.5 - 1.0

        h_tilde, telem = transport(h_vision, h_text)
        self.assertEqual(h_tilde.shape, h_vision.shape)
        self.assertTrue(telem["transport_active"])
        self.assertAlmostEqual(h_tilde.mean().item(), h_text.mean().item(), delta=0.5)

    def test_spatio_temporal_entropic_cwm(self):
        cwm = SpatioTemporalEntropicCWM(d_model=self.d_model, num_slots=8, n_heads=4).to(self.device)
        h_text = torch.randn(2, 20, self.d_model, device=self.device)
        h_vision = torch.randn(2, 64, self.d_model, device=self.device)

        memory, visual_slots, telem = cwm(h_text, h_vision)
        self.assertEqual(memory.shape, (2, 8, self.d_model))
        self.assertEqual(visual_slots.shape, (2, 4, self.d_model))
        self.assertIn("compression_ratio", telem)
        self.assertGreaterEqual(telem["compression_ratio"], 0.0)

    def test_hetero_associative_bidirectional_plasticity(self):
        mem = HeteroAssociativePlasticMemory(
            d_model=self.d_model,
            rank=self.rank,
            plastic_lr=0.5,
            decay_rate=0.0
        ).to(self.device)
        mem.set_continual_mode(True, decay=1.0)

        img = torch.randn(1, 16, self.d_model, device=self.device)
        txt = torch.randn(1, 1, self.d_model, device=self.device)

        # 1. Bind concept
        res_bind = mem.bind_concept(img, txt)
        self.assertIn("m_cross_norm", res_bind)
        self.assertGreater(res_bind["m_cross_norm"], 0.0)

        # 2. Forward Recall: Image -> Text under noise
        noisy_img = img + 0.15 * torch.randn_like(img)
        rec_txt, telem_txt = mem.recall_from_visual(noisy_img)
        self.assertEqual(rec_txt.shape, noisy_img.shape)
        self.assertTrue(telem_txt["recalled"])

        # Compare cosine
        cos_fwd = F.cosine_similarity(rec_txt.mean(dim=1, keepdim=True), txt, dim=-1).item()
        self.assertGreater(cos_fwd, 0.10)

        # 3. Transposed Recall: Text -> Image Mental Imagery
        synth_img, telem_img = mem.recall_from_text(txt)
        self.assertEqual(synth_img.shape, txt.shape)
        self.assertTrue(telem_img["recalled"])

        # Compare cosine with mean visual representation
        cos_bwd = F.cosine_similarity(synth_img, img.mean(dim=1, keepdim=True), dim=-1).item()
        self.assertGreater(cos_bwd, 0.10)

    def test_latent_deliberation_adapter_multimodal_api(self):
        adapter = LatentDeliberationAdapter(
            d_model=self.d_model,
            n_heads=4,
            num_thought_tokens=2,
            max_ponder_steps=2,
            num_cwm_slots=8,
            adapter_mode="residual"
        ).to(self.device)
        adapter.eval()
        adapter.set_continual_mode(True, decay=1.0)

        img = torch.randn(1, 32, self.d_model, device=self.device)
        txt = torch.randn(1, 1, self.d_model, device=self.device)

        # Bind through adapter
        res_bind = adapter.bind_visual_concept(img, txt)
        self.assertTrue(res_bind["bound_active"])

        # Recall text from sensory
        rec_txt, _ = adapter.recall_text_from_sensory(img)
        self.assertEqual(rec_txt.shape, img.shape)

        # Recall sensory from text
        synth_img, _ = adapter.recall_sensory_from_text(txt)
        self.assertEqual(synth_img.shape, txt.shape)

        # Forward pass with visual embeds
        prompt = torch.randn(1, 10, self.d_model, device=self.device)
        enhanced, telem = adapter(prompt, visual_embeds=img, k_steps=2)
        self.assertEqual(enhanced.shape, prompt.shape)
        self.assertIn("multimodal_transport", telem)


if __name__ == "__main__":
    unittest.main()
