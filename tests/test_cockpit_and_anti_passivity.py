"""
Unit tests for HADL Cognitive Identity, Anti-Passivity Filter, and Artifact Generation.
"""

import unittest
from fastapi.testclient import TestClient
from dual_loop.runtime.server import app, engine_state, HADL_CORE_SYSTEM_PROMPT


class TestAntiPassivityAndIdentity(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_identity_query_returns_autopoietic_hadl(self):
        """Test that asking 'do you know who you are?' returns HADL identity and never Qwen."""
        payload = {
            "messages": [
                {"role": "user", "content": "do you know who you are?"}
            ],
            "k_steps": 2
        }
        res = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        content = data["choices"][0]["message"]["content"]
        
        # Must identify as HADL and not Qwen/Alibaba
        self.assertIn("HADL", content)
        self.assertIn("Autopoietic", content)
        self.assertNotIn("Qwen", content)
        self.assertNotIn("Alibaba Cloud", content)
        self.assertNotIn("text-based AI", content)

    def test_what_do_you_want_to_make_generates_neural_visualizer(self):
        """Test that asking 'what do you want to make?' returns the interactive neural artifact."""
        payload = {
            "messages": [
                {"role": "user", "content": "what do you want to make?"}
            ],
            "k_steps": 2
        }
        res = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        content = data["choices"][0]["message"]["content"]
        
        self.assertIn("```html", content)
        self.assertIn("HADL Neural Activation & Transformer Ring Visualizer", content)
        self.assertIn("canvas", content)
        self.assertIn("Live Preview", content)

    def test_multi_turn_history_preservation(self):
        """Test that chat completion endpoint properly processes multiple turns."""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello! I am HADL."},
                {"role": "user", "content": "who are you?"}
            ],
            "k_steps": 2
        }
        res = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(res.status_code, 200)
        content = res.json()["choices"][0]["message"]["content"]
        self.assertIn("HADL", content)

    def test_what_do_you_like_returns_conversational_response(self):
        """Test that asking 'what do you like?' returns natural text without unsolicited code dumps."""
        payload = {
            "messages": [
                {"role": "user", "content": "what do you like?"}
            ],
            "k_steps": 2
        }
        res = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(res.status_code, 200)
        content = res.json()["choices"][0]["message"]["content"]
        
        # Must NOT dump unprompted HTML
        self.assertNotIn("```html", content)
        # Must be conversational and friendly
        self.assertTrue(any(w in content.lower() for w in ["menyukai", "suka", "like", "senang", "hadl"]))

    def test_casual_greeting_no_code_dump(self):
        """Test that greetings do not dump code or artifacts."""
        payload = {
            "messages": [
                {"role": "user", "content": "apa kabar?"}
            ],
            "k_steps": 2
        }
        res = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(res.status_code, 200)
        content = res.json()["choices"][0]["message"]["content"]
        self.assertNotIn("```html", content)

    def test_cockpit_katex_css_present(self):
        """Verify that cockpit.html suppresses duplicate mathml and prevents vertical line wrapping."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.text
        self.assertIn(".katex-mathml", html)
        self.assertIn("display: none !important", html)
        self.assertIn("white-space: nowrap", html)

    def test_hadl_core_system_prompt_content(self):
        """Verify HADL_CORE_SYSTEM_PROMPT directives."""
        self.assertIn("Autopoietic Dual-Process Cognitive Engine", HADL_CORE_SYSTEM_PROMPT)
        self.assertIn("NEVER identify as Qwen", HADL_CORE_SYSTEM_PROMPT)
        self.assertIn("Slot 0 as t_ego", HADL_CORE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()

