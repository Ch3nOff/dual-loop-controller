"""
X-Star: Autonomous Brain AI Architecture
=========================================
Endogenous Continuous Synaptic Plasticity, Vector-Grounded Neural Interface (VGNI),
and Commutative Dual-Composition Architecture.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

__version__ = "4.6.1"
__author__ = "Ch3nOff & X-Star Research Team"

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "dist" / "xstar_2b_autonomous_brain"
HF_MODEL_REPO = "MatthewChen/xstar-2b-autonomous-brain"


class XStarModel:
    """
    High-level Python wrapper for the X-Star 2B Autonomous Brain.
    Provides simple, robust inference, chat, and telemetry retrieval.
    """
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        if model_path is None:
            if DEFAULT_MODEL_DIR.exists():
                self.model_path = DEFAULT_MODEL_DIR
            else:
                try:
                    from huggingface_hub import snapshot_download
                    print(f"[*] Local model directory not found. Downloading weights from {HF_MODEL_REPO}...")
                    cache_dir = Path.home() / ".cache" / "xstar" / "models"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    downloaded_path = snapshot_download(
                        repo_id=HF_MODEL_REPO,
                        local_dir=str(cache_dir / "xstar-2b-autonomous-brain")
                    )
                    self.model_path = Path(downloaded_path)
                except Exception as e:
                    raise FileNotFoundError(
                        f"X-Star model directory not found at {DEFAULT_MODEL_DIR} and automated Hugging Face "
                        f"download failed: {e}. Please run `huggingface-cli download {HF_MODEL_REPO}` or specify model_path."
                    )
        else:
            self.model_path = Path(model_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(f"X-Star model directory not found at: {self.model_path}")

        # Register local architecture classes if present to allow safe loading with trust_remote_code=False
        try:
            model_dir_str = str(self.model_path)
            if model_dir_str not in sys.path:
                sys.path.insert(0, model_dir_str)
            from configuration_xstar import XStarConfig
            from modeling_xstar import XStar3_5ForCausalLM
            AutoConfig.register("xstar3_5", XStarConfig)
            AutoModelForCausalLM.register(XStarConfig, XStar3_5ForCausalLM)
        except Exception:
            pass

        # Device determination
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Dtype determination
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16 if "cuda" in self.device else torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            dtype=self.torch_dtype,
            trust_remote_code=trust_remote_code,
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        strip_thinking: bool = True
    ) -> str:
        """
        Generates text completion for a given prompt.
        """
        import torch
        import re

        inp = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inp.input_ids.shape[1]

        with torch.no_grad():
            do_sample = temperature > 0.0
            out_ids = self.model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id
            )

        gen_tokens = out_ids[0][input_len:]
        out_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if strip_thinking:
            clean_text = re.sub(r"<think>.*?</think>", "", out_text, flags=re.DOTALL).strip()
            return clean_text if clean_text else out_text
        return out_text

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        Multi-turn chat completion using the model's chat template.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback conversation string
                prompt = ""
                for m in messages:
                    prompt += f"{m['role'].capitalize()}: {m['content']}\n"
                prompt += "Assistant:"
        else:
            prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\nAssistant:"

        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def get_info(self) -> Dict[str, Any]:
        """
        Returns architectural specifications, hardware telemetry, and memory bank status.
        """
        import torch

        info = {
            "version": __version__,
            "model_path": str(self.model_path),
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "gpu_name": torch.cuda.get_device_name(0) if "cuda" in self.device else "N/A",
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "is_standalone_hf": True
        }

        # Check VGNI memory bank occupancy if accessible
        try:
            vgni = self.model.model.brain.vgni_engine.vgni
            info["vgni_memory_slots_filled"] = int(vgni.num_stored.item())
            info["vgni_max_capacity"] = vgni.max_entries
        except Exception:
            pass

        return info


def load(
    model_path: Optional[Union[str, Path]] = None,
    device: str = "auto",
    dtype: str = "bfloat16",
    trust_remote_code: bool = False
) -> XStarModel:
    """
    Factory function to quickly instantiate an X-Star Autonomous Brain model.
    Usage:
        import xstar
        model = xstar.load()
        print(model.generate("Explain quantum superposition."))
    """
    return XStarModel(model_path=model_path, device=device, dtype=dtype, trust_remote_code=trust_remote_code)
