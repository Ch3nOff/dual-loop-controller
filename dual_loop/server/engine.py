"""
Dual-Loop Custom Inference Engine
=================================
Model-agnostic inference engine that loads and runs any open-source Hugging Face
model with Dual-Loop Latent Deliberation and dynamic VRAM auto-tuning.
Supports real-time streaming token generation and standard synchronous generation.
"""

from __future__ import annotations

import os
import sys
import time
import math
import threading
from typing import Dict, Any, Optional, List, Tuple, Generator, Union

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

from .vram_tuner import VRAMAutoTuner, AllocationPlan
from ..adapters.universal_adapter import attach_universal_dual_loop


class DualLoopInferenceEngine:
    """
    High-performance, hardware-aligned inference engine wrapping any Hugging Face
    causal language model with Dual-Loop System 2 latent deliberation.
    """

    def __init__(
        self,
        model_id_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        headroom_gib: Union[str, float] = "auto",
        forced_regime: Optional[str] = None,
        k_steps: int = 2,
        trust_remote_code: bool = False,
        device: str = "cuda:0"
    ):
        self.model_id = model_id_or_path
        self.k_steps = k_steps
        self.trust_remote_code = trust_remote_code
        self.device = device if torch.cuda.is_available() else "cpu"

        # 1. Calculate hardware allocation plan
        self.plan: AllocationPlan = VRAMAutoTuner.calculate_plan(
            model_id_or_path=model_id_or_path,
            headroom_gib=headroom_gib,
            forced_regime=forced_regime,
            device_id=0 if "cuda" in self.device else -1
        )

        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.total_tokens_generated = 0
        self.total_generation_time_sec = 0.0

    def _resolve_model_path(self, model_id_or_path: str) -> str:
        """Resolves model path dynamically from HF cache or local path without hardcoded PII (SEC-03)."""
        if os.path.exists(model_id_or_path):
            return model_id_or_path

        # Check standard user Hugging Face cache dynamically
        hf_cache = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        if os.path.isdir(hf_cache):
            sanitized = f"models--{model_id_or_path.replace('/', '--')}"
            snapshots_dir = os.path.join(hf_cache, sanitized, "snapshots")
            if os.path.isdir(snapshots_dir):
                snapshots = [os.path.join(snapshots_dir, s) for s in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, s))]
                if snapshots:
                    return snapshots[0]
        return model_id_or_path

    def _load_base_model(self, **kwargs):
        """Dispatches to the correct model loader based on model architecture."""
        model_path = self._resolve_model_path(self.model_id)

        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=self.trust_remote_code)
            if getattr(cfg, "model_type", None) == "qwen3_5":
                from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
                return Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_path,
                    trust_remote_code=self.trust_remote_code,
                    **kwargs
                )
        except Exception:
            pass
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
            **kwargs
        )

    def load_model(self) -> None:
        """Loads tokenizer and attaches Dual-Loop Controller according to the allocation plan."""
        print("=" * 80)
        print(f"[*] INITIALIZING DUAL-LOOP INFERENCE ENGINE")
        print(f"[*] Model ID            : {self.model_id}")
        print(f"[*] Architecture        : {self.plan.model_spec.architecture} (~{self.plan.model_spec.parameters_billion}B params)")
        print(f"[*] Detected GPU        : {self.plan.hardware.device_name} ({self.plan.hardware.total_vram_gib} GiB VRAM)")
        print(f"[*] Headroom Reserved   : {self.plan.requested_headroom_gib} GiB")
        print(f"[*] Effective Budget    : {self.plan.effective_budget_gib:.2f} GiB")
        print(f"[*] Selected Regime     : {self.plan.selected_regime}")
        print(f"[*] Execution Strategy  : {self.plan.description}")
        print("=" * 80)

        t0 = time.time()

        # 1. Load Tokenizer (check local cache snapshot first to prevent HF Hub timeout)
        tok_source = self._resolve_model_path(self.model_id)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tok_source,
                trust_remote_code=self.trust_remote_code
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"[!] Warning loading tokenizer from {tok_source}: {e}")
            print(f"[*] Falling back to universal Qwen tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Model based on selected regime
        regime = self.plan.selected_regime

        if regime == "NF4":
            print(f"[*] Loading 4-bit NormalFloat (NF4) quantized base model with Hybrid GPU+CPU Allocation...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            # Reserve headroom on GPU 0 and allocate remainder to Host RAM
            gpu_mem = f"{max(2.0, self.plan.effective_budget_gib):.1f}GiB"
            cpu_mem = f"{max(8.0, self.plan.hardware.host_ram_avail_gib - 2.0):.1f}GiB"
            max_mem_plan = {0: gpu_mem, "cpu": cpu_mem}
            print(f"[*] Memory Target: GPU={gpu_mem}, Host RAM={cpu_mem}")
            base = self._load_base_model(
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_mem_plan
            )
            self.model = attach_universal_dual_loop(
                base,
                max_ponder_steps=self.k_steps,
                enable_plasticity=True,
                enable_firewall=True
            )

        elif regime == "INT8":
            print(f"[*] Loading 8-bit integer quantized base model...")
            base = self._load_base_model(
                load_in_8bit=True,
                device_map="auto"
            )
            self.model = attach_universal_dual_loop(
                base,
                max_ponder_steps=self.k_steps,
                enable_plasticity=True,
                enable_firewall=True
            )

        else:  # BF16
            print(f"[*] Loading full BF16 base model directly to GPU...")
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            base = self._load_base_model(
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.model = attach_universal_dual_loop(
                base,
                max_ponder_steps=self.k_steps,
                enable_plasticity=True,
                enable_firewall=True
            )

        load_sec = time.time() - t0
        self.is_loaded = True

        # Verify GPU VRAM status
        if torch.cuda.is_available():
            alloc_gib = torch.cuda.memory_allocated(0) / (1024 ** 3)
            free_b, total_b = torch.cuda.mem_get_info(0)
            print(f"[OK] Model successfully loaded into VRAM in {load_sec:.2f}s!")
            print(f"[*] VRAM Allocated       : {alloc_gib:.2f} GiB")
            print(f"[*] VRAM Free Headroom   : {free_b / (1024 ** 3):.2f} GiB")
        else:
            print(f"[OK] Model loaded in CPU RAM in {load_sec:.2f}s!")

    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Formats a list of OpenAI chat messages into a standard prompt string.
        """
        formatted = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    def reset_memory_states(self, force: bool = True) -> None:
        """
        Resets in-situ fast plasticity working memory traces across generation requests
        to prevent cross-query Hebbian associative leakage.
        """
        if not hasattr(self, "model") or self.model is None:
            return

        # 1. Inspect wrapper.adapter
        adapter = getattr(self.model, "adapter", None)
        if adapter is not None:
            if hasattr(adapter, "plastic_memory") and adapter.plastic_memory is not None:
                if hasattr(adapter.plastic_memory, "reset_state"):
                    adapter.plastic_memory.reset_state(force=force)
            if hasattr(adapter, "controller") and adapter.controller is not None:
                if hasattr(adapter.controller, "reset_state"):
                    adapter.controller.reset_state(force=force)
                if hasattr(adapter.controller, "plastic_unit") and adapter.controller.plastic_unit is not None:
                    if hasattr(adapter.controller.plastic_unit, "reset_state"):
                        adapter.controller.plastic_unit.reset_state(force=force)

        # 2. Inspect root model directly
        if hasattr(self.model, "plastic_memory") and self.model.plastic_memory is not None:
            if hasattr(self.model.plastic_memory, "reset_state"):
                self.model.plastic_memory.reset_state(force=force)
        if hasattr(self.model, "controller") and self.model.controller is not None:
            if hasattr(self.model.controller, "reset_state"):
                self.model.controller.reset_state(force=force)

    def generate_sync(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Synchronous text generation returning complete output and telemetry."""
        if not self.is_loaded:
            self.load_model()

        # Prevent cross-request fast weight associative state leakage
        self.reset_memory_states(force=True)

        t0 = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            if hasattr(self.model, "generate"):
                out_tokens = self.model.generate(**inputs, **gen_kwargs)
                out_ids = out_tokens[0][prompt_len:]
                output_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
                completion_tokens = len(out_ids)
            else:
                raise RuntimeError("Loaded model does not implement generate().")

        elapsed = max(0.001, time.time() - t0)
        tok_sec = completion_tokens / elapsed

        self.total_tokens_generated += completion_tokens
        self.total_generation_time_sec += elapsed

        return {
            "text": output_text,
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_len + completion_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(tok_sec, 2),
            "regime": self.plan.selected_regime
        }

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> Generator[str, None, None]:
        """
        Streaming token generator yielding real-time text chunks.
        Used for OpenAI Server-Sent Events (SSE) streaming with Hermes Agent.
        """
        if not self.is_loaded:
            self.load_model()

        # Prevent cross-request fast weight associative state leakage
        self.reset_memory_states(force=True)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Run model.generate in a background thread while main thread yields from streamer
        if hasattr(self.model, "generate"):
            thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

            for text_chunk in streamer:
                yield text_chunk

            thread.join()
        else:
            raise RuntimeError("Loaded model does not implement generate().")
