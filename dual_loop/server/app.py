"""
OpenAI-Compatible REST API Server for Dual-Loop Inference Engine
================================================================
Exposes standard OpenAI v1 endpoints (/v1/chat/completions, /v1/models, /v1/completions)
enabling seamless drop-in integration with Hermes Agent, LM Studio, Ollama clients,
LangChain, AutoGen, and Cursor.
"""

from __future__ import annotations

import time
import uuid
import json
import asyncio
from typing import Dict, Any, Optional, List, Union

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import DualLoopInferenceEngine


# Pydantic Schemas matching OpenAI API
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


def create_app(engine: DualLoopInferenceEngine) -> FastAPI:
    """Creates the FastAPI server application with OpenAI v1 endpoints."""
    app = FastAPI(
        title="Dual-Loop Inference Engine (OpenAI Compatible)",
        description="Autonomous System 2 Deliberative Cognitive OS Inference Server",
        version="3.1.0"
    )

    # Enable CORS for all external clients (Hermes Agent, Open-WebUI, LM Studio)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        hw = engine.plan.hardware
        plan = engine.plan
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Dual-Loop Cognitive OS Inference Server</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; max-width: 800px; margin: 0 auto; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155; }}
        h1 {{ color: #38bdf8; margin-top: 0; display: flex; align-items: center; gap: 10px; }}
        .badge {{ background: #0284c7; color: white; padding: 4px 10px; border-radius: 20px; font-size: 14px; font-weight: bold; }}
        .stats {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 20px 0; }}
        .stat-box {{ background: #0f172a; padding: 14px; border-radius: 8px; border: 1px solid #1e293b; }}
        .stat-lbl {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-val {{ font-size: 18px; font-weight: bold; color: #f1f5f9; margin-top: 4px; }}
        code {{ background: #0f172a; padding: 2px 6px; border-radius: 4px; color: #38bdf8; font-family: monospace; }}
        pre {{ background: #0f172a; padding: 14px; border-radius: 8px; overflow-x: auto; color: #a5f3fc; border: 1px solid #334155; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Dual-Loop Inference Engine <span class="badge">v3.1.0 Online</span></h1>
        <p>OpenAI-Compatible Local API Server with Hardware-Aligned Latent Deliberation & Zero OOM.</p>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-lbl">Active Model</div>
                <div class="stat-val">{engine.model_id}</div>
            </div>
            <div class="stat-box">
                <div class="stat-lbl">Execution Regime</div>
                <div class="stat-val" style="color: #4ade80;">{plan.selected_regime}</div>
            </div>
            <div class="stat-box">
                <div class="stat-lbl">Detected Hardware</div>
                <div class="stat-val">{hw.device_name} ({hw.total_vram_gib} GiB VRAM)</div>
            </div>
            <div class="stat-box">
                <div class="stat-lbl">VRAM Headroom Reserved</div>
                <div class="stat-val">{plan.requested_headroom_gib} GiB Safety Buffer</div>
            </div>
        </div>

        <h3>🔗 Connect with Hermes Agent / OpenAI Client</h3>
        <p>In your agent configuration (Hermes Agent, LangChain, Cursor, AutoGen):</p>
        <pre><code>export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="not-needed"
export OPENAI_MODEL_NAME="{engine.model_id}"</code></pre>

        <h3>Python Client Example</h3>
        <pre><code>from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dual-loop")
response = client.chat.completions.create(
    model="{engine.model_id}",
    messages=[{{"role": "user", "content": "Hello Hermes, explain active inference."}}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)</code></pre>
    </div>
</body>
</html>"""

    @app.get("/v1/models")
    @app.get("/models")
    async def list_models():
        """Lists active loaded model in standard OpenAI format."""
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "dual-loop-controller",
                    "permission": [],
                    "root": engine.model_id,
                    "parent": None,
                    "dual_loop_regime": engine.plan.selected_regime,
                    "vram_headroom_gib": engine.plan.requested_headroom_gib
                }
            ]
        }

    @app.get("/v1/models/{model_id:path}")
    @app.get("/models/{model_id:path}")
    async def get_model(model_id: str):
        """Returns model details for OpenAI / LM Studio clients."""
        return {
            "id": engine.model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "dual-loop-controller",
            "permission": [],
            "root": engine.model_id,
            "parent": None,
            "dual_loop_regime": engine.plan.selected_regime,
            "vram_headroom_gib": engine.plan.requested_headroom_gib
        }

    # Compatibility endpoints for Ollama & LM Studio clients
    @app.get("/api/tags")
    @app.get("/api/v1/models")
    async def ollama_tags():
        """Returns model list in Ollama format so Ollama WebUIs/clients recognize the server."""
        return {
            "models": [
                {
                    "name": engine.model_id,
                    "model": engine.model_id,
                    "modified_at": "2026-09-30T12:00:00Z",
                    "size": int(engine.plan.estimated_model_vram_gib * (1024 ** 3)),
                    "digest": "sha256:dual-loop-controller",
                    "details": {
                        "parent_model": "",
                        "format": "safetensors",
                        "family": "qwen",
                        "families": ["qwen"],
                        "parameter_size": f"{engine.plan.model_spec.parameters_billion}B",
                        "quantization_level": engine.plan.selected_regime
                    }
                }
            ]
        }

    @app.post("/api/show")
    async def ollama_show():
        """Returns model metadata in Ollama format."""
        return {
            "modelfile": f"FROM {engine.model_id}",
            "parameters": "",
            "template": "",
            "details": {
                "format": "safetensors",
                "family": "qwen",
                "parameter_size": f"{engine.plan.model_spec.parameters_billion}B"
            }
        }

    @app.get("/version")
    @app.get("/api/version")
    async def server_version():
        """Returns server version for client health check."""
        return {"version": "3.1.0"}

    @app.get("/props")
    @app.get("/v1/props")
    async def server_props():
        """Returns server properties for LM Studio."""
        return {"version": "3.1.0", "status": "ok"}

    @app.get("/v1/health")
    @app.get("/health")
    async def health():
        """Diagnostics, memory telemetry, and active brain organ telemetry."""
        return {
            "status": "healthy",
            "model_id": engine.model_id,
            "architecture": engine.plan.model_spec.architecture,
            "regime": engine.plan.selected_regime,
            "vram_allocated_estimated_gib": engine.plan.estimated_model_vram_gib,
            "vram_headroom_reserved_gib": engine.plan.requested_headroom_gib,
            "hardware": {
                "gpu": engine.plan.hardware.device_name,
                "total_vram_gib": engine.plan.hardware.total_vram_gib,
                "free_vram_gib": engine.plan.hardware.free_vram_gib,
                "host_ram_total_gib": engine.plan.hardware.host_ram_total_gib
            },
            "telemetry": {
                "total_tokens_generated": engine.total_tokens_generated,
                "total_generation_time_sec": round(engine.total_generation_time_sec, 2),
                "k_steps": engine.k_steps
            }
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """
        OpenAI-compatible chat completion endpoint.
        Handles both streaming (text/event-stream) and non-streaming responses.
        """
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())
        messages = [m.model_dump() for m in request.messages]

        # Format prompt using model chat template or Hermes/ChatML fallback
        prompt = engine.format_chat_prompt(messages)
        stop_tokens = [request.stop] if isinstance(request.stop, str) else (request.stop or [])

        # 1. STREAMING MODE (Hermes Agent / Web UI)
        if request.stream:
            async def event_generator():
                # Initial role chunk
                init_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": engine.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(init_chunk)}\n\n"

                # Stream token chunks from engine
                temp = 0.0 if request.temperature is None else float(request.temperature)
                top_p = 0.9 if request.top_p is None else float(request.top_p)
                for text_chunk in engine.generate_stream(
                    prompt=prompt,
                    max_tokens=request.max_tokens or 1024,
                    temperature=temp,
                    top_p=top_p,
                    stop=stop_tokens
                ):
                    chunk_payload = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": engine.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text_chunk},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk_payload)}\n\n"
                    await asyncio.sleep(0.001)

                # Final finish chunk
                finish_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": engine.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 2. NON-STREAMING MODE
        temp = 0.0 if request.temperature is None else float(request.temperature)
        top_p = 0.9 if request.top_p is None else float(request.top_p)
        result = engine.generate_sync(
            prompt=prompt,
            max_tokens=request.max_tokens or 1024,
            temperature=temp,
            top_p=top_p,
            stop=stop_tokens
        )

        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_time,
            "model": engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["text"]
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"]
            },
            "dual_loop_telemetry": {
                "elapsed_seconds": result["elapsed_seconds"],
                "tokens_per_second": result["tokens_per_second"],
                "regime": result["regime"]
            }
        }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        """Legacy text completion endpoint."""
        cmpl_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())
        prompt = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)

        result = engine.generate_sync(
            prompt=prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9
        )

        return {
            "id": cmpl_id,
            "object": "text_completion",
            "created": created_time,
            "model": engine.model_id,
            "choices": [
                {
                    "text": result["text"],
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"]
            }
        }

    return app
