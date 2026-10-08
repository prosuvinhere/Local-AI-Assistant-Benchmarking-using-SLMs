"""FastAPI wrapper around a local Ollama daemon.

Run:  uvicorn app.main:app --port 8000
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .benchmarking import LatencyReport, run_latency_benchmark
from .config import settings
from .ollama_client import (
    ChatResult,
    ModelNotFoundError,
    OllamaClient,
    OllamaError,
    OllamaUnavailableError,
)
from .output_schemas import SCHEMA_REGISTRY, get_schema
from .schemas import BenchmarkRequest, ChatRequest, ChatResponse, StructuredRequest
from .structured import StructuredResult, generate_structured


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.client = OllamaClient()
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(
    title="Local SLM Assistant",
    version=__version__,
    description="Privacy-first assistant: every token is generated on this machine via Ollama.",
    lifespan=lifespan,
)


def _client(request: Request) -> OllamaClient:
    return request.app.state.client


# ------------------------------------------------------------ error mapping
@app.exception_handler(OllamaUnavailableError)
async def _unavailable(_: Request, exc: OllamaUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": "ollama_unavailable", "detail": str(exc)})


@app.exception_handler(ModelNotFoundError)
async def _not_found(_: Request, exc: ModelNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "model_not_found", "detail": str(exc)})


@app.exception_handler(OllamaError)
async def _ollama_error(_: Request, exc: OllamaError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"error": "ollama_error", "detail": str(exc)})


# ------------------------------------------------------------------ routes
@app.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    client = _client(request)
    return {
        "status": "ok",
        "ollama_host": client.host,
        "ollama_version": await client.version(),
        "default_model": settings.default_model,
        "loaded_models": [m["name"] for m in await client.ps()],
    }


@app.get("/v1/models")
async def models(request: Request) -> Dict[str, Any]:
    client = _client(request)
    return {
        "models": [
            {
                "name": m["name"],
                "size_bytes": m.get("size"),
                "parameter_size": (m.get("details") or {}).get("parameter_size"),
                "quantization": (m.get("details") or {}).get("quantization_level"),
                "family": (m.get("details") or {}).get("family"),
            }
            for m in await client.list_models()
        ]
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    result: ChatResult = await _client(request).chat(
        req.model or settings.default_model, req.to_messages(), options=req.options.to_ollama()
    )
    return ChatResponse(**result.model_dump())


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events: `token` events, then one `done` event with metrics."""
    client = _client(request)
    model = req.model or settings.default_model

    async def events() -> AsyncIterator[str]:
        try:
            async for ev in client.stream_chat(model, req.to_messages(), options=req.options.to_ollama()):
                if isinstance(ev, ChatResult):
                    yield f"event: done\ndata: {ev.metrics.model_dump_json()}\n\n"
                else:
                    kind = "thinking" if ev.is_thinking else "token"
                    yield f"event: {kind}\ndata: {json.dumps({'text': ev.text})}\n\n"
        except OllamaError as exc:  # headers already sent; report in-band
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/v1/schemas")
async def schemas() -> Dict[str, Any]:
    return {name: cls.model_json_schema() for name, cls in SCHEMA_REGISTRY.items()}


@app.post("/v1/structured", response_model=StructuredResult)
async def structured(req: StructuredRequest, request: Request) -> Any:
    """Schema-forced JSON with validation and exactly one repair retry.

    Returns 200 on success and 422 (same body shape, success=false) when the
    model fails validation twice - the caller always gets raw outputs + errors.
    """
    result = await generate_structured(
        _client(request),
        req.model or settings.default_model,
        req.prompt,
        get_schema(req.schema_name),
        schema_name=req.schema_name,
        enforcement=req.enforcement,
        options=req.options.to_ollama(),
        lenient=req.lenient,
    )
    if not result.success:
        return JSONResponse(status_code=422, content=result.model_dump())
    return result


@app.post("/v1/benchmark", response_model=LatencyReport)
async def benchmark(req: BenchmarkRequest, request: Request) -> LatencyReport:
    """Quick in-service benchmark. Use bench/phase1_latency.py for full runs."""
    return await run_latency_benchmark(
        _client(request),
        req.model or settings.default_model,
        [{"role": "user", "content": req.prompt}],
        req.options.to_ollama(),
        iterations=req.iterations,
        warmup=req.warmup,
        cache_bust=req.cache_bust,
        prompt_label="api",
    )
