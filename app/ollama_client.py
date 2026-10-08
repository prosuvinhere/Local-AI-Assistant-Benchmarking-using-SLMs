"""Async Ollama REST client with high-resolution latency instrumentation.

All generation goes through :meth:`OllamaClient.stream_chat`, which always
streams from Ollama (even when the caller wants a single blob) so that time to
first token and inter-token latency can be observed on the wire.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel

from .config import settings
from .metrics import GenerationMetrics


class OllamaError(RuntimeError):
    """Generic Ollama failure (HTTP error, malformed stream, ...)."""


class OllamaUnavailableError(OllamaError):
    """The Ollama daemon cannot be reached."""


class ModelNotFoundError(OllamaError):
    """The requested model has not been pulled."""


@dataclass
class TokenChunk:
    text: str
    is_thinking: bool = False


class ChatResult(BaseModel):
    model: str
    content: str
    thinking: Optional[str] = None
    metrics: GenerationMetrics


def canonical_model_name(name: str) -> str:
    """'phi4-mini' -> 'phi4-mini:latest' (Ollama's implicit tag)."""
    last = name.rsplit("/", 1)[-1]
    return name if ":" in last else f"{name}:latest"


Message = Dict[str, str]
FormatSpec = Union[str, Dict[str, Any], None]


class OllamaClient:
    def __init__(
        self,
        host: Optional[str] = None,
        timeout_s: Optional[float] = None,
        keep_alive: Optional[str] = None,
    ) -> None:
        self.host = host or settings.ollama_host
        self.keep_alive = keep_alive or settings.keep_alive
        self._http = httpx.AsyncClient(
            base_url=self.host,
            timeout=httpx.Timeout(timeout_s or settings.request_timeout_s, connect=5.0),
        )

    # ------------------------------------------------------------------ infra
    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def _unavailable(self, exc: Exception) -> OllamaUnavailableError:
        return OllamaUnavailableError(
            f"Cannot reach Ollama at {self.host} ({exc.__class__.__name__}). "
            "Is the daemon running? Start it with `ollama serve` or the Ollama app."
        )

    @staticmethod
    def _raise_for_status(status: int, body: str, payload: Optional[Dict[str, Any]]) -> None:
        if status < 400:
            return
        try:
            msg = json.loads(body).get("error", body)
        except (ValueError, AttributeError):
            msg = body
        model = (payload or {}).get("model")
        if status == 404 and model:
            raise ModelNotFoundError(f"Model '{model}' is not available locally ({msg}). Run: ollama pull {model}")
        raise OllamaError(f"Ollama HTTP {status}: {msg}")

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = await self._http.request(method, path, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise self._unavailable(exc) from exc
        self._raise_for_status(resp.status_code, resp.text, kwargs.get("json"))
        return resp.json() if resp.content else {}

    # ------------------------------------------------------------- management
    async def version(self) -> str:
        return (await self._request("GET", "/api/version")).get("version", "unknown")

    async def list_models(self) -> List[Dict[str, Any]]:
        return (await self._request("GET", "/api/tags")).get("models", [])

    async def show(self, model: str) -> Dict[str, Any]:
        return await self._request("POST", "/api/show", json={"model": model})

    async def ps(self) -> List[Dict[str, Any]]:
        """Models currently resident in memory (incl. size / size_vram)."""
        return (await self._request("GET", "/api/ps")).get("models", [])

    async def is_available(self, model: str) -> bool:
        wanted = canonical_model_name(model)
        return any(canonical_model_name(m["name"]) == wanted for m in await self.list_models())

    async def load(self, model: str) -> Dict[str, float]:
        """Load a model without generating; returns wall and server load time."""
        t0 = time.perf_counter_ns()
        resp = await self._request(
            "POST", "/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": self.keep_alive},
        )
        return {
            "wall_ms": (time.perf_counter_ns() - t0) / 1e6,
            "load_ms": float(resp.get("load_duration") or 0) / 1e6,
        }

    async def unload(self, model: str, wait: bool = True, timeout_s: float = 30.0) -> None:
        """Evict a model from RAM/VRAM (keep_alive=0) and optionally wait until gone."""
        await self._request("POST", "/api/generate", json={"model": model, "keep_alive": 0})
        if not wait:
            return
        wanted = canonical_model_name(model)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if all(canonical_model_name(m["name"]) != wanted for m in await self.ps()):
                return
            await asyncio.sleep(0.25)

    async def unload_all(self) -> None:
        for m in await self.ps():
            await self.unload(m["name"])

    async def pull(self, model: str) -> AsyncIterator[Dict[str, Any]]:
        """Stream pull progress events ({status, total, completed})."""
        try:
            async with self._http.stream(
                "POST", "/api/pull", json={"model": model, "stream": True}, timeout=None
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    self._raise_for_status(resp.status_code, body, {"model": model})
                async for line in resp.aiter_lines():
                    if line.strip():
                        event = json.loads(line)
                        if "error" in event:
                            raise OllamaError(event["error"])
                        yield event
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise self._unavailable(exc) from exc

    # ------------------------------------------------------------- generation
    async def stream_chat(
        self,
        model: str,
        messages: List[Message],
        *,
        options: Optional[Dict[str, Any]] = None,
        format: FormatSpec = None,
        keep_alive: Optional[str] = None,
    ) -> AsyncIterator[Union[TokenChunk, ChatResult]]:
        """Yield TokenChunk events as they arrive, then a final ChatResult."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options or {},
            "keep_alive": keep_alive if keep_alive is not None else self.keep_alive,
        }
        if format is not None:
            payload["format"] = format

        content: List[str] = []
        thinking: List[str] = []
        arrivals: List[int] = []
        t_first: Optional[int] = None
        final: Optional[Dict[str, Any]] = None

        t_start = time.perf_counter_ns()
        try:
            async with self._http.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    self._raise_for_status(resp.status_code, body, payload)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise OllamaError(chunk["error"])
                    msg = chunk.get("message") or {}
                    piece, think = msg.get("content") or "", msg.get("thinking") or ""
                    if piece or think:
                        now = time.perf_counter_ns()
                        if t_first is None:
                            t_first = now
                        arrivals.append(now)
                        if think:
                            thinking.append(think)
                            yield TokenChunk(think, is_thinking=True)
                        if piece:
                            content.append(piece)
                            yield TokenChunk(piece)
                    if chunk.get("done"):
                        final = chunk
                        break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise self._unavailable(exc) from exc
        t_end = time.perf_counter_ns()

        if final is None:
            raise OllamaError("Ollama stream ended without a final 'done' message")

        metrics = GenerationMetrics.from_ollama(
            model, final, t_start_ns=t_start, t_first_ns=t_first, t_end_ns=t_end, arrivals_ns=arrivals
        )
        yield ChatResult(
            model=model,
            content="".join(content),
            thinking="".join(thinking) or None,
            metrics=metrics,
        )

    async def chat(
        self,
        model: str,
        messages: List[Message],
        *,
        options: Optional[Dict[str, Any]] = None,
        format: FormatSpec = None,
        keep_alive: Optional[str] = None,
    ) -> ChatResult:
        """Non-streaming convenience wrapper (still streams internally for TTFT)."""
        result: Optional[ChatResult] = None
        async for event in self.stream_chat(
            model, messages, options=options, format=format, keep_alive=keep_alive
        ):
            if isinstance(event, ChatResult):
                result = event
        assert result is not None
        return result
