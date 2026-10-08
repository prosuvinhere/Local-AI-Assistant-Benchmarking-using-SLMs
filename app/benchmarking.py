"""Reusable latency benchmark engine (used by the API and Phase 1 CLI).

Methodology baked in:
* Warm-up iterations are discarded (first call pays model load + Metal/CUDA
  kernel compilation).
* Cold start is measured separately, after explicitly unloading the model.
* Ollama caches the KV state of a matching prompt *prefix*. Repeating an
  identical prompt therefore makes prefill (and TTFT) look near-free. With
  ``cache_bust=True`` a unique nonce is prepended to the first message so each
  iteration pays a realistic prefill cost.
* Requests are sequential by default (concurrency=1) so that per-request
  latency is not polluted by queueing.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from .metrics import GenerationMetrics, Summary, summarize
from .ollama_client import OllamaClient

ProgressFn = Callable[[int, int, GenerationMetrics], None]


class LatencyReport(BaseModel):
    model: str
    prompt_label: str
    iterations: int
    warmup: int
    concurrency: int
    cache_bust: bool
    options: Dict[str, Any]
    cold_start: Optional[GenerationMetrics] = None
    runs: List[GenerationMetrics]
    ttft_ms: Optional[Summary] = None
    server_ttft_ms: Optional[Summary] = None
    total_latency_ms: Optional[Summary] = None
    decode_tps: Optional[Summary] = None
    prompt_tps: Optional[Summary] = None
    e2e_tps: Optional[Summary] = None
    itl_ms: Optional[Summary] = None
    prompt_tokens: Optional[Summary] = None
    completion_tokens: Optional[Summary] = None
    aggregate_tps: Optional[Summary] = None  # only meaningful when concurrency > 1


def with_nonce(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs = [dict(m) for m in messages]
    msgs[0]["content"] = f"[run {uuid.uuid4().hex[:8]}]\n" + msgs[0]["content"]
    return msgs


async def run_latency_benchmark(
    client: OllamaClient,
    model: str,
    messages: List[Dict[str, str]],
    options: Dict[str, Any],
    *,
    iterations: int = 10,
    warmup: int = 2,
    cold_start: bool = False,
    cache_bust: bool = True,
    concurrency: int = 1,
    prompt_label: str = "custom",
    on_progress: Optional[ProgressFn] = None,
) -> LatencyReport:
    def prep() -> List[Dict[str, str]]:
        return with_nonce(messages) if cache_bust else messages

    cold: Optional[GenerationMetrics] = None
    if cold_start:
        await client.unload(model)
        cold = (await client.chat(model, prep(), options=options)).metrics

    for _ in range(warmup):
        await client.chat(model, prep(), options=options)

    runs: List[GenerationMetrics] = []
    aggregate: List[float] = []
    done = 0
    while done < iterations:
        batch = min(concurrency, iterations - done)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(client.chat(model, prep(), options=options) for _ in range(batch)))
        wall_s = time.perf_counter() - t0
        runs.extend(r.metrics for r in results)
        if concurrency > 1:
            aggregate.append(sum(r.metrics.completion_tokens for r in results) / wall_s)
        done += batch
        if on_progress:
            on_progress(done, iterations, results[-1].metrics)

    return LatencyReport(
        model=model,
        prompt_label=prompt_label,
        iterations=iterations,
        warmup=warmup,
        concurrency=concurrency,
        cache_bust=cache_bust,
        options=options,
        cold_start=cold,
        runs=runs,
        ttft_ms=summarize(r.ttft_ms for r in runs),
        server_ttft_ms=summarize(r.server_ttft_ms for r in runs),
        total_latency_ms=summarize(r.total_latency_ms for r in runs),
        decode_tps=summarize(r.decode_tps for r in runs),
        prompt_tps=summarize(r.prompt_tps for r in runs),
        e2e_tps=summarize(r.e2e_tps for r in runs),
        itl_ms=summarize(r.mean_itl_ms for r in runs),
        prompt_tokens=summarize(r.prompt_tokens for r in runs),
        completion_tokens=summarize(r.completion_tokens for r in runs),
        aggregate_tps=summarize(aggregate) if aggregate else None,
    )
