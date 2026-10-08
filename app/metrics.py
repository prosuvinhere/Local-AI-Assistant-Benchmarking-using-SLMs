"""Per-request performance metrics and robust summary statistics.

Two independent clocks are captured for every generation:

* **Client-side wall clock** (``time.perf_counter_ns``): what a user actually
  experiences - includes HTTP, JSON decoding, scheduling and model load.
* **Server-side counters** reported by Ollama in the final stream chunk
  (``load_duration``, ``prompt_eval_duration``, ``eval_duration`` ...). These
  isolate pure compute and are the right numbers for comparing models.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel

NS_PER_MS = 1_000_000.0

# Two-sided 95% Student-t critical values (df = n - 1). Benchmarks typically
# have small n, where the normal approximation (1.96) understates uncertainty.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042,
}


def _t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    if df < 30:
        # use the next-smaller tabulated df (conservative)
        return _T95[max(k for k in _T95 if k <= df)]
    return 1.96


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (matches numpy's default method)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * (q / 100.0)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return data[int(pos)]
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


class Summary(BaseModel):
    n: int
    mean: float
    median: float
    stdev: float
    min: float
    max: float
    p90: float
    p95: float
    p99: float
    ci95: float  # half-width of the 95% CI of the mean
    cv: float  # coefficient of variation (stdev / mean) - run-to-run stability

    def fmt(self, digits: int = 1) -> str:
        return f"{self.mean:.{digits}f} ± {self.ci95:.{digits}f}"


def summarize(values: Iterable[Optional[float]]) -> Optional[Summary]:
    data = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not data:
        return None
    n = len(data)
    mean = statistics.fmean(data)
    stdev = statistics.stdev(data) if n > 1 else 0.0
    ci95 = _t_crit(n - 1) * stdev / math.sqrt(n) if n > 1 else 0.0
    return Summary(
        n=n,
        mean=mean,
        median=statistics.median(data),
        stdev=stdev,
        min=min(data),
        max=max(data),
        p90=percentile(data, 90),
        p95=percentile(data, 95),
        p99=percentile(data, 99),
        ci95=ci95,
        cv=(stdev / mean) if mean else 0.0,
    )


def _rate(tokens: int, ms: float) -> Optional[float]:
    return tokens / (ms / 1000.0) if tokens > 0 and ms > 0 else None


class GenerationMetrics(BaseModel):
    model: str

    # --- client-side wall clock -------------------------------------------
    ttft_ms: Optional[float] = None  # request sent -> first token received
    total_latency_ms: float = 0.0  # request sent -> final chunk received
    mean_itl_ms: Optional[float] = None  # inter-token latency (stream chunk gaps)
    p95_itl_ms: Optional[float] = None
    stream_chunks: int = 0

    # --- server-side (Ollama-reported) ------------------------------------
    prompt_tokens: int = 0
    completion_tokens: int = 0
    load_ms: float = 0.0  # >0 means the model was (re)loaded: a cold start
    prompt_eval_ms: float = 0.0  # prefill time
    eval_ms: float = 0.0  # decode time
    server_total_ms: float = 0.0
    done_reason: Optional[str] = None

    # --- derived ----------------------------------------------------------
    server_ttft_ms: Optional[float] = None  # load + prefill (compute-only TTFT)
    client_overhead_ms: Optional[float] = None  # wall - server total (HTTP etc.)
    prompt_tps: Optional[float] = None  # prefill throughput
    decode_tps: Optional[float] = None  # generation throughput (the headline tok/s)
    e2e_tps: Optional[float] = None  # completion tokens / wall latency

    @classmethod
    def from_ollama(
        cls,
        model: str,
        final: Dict[str, Any],
        *,
        t_start_ns: int,
        t_first_ns: Optional[int],
        t_end_ns: int,
        arrivals_ns: List[int],
    ) -> "GenerationMetrics":
        def ms(key: str) -> float:
            return float(final.get(key) or 0) / NS_PER_MS

        prompt_tokens = int(final.get("prompt_eval_count") or 0)
        completion_tokens = int(final.get("eval_count") or 0)
        load_ms, prompt_eval_ms, eval_ms = ms("load_duration"), ms("prompt_eval_duration"), ms("eval_duration")
        server_total = ms("total_duration")
        wall = (t_end_ns - t_start_ns) / NS_PER_MS
        itl = [(b - a) / NS_PER_MS for a, b in zip(arrivals_ns, arrivals_ns[1:])]

        return cls(
            model=model,
            ttft_ms=(t_first_ns - t_start_ns) / NS_PER_MS if t_first_ns else None,
            total_latency_ms=wall,
            mean_itl_ms=statistics.fmean(itl) if itl else None,
            p95_itl_ms=percentile(itl, 95) if itl else None,
            stream_chunks=len(arrivals_ns),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            load_ms=load_ms,
            prompt_eval_ms=prompt_eval_ms,
            eval_ms=eval_ms,
            server_total_ms=server_total,
            done_reason=final.get("done_reason"),
            server_ttft_ms=load_ms + prompt_eval_ms,
            client_overhead_ms=(wall - server_total) if server_total else None,
            prompt_tps=_rate(prompt_tokens, prompt_eval_ms),
            decode_tps=_rate(completion_tokens, eval_ms),
            e2e_tps=_rate(completion_tokens, wall),
        )
