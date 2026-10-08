"""Phase 3 - Automated multi-model comparison (+ quantization sweeps).

For each model, in isolation:
  1. unload everything, let memory settle
  2. cold-load the model while sampling memory (Ollama /api/ps + RSS + system)
  3. one warm-up generation
  4. run the standardized prompt suite (bench/prompts.json) with fixed
     temperature / seed / num_ctx, scoring every answer with objective checks
  5. unload
Then writes a technical report (report.md), raw data (results.json) and a
flat CSV (summary.csv).

Examples
--------
python -m bench.phase3_compare                                   # llama3.2:3b, phi4-mini, mistral:7b
python -m bench.phase3_compare --models llama3.2:3b phi4-mini --repeats 3 --allow-code-exec
# quantization sweep: compare quants of one model against a high-precision baseline
python -m bench.phase3_compare --baseline llama3.2:3b-instruct-q8_0 \
    --models llama3.2:3b-instruct-q4_K_M llama3.2:3b-instruct-q5_K_M llama3.2:3b-instruct-q8_0
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import difflib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.metrics import GenerationMetrics, summarize
from app.ollama_client import OllamaClient, OllamaError, canonical_model_name
from app.output_schemas import get_schema
from app.structured import generate_structured
from bench.common import (
    PROJECT_ROOT, capture_environment, env_table, environment_warnings, fmt, fmt_bytes,
    make_run_dir, md_table, write_json,
)
from bench.memory import MemorySampler, wait_for_memory_settle
from bench.quality import QualityResult, score_response

DEFAULT_MODELS = ["llama3.2:3b", "phi4-mini", "mistral:7b"]


class PromptRun(BaseModel):
    prompt_id: str
    category: str
    repeat: int
    content: str
    generations: List[GenerationMetrics]
    quality: QualityResult
    structured_success: Optional[bool] = None
    structured_first_try: Optional[bool] = None
    structured_retried: Optional[bool] = None


class ModelResult(BaseModel):
    model: str
    status: Literal["ok", "skipped", "error"] = "ok"
    error: Optional[str] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None
    disk_bytes: Optional[int] = None
    loaded_bytes: Optional[int] = None
    vram_bytes: Optional[int] = None
    context_length: Optional[int] = None
    cold_load_ms: Optional[float] = None
    rss_peak_bytes: Optional[int] = None
    rss_delta_bytes: Optional[int] = None
    sys_delta_bytes: Optional[int] = None
    wall_s: float = 0.0
    runs: List[PromptRun] = Field(default_factory=list)


# ------------------------------------------------------------- execution
async def _pull(client: OllamaClient, model: str) -> None:
    print(f"  pulling {model} ...")
    last = -1
    async for ev in client.pull(model):
        if ev.get("total") and ev.get("completed"):
            pct = int(ev["completed"] * 100 / ev["total"])
            if pct // 10 != last // 10:
                print(f"    {ev.get('status', '')[:30]:<30} {pct:>3}%")
                last = pct
    print("  pull complete")


async def run_model(client: OllamaClient, model: str, suite: List[Dict[str, Any]],
                    args: argparse.Namespace) -> ModelResult:
    mr = ModelResult(model=model)
    local = {canonical_model_name(m["name"]): m for m in await client.list_models()}
    if canonical_model_name(model) not in local:
        if not args.pull:
            mr.status, mr.error = "skipped", f"not pulled (run `ollama pull {model}` or pass --pull)"
            return mr
        await _pull(client, model)
        local = {canonical_model_name(m["name"]): m for m in await client.list_models()}

    info = local[canonical_model_name(model)]
    details = info.get("details") or {}
    mr.family, mr.parameter_size = details.get("family"), details.get("parameter_size")
    mr.quantization, mr.disk_bytes = details.get("quantization_level"), info.get("size")

    options = {"temperature": args.temperature, "seed": args.seed,
               "num_predict": args.max_tokens, "num_ctx": args.num_ctx}
    t_model = time.perf_counter()

    await client.unload_all()
    wait_for_memory_settle(args.settle_s)

    with MemorySampler() as mem:
        load = await client.load(model)
        mr.cold_load_ms = load["wall_ms"]
        # Allocations are finalised on first generation (KV cache for num_ctx).
        await client.chat(model, [{"role": "user", "content": "Say OK."}],
                          options={**options, "num_predict": 8})
        for entry in await client.ps():
            if canonical_model_name(entry["name"]) == canonical_model_name(model):
                mr.loaded_bytes, mr.vram_bytes = entry.get("size"), entry.get("size_vram")
                mr.context_length = entry.get("context_length")

        for item in suite:
            for rep in range(args.repeats):
                mr.runs.append(await _run_prompt(client, model, item, rep, options, args))
                last = mr.runs[-1]
                g = last.generations[-1]
                print(f"  {item['id']:<22} rep{rep}  quality={last.quality.score:.2f}  "
                      f"ttft={fmt(g.ttft_ms, 0)}ms  decode={fmt(g.decode_tps)} tok/s")

    mr.rss_peak_bytes, mr.rss_delta_bytes, mr.sys_delta_bytes = mem.peak_rss, mem.rss_delta, mem.sys_delta
    mr.wall_s = time.perf_counter() - t_model
    if not args.keep_loaded:
        await client.unload(model)
    return mr


async def _run_prompt(client, model, item, rep, options, args) -> PromptRun:
    if item.get("schema"):
        res = await generate_structured(
            client, model, item["prompt"], get_schema(item["schema"]), schema_name=item["schema"],
            enforcement=args.json_enforcement, options=options,
        )
        content = json.dumps(res.data) if res.success else res.attempts[-1].raw_output
        quality = score_response(content, item["checks"], parsed=res.data, structured_ok=res.success,
                                 structured_error=res.error)
        return PromptRun(prompt_id=item["id"], category=item["category"], repeat=rep, content=content,
                         generations=[a.metrics for a in res.attempts], quality=quality,
                         structured_success=res.success, structured_first_try=res.first_attempt_valid,
                         structured_retried=res.retried)
    r = await client.chat(model, [{"role": "user", "content": item["prompt"]}], options=options)
    quality = score_response(r.content.strip(), item["checks"], allow_code_exec=args.allow_code_exec)
    return PromptRun(prompt_id=item["id"], category=item["category"], repeat=rep, content=r.content,
                     generations=[r.metrics], quality=quality)


# --------------------------------------------------------------- analysis
def summarize_model(mr: ModelResult) -> Dict[str, Any]:
    gens = [g for r in mr.runs for g in r.generations]
    per_cat: Dict[str, List[float]] = {}
    for r in mr.runs:
        per_cat.setdefault(r.category, []).append(r.quality.score)
    structured = [r for r in mr.runs if r.structured_success is not None]
    quality = statistics.fmean(r.quality.score for r in mr.runs) if mr.runs else None
    mem_gb = (mr.loaded_bytes or 0) / 1e9 or None
    decode = summarize(g.decode_tps for g in gens)
    return {
        "model": mr.model,
        "quality": quality,
        "quality_by_category": {k: statistics.fmean(v) for k, v in per_cat.items()},
        "ttft_ms": summarize(g.ttft_ms for g in gens),
        "decode_tps": decode,
        "prefill_tps": summarize(g.prompt_tps for g in gens),
        "latency_ms": summarize(sum(g.total_latency_ms for g in r.generations) for r in mr.runs),
        "out_tokens": summarize(g.completion_tokens for g in gens),
        "json_valid_rate": statistics.fmean(bool(r.structured_success) for r in structured) if structured else None,
        "json_first_try_rate": statistics.fmean(bool(r.structured_first_try) for r in structured) if structured else None,
        "json_retry_rate": statistics.fmean(bool(r.structured_retried) for r in structured) if structured else None,
        "mem_gb": mem_gb,
        "quality_per_gb": (quality / mem_gb) if (quality is not None and mem_gb) else None,
        "tps_per_gb": (decode.mean / mem_gb) if (decode and mem_gb) else None,
    }


def pareto_front(summaries: List[Dict[str, Any]]) -> List[str]:
    """Models not dominated on (quality ↑, decode tok/s ↑, memory ↓)."""
    pts = [(s["model"], s["quality"] or 0, s["decode_tps"].mean if s["decode_tps"] else 0, s["mem_gb"] or 1e9)
           for s in summaries]
    front = []
    for m, q, t, g in pts:
        dominated = any(
            (q2 >= q and t2 >= t and g2 <= g) and (q2 > q or t2 > t or g2 < g)
            for m2, q2, t2, g2 in pts if m2 != m
        )
        if not dominated:
            front.append(m)
    return front


def baseline_similarity(base: ModelResult, other: ModelResult) -> Optional[float]:
    b = {r.prompt_id: r.content for r in base.runs if r.repeat == 0}
    sims = [difflib.SequenceMatcher(None, b[r.prompt_id], r.content, autojunk=False).ratio()
            for r in other.runs if r.repeat == 0 and r.prompt_id in b]
    return statistics.fmean(sims) if sims else None


# ----------------------------------------------------------------- report
def _pct(v: Optional[float]) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


def render_report(env, args, results: List[ModelResult], suite: List[Dict[str, Any]]) -> str:
    ok = [m for m in results if m.status == "ok" and m.runs]
    sums = {m.model: summarize_model(m) for m in ok}
    S = list(sums.values())
    cats = sorted({p["category"] for p in suite})

    out = ["# SLM Benchmark - Technical Report", "",
           f"_Models: {', '.join(m.model for m in results)} · Suite: {len(suite)} prompts × "
           f"{args.repeats} repeat(s)_", ""]

    # Executive summary
    out += ["## Executive summary"]
    if S:
        def best(key, fn, reverse=True):
            cand = [s for s in S if fn(s) is not None]
            return sorted(cand, key=fn, reverse=reverse)[0] if cand else None
        bq = best("q", lambda s: s["quality"])
        bt = best("t", lambda s: s["decode_tps"].mean if s["decode_tps"] else None)
        bf = best("f", lambda s: s["ttft_ms"].median if s["ttft_ms"] else None, reverse=False)
        bm = best("m", lambda s: s["mem_gb"], reverse=False)
        be = best("e", lambda s: s["quality_per_gb"])
        if bq: out.append(f"- **Highest quality:** `{bq['model']}` ({_pct(bq['quality'])} of checks passed)")
        if bt: out.append(f"- **Fastest generation:** `{bt['model']}` ({bt['decode_tps'].mean:.1f} tok/s)")
        if bf: out.append(f"- **Lowest TTFT (p50):** `{bf['model']}` ({bf['ttft_ms'].median:.0f} ms)")
        if bm: out.append(f"- **Smallest memory footprint:** `{bm['model']}` ({bm['mem_gb']:.2f} GB loaded)")
        if be: out.append(f"- **Best quality per GB:** `{be['model']}` ({be['quality_per_gb']:.2f} /GB)")
        out.append(f"- **Pareto-optimal (quality ↑, speed ↑, memory ↓):** "
                   f"{', '.join(f'`{m}`' for m in pareto_front(S))}")
    skipped = [m for m in results if m.status != "ok"]
    for m in skipped:
        out.append(f"- ⚠️ `{m.model}` {m.status}: {m.error}")
    out.append("")

    out += ["## 1. Test environment", env_table(env), ""]
    warns = environment_warnings(env)
    if warns:
        out += [f"> ⚠️ {w}" for w in warns] + [""]

    out += ["## 2. Methodology",
            f"- Decoding: temperature={args.temperature}, seed={args.seed}, num_predict={args.max_tokens}, "
            f"num_ctx={args.num_ctx} (identical for all models so KV-cache memory is comparable).",
            "- Each model is benchmarked **in isolation**: all models unloaded, memory allowed to settle, "
            "cold-loaded, one warm-up generation, then the suite.",
            "- Quality = fraction of objective checks passed (exact facts, numeric answers, word/line limits, "
            "regex format rules, schema validity, JSON field correctness"
            + (", unit tests on generated code" if args.allow_code_exec else
               "; generated code is only syntax/definition-checked - pass `--allow-code-exec` to run unit tests")
            + ").",
            f"- Structured prompts use enforcement=`{args.json_enforcement}` with Pydantic strict validation "
            "and exactly one repair retry.",
            "- **Loaded memory** = Ollama `/api/ps` size (weights + KV cache + buffers): the authoritative "
            "figure. RSS is a lower bound on Apple Silicon (Metal/mmap pages are not fully attributed).", ""]

    out += ["## 3. Model inventory",
            md_table(["Model", "Family", "Params", "Quant", "Disk", "Context"],
                     [[m.model, m.family, m.parameter_size, m.quantization, fmt_bytes(m.disk_bytes),
                       m.context_length] for m in results if m.status == "ok"]), ""]

    out += ["## 4. Memory",
            md_table(["Model", "Loaded (Ollama)", "On GPU", "Cold load (ms)", "Peak Ollama RSS",
                      "RSS Δ", "System mem Δ"],
                     [[m.model, fmt_bytes(m.loaded_bytes),
                       _pct(m.vram_bytes / m.loaded_bytes) if m.loaded_bytes and m.vram_bytes is not None else "–",
                       fmt(m.cold_load_ms, 0), fmt_bytes(m.rss_peak_bytes), fmt_bytes(m.rss_delta_bytes),
                       fmt_bytes(m.sys_delta_bytes)] for m in ok]), ""]

    out += ["## 5. Speed",
            md_table(["Model", "TTFT p50 (ms)", "TTFT p95 (ms)", "Decode tok/s (mean ± CI95)", "Decode CV",
                      "Prefill tok/s p50", "Latency/prompt p50 (ms)", "Latency/prompt p95 (ms)", "Avg out tokens"],
                     [[s["model"],
                       fmt(s["ttft_ms"].median if s["ttft_ms"] else None, 0),
                       fmt(s["ttft_ms"].p95 if s["ttft_ms"] else None, 0),
                       s["decode_tps"].fmt() if s["decode_tps"] else "–",
                       _pct(s["decode_tps"].cv) if s["decode_tps"] else "–",
                       fmt(s["prefill_tps"].median if s["prefill_tps"] else None, 0),
                       fmt(s["latency_ms"].median if s["latency_ms"] else None, 0),
                       fmt(s["latency_ms"].p95 if s["latency_ms"] else None, 0),
                       fmt(s["out_tokens"].mean if s["out_tokens"] else None, 0)] for s in S]), "",
            "_Latency/prompt includes the repair retry for structured prompts and scales with answer "
            "length - compare decode tok/s for raw speed._", ""]

    out += ["## 6. Output quality",
            md_table(["Model", "Overall"] + cats + ["JSON valid", "JSON 1st-try", "JSON retried"],
                     [[s["model"], f"**{_pct(s['quality'])}**"]
                      + [_pct(s["quality_by_category"].get(c)) for c in cats]
                      + [_pct(s["json_valid_rate"]), _pct(s["json_first_try_rate"]), _pct(s["json_retry_rate"])]
                      for s in S]), ""]

    out += ["## 7. Efficiency",
            md_table(["Model", "Quality / GB", "Decode tok/s / GB", "Quality × tok/s"],
                     [[s["model"], fmt(s["quality_per_gb"], 2), fmt(s["tps_per_gb"], 1),
                       fmt((s["quality"] or 0) * (s["decode_tps"].mean if s["decode_tps"] else 0), 1)]
                      for s in S]), ""]

    if args.baseline:
        base = next((m for m in ok if canonical_model_name(m.model) == canonical_model_name(args.baseline)), None)
        out += ["## 8. Comparison vs baseline (quantization trade-off)"]
        if base is None:
            out += [f"_Baseline `{args.baseline}` was not benchmarked successfully._", ""]
        else:
            b = sums[base.model]
            rows = []
            for m in ok:
                s = sums[m.model]
                rows.append([
                    m.model, m.quantization,
                    fmt(s["decode_tps"].mean / b["decode_tps"].mean, 2, "×") if s["decode_tps"] and b["decode_tps"] else "–",
                    fmt(b["ttft_ms"].median / s["ttft_ms"].median, 2, "×") if s["ttft_ms"] and b["ttft_ms"] else "–",
                    fmt(s["mem_gb"] / b["mem_gb"] * 100, 0, "%") if s["mem_gb"] and b["mem_gb"] else "–",
                    fmt(s["quality"] / b["quality"] * 100, 0, "%") if b["quality"] else "–",
                    fmt(baseline_similarity(base, m), 3),
                ])
            out += [f"Baseline: `{base.model}` ({base.quantization}).",
                    md_table(["Model", "Quant", "Decode speedup", "TTFT speedup", "Memory vs base",
                              "Quality retained", "Output similarity to base"], rows),
                    "", "_Output similarity = mean difflib ratio of greedy (T=0) answers vs the baseline's; "
                    "it measures how far quantization drifts the model's behaviour even when checks still pass._", ""]

    # Failures
    out += ["## Appendix A. Per-prompt quality matrix"]
    pids = [p["id"] for p in suite]
    matrix = []
    for pid in pids:
        row = [pid]
        for m in ok:
            scores = [r.quality.score for r in m.runs if r.prompt_id == pid]
            row.append(_pct(statistics.fmean(scores)) if scores else "–")
        matrix.append(row)
    out += [md_table(["Prompt"] + [m.model for m in ok], matrix), ""]

    out += ["## Appendix B. Failed checks"]
    fails = []
    for m in ok:
        for r in m.runs:
            for c in r.quality.failed:
                fails.append([m.model, r.prompt_id, r.repeat, c.type, c.detail.replace("|", "\\|")[:120]])
    out += [md_table(["Model", "Prompt", "Rep", "Check", "Detail"], fails) if fails else "_No failed checks._", ""]
    return "\n".join(out) + "\n"


def write_csv(path: Path, results: List[ModelResult]) -> None:
    cols = ["model", "quantization", "parameter_size", "loaded_gb", "cold_load_ms", "ttft_p50_ms",
            "ttft_p95_ms", "decode_tps_mean", "decode_tps_ci95", "prefill_tps_p50", "quality",
            "json_valid_rate", "quality_per_gb"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in results:
            if m.status != "ok" or not m.runs:
                continue
            s = summarize_model(m)
            w.writerow([m.model, m.quantization, m.parameter_size, s["mem_gb"], m.cold_load_ms,
                        s["ttft_ms"].median if s["ttft_ms"] else None, s["ttft_ms"].p95 if s["ttft_ms"] else None,
                        s["decode_tps"].mean if s["decode_tps"] else None,
                        s["decode_tps"].ci95 if s["decode_tps"] else None,
                        s["prefill_tps"].median if s["prefill_tps"] else None, s["quality"],
                        s["json_valid_rate"], s["quality_per_gb"]])


async def main(args: argparse.Namespace) -> int:
    suite_doc = json.loads(Path(args.prompts).read_text())
    suite = [p for p in suite_doc["prompts"] if not args.categories or p["category"] in args.categories]
    if args.limit:
        suite = suite[: args.limit]
    models = list(args.models)
    if args.baseline and canonical_model_name(args.baseline) not in {canonical_model_name(m) for m in models}:
        models.insert(0, args.baseline)

    async with OllamaClient() as client:
        env = await capture_environment(client)
        for w in environment_warnings(env):
            print(f"WARNING: {w}")
        out_dir = make_run_dir("phase3")
        results: List[ModelResult] = []
        for model in models:
            print(f"\n=== {model} ===")
            try:
                mr = await run_model(client, model, suite, args)
            except OllamaError as exc:
                mr = ModelResult(model=model, status="error", error=str(exc))
            if mr.status != "ok":
                print(f"  {mr.status}: {mr.error}")
            results.append(mr)
            write_json(out_dir / "results.json", {"environment": env, "args": vars(args), "models": results})

    md = render_report(env, args, results, suite)
    (out_dir / "report.md").write_text(md)
    write_csv(out_dir / "summary.csv", results)
    print("\n" + md.split("## Appendix A")[0])
    print(f"Saved to {out_dir}")
    return 0 if any(m.status == "ok" for m in results) else 1


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--baseline", help="Reference model for relative speed/memory/quality (quant sweeps)")
    p.add_argument("--prompts", default=str(PROJECT_ROOT / "bench" / "prompts.json"))
    p.add_argument("--categories", nargs="*", help="Only run these categories")
    p.add_argument("--limit", type=int, help="Only the first N prompts (smoke tests)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--json-enforcement", choices=["schema", "json", "prompt"], default="prompt",
                   help="'prompt' (default) measures each model's native JSON ability; retries are counted")
    p.add_argument("--allow-code-exec", action="store_true",
                   help="Execute model-written code in a subprocess to run unit tests")
    p.add_argument("--pull", action="store_true", help="Pull missing models automatically")
    p.add_argument("--keep-loaded", action="store_true", help="Do not unload after each model")
    p.add_argument("--settle-s", type=float, default=2.0)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
