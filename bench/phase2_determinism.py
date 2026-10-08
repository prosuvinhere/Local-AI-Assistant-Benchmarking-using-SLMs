"""Phase 2 - Structure & determinism.

Runs every prompt N times at each temperature (default 0.0 vs 0.7) and logs
output variance. Text prompts measure free-form drift; structured prompts go
through the Pydantic validation + one-retry layer and additionally report
schema validity, retry rate and per-field agreement.

Examples
--------
python -m bench.phase2_determinism --model llama3.2:3b
python -m bench.phase2_determinism --model llama3.2:3b --runs 20 --enforcement prompt
python -m bench.phase2_determinism --model llama3.2:3b --temperatures 0 0.3 0.7 1.0 --seed 7
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import itertools
import json
import statistics
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.metrics import summarize
from app.ollama_client import OllamaClient
from app.output_schemas import get_schema
from app.structured import generate_structured
from bench.common import (
    append_jsonl, capture_environment, env_table, fmt, make_run_dir, md_table, write_json,
)

TICKET_EMAIL = (
    "Extract the support ticket fields from this email:\n\n"
    "From: Priya Raman\nSubject: Payroll export failing - URGENT\n\n"
    "Hi team, since this morning's update the CSV payroll export in LedgerLite Pro crashes "
    "every time. Salaries are due tomorrow and 400 employees won't get paid if this isn't "
    "fixed today. I've already restarted twice. Please call me back ASAP. - Priya"
)

TEXT_PROMPTS: Dict[str, str] = {
    "local_benefits": "Describe the benefits of running language models locally in exactly three sentences.",
    "tagline": "Invent a product name and a one-line tagline for a privacy-first note-taking app.",
}
STRUCTURED_PROMPTS: Dict[str, Tuple[str, str]] = {
    "tomato": ("assistant_answer", "Is a tomato a fruit or a vegetable? Answer briefly."),
    "ticket": ("ticket_extraction", TICKET_EMAIL),
}


# ----------------------------------------------------------- variance stats
def _tokens(s: str) -> set:
    return set(s.lower().split())


def pairwise(outputs: List[str]) -> Dict[str, Optional[float]]:
    pairs = list(itertools.combinations(outputs, 2))
    if not pairs:
        return {"mean_similarity": 1.0, "min_similarity": 1.0, "mean_jaccard": 1.0}
    sims = [difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() for a, b in pairs]
    jac = []
    for a, b in pairs:
        ta, tb = _tokens(a), _tokens(b)
        jac.append(len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0)
    return {"mean_similarity": statistics.fmean(sims), "min_similarity": min(sims),
            "mean_jaccard": statistics.fmean(jac)}


def field_agreement(objs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Per field: share of valid runs whose value equals the modal value."""
    if not objs:
        return {}
    fields = sorted({k for o in objs for k in o})
    out = {}
    for f in fields:
        vals = [json.dumps(o.get(f), sort_keys=True) for o in objs]
        out[f] = Counter(vals).most_common(1)[0][1] / len(vals)
    return out


def variance_summary(outputs: List[str], tokens: List[int], latencies: List[float]) -> Dict[str, Any]:
    counts = Counter(outputs)
    tok = summarize(tokens)
    return {
        "n": len(outputs),
        "unique_outputs": len(counts),
        "exact_match_rate": counts.most_common(1)[0][1] / len(outputs) if outputs else 0.0,
        **pairwise(outputs),
        "tokens_mean": tok.mean if tok else None,
        "tokens_stdev": tok.stdev if tok else None,
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
    }


# --------------------------------------------------------------------- run
async def run_text(client, model, pid, prompt, temp, args, log) -> Dict[str, Any]:
    outputs, tokens, lat = [], [], []
    options = {"temperature": temp, "num_predict": args.max_tokens, "num_ctx": args.num_ctx}
    if args.seed is not None:
        options["seed"] = args.seed
    for i in range(args.runs):
        r = await client.chat(model, [{"role": "user", "content": prompt}], options=options)
        outputs.append(r.content.strip())
        tokens.append(r.metrics.completion_tokens)
        lat.append(r.metrics.total_latency_ms)
        append_jsonl(log, {"prompt_id": pid, "mode": "text", "temperature": temp, "run": i,
                           "output": r.content, "metrics": r.metrics})
        _tick(pid, temp, i + 1, args.runs)
    return {"prompt_id": pid, "mode": "text", "temperature": temp,
            **variance_summary(outputs, tokens, lat), "samples": _samples(outputs)}


async def run_structured(client, model, pid, schema_name, prompt, temp, args, log) -> Dict[str, Any]:
    schema = get_schema(schema_name)
    options = {"temperature": temp, "num_predict": args.max_tokens, "num_ctx": args.num_ctx}
    if args.seed is not None:
        options["seed"] = args.seed
    canon, tokens, lat, valid_objs = [], [], [], []
    successes = first_try = retries = 0
    error_types: Counter = Counter()
    for i in range(args.runs):
        res = await generate_structured(client, model, prompt, schema, schema_name=schema_name,
                                        enforcement=args.enforcement, options=options, lenient=args.lenient)
        successes += res.success
        first_try += res.first_attempt_valid
        retries += res.retried
        for a in res.attempts:
            if a.error_type:
                error_types[a.error_type] += 1
        if res.success:
            valid_objs.append(res.data)
            canon.append(json.dumps(res.data, sort_keys=True))
        else:
            canon.append(res.attempts[-1].raw_output.strip())
        tokens.append(sum(a.metrics.completion_tokens for a in res.attempts))
        lat.append(res.total_latency_ms)
        append_jsonl(log, {"prompt_id": pid, "mode": "structured", "temperature": temp, "run": i,
                           "result": res})
        _tick(pid, temp, i + 1, args.runs)
    agreement = field_agreement(valid_objs)
    return {
        "prompt_id": pid, "mode": "structured", "schema": schema_name, "temperature": temp,
        **variance_summary(canon, tokens, lat),
        "valid_rate": successes / args.runs,
        "first_try_valid_rate": first_try / args.runs,
        "retry_rate": retries / args.runs,
        "error_types": dict(error_types),
        "field_agreement": agreement,
        "mean_field_agreement": statistics.fmean(agreement.values()) if agreement else None,
        "samples": _samples(canon),
    }


def _samples(outputs: List[str], k: int = 2) -> List[str]:
    return [o for o, _ in Counter(outputs).most_common(k)]


def _tick(pid: str, temp: float, done: int, total: int) -> None:
    sys.stdout.write(f"\r  {pid:<16} T={temp:<4} {done:>3}/{total}   ")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


# ------------------------------------------------------------------ report
def _pct(v: Optional[float]) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


def render_report(env, model, args, results: List[Dict[str, Any]]) -> str:
    rows = []
    for r in results:
        rows.append([
            r["prompt_id"], r["mode"], r["temperature"],
            _pct(r.get("valid_rate")), _pct(r.get("first_try_valid_rate")), _pct(r.get("retry_rate")),
            f"{r['unique_outputs']}/{r['n']}", _pct(r["exact_match_rate"]),
            fmt(r["mean_similarity"], 3), fmt(r["min_similarity"], 3), fmt(r["mean_jaccard"], 3),
            _pct(r.get("mean_field_agreement")),
            f"{fmt(r['tokens_mean'], 0)} ± {fmt(r['tokens_stdev'], 0)}",
            fmt(r["latency_ms_mean"], 0),
        ])

    deltas = []
    by_prompt: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    for pid, rs in by_prompt.items():
        rs = sorted(rs, key=lambda x: x["temperature"])
        lo, hi = rs[0], rs[-1]
        deltas.append([pid, f"{lo['temperature']} → {hi['temperature']}",
                       f"{lo['unique_outputs']} → {hi['unique_outputs']}",
                       f"{fmt(lo['mean_similarity'], 3)} → {fmt(hi['mean_similarity'], 3)}",
                       f"{_pct(lo.get('valid_rate'))} → {_pct(hi.get('valid_rate'))}"
                       if lo["mode"] == "structured" else "n/a"])

    field_rows = []
    for r in results:
        for f, v in (r.get("field_agreement") or {}).items():
            field_rows.append([r["prompt_id"], r["temperature"], f, _pct(v)])

    samples = []
    for r in results:
        samples.append(f"<details><summary>{r['prompt_id']} @ T={r['temperature']} "
                       f"({r['unique_outputs']} unique)</summary>\n")
        for i, s in enumerate(r["samples"], 1):
            samples.append(f"**Variant {i}**\n```\n{s[:800]}\n```")
        samples.append("</details>\n")

    return "\n".join([
        f"# Phase 2 - Structure & Determinism Report (`{model}`)", "",
        "## Environment", env_table(env), "",
        "## Methodology",
        f"- {args.runs} runs per (prompt, temperature); temperatures = {args.temperatures}.",
        f"- Seed: {args.seed if args.seed is not None else 'not fixed (Ollama picks a random seed per request)'}.",
        f"- Structured enforcement: `{args.enforcement}` (schema = grammar-constrained, json = JSON mode, "
        "prompt = instructions only). Validation is Pydantic **strict** mode; invalid output gets exactly one "
        "repair re-prompt, then fails gracefully.",
        "- **Exact-match** = share of runs identical to the most common output.",
        "- **Similarity** = mean / min pairwise `difflib` ratio (1.0 = identical). **Jaccard** = word-set overlap.",
        "- **Field agreement** = per JSON field, share of valid runs equal to the modal value (averaged).", "",
        "## Results",
        md_table(["Prompt", "Mode", "Temp", "Valid", "1st-try valid", "Retried", "Unique",
                  "Exact-match", "Mean sim", "Min sim", "Jaccard", "Field agree", "Out tokens",
                  "Latency (ms)"], rows), "",
        "## Temperature effect",
        md_table(["Prompt", "Temp", "Unique outputs", "Mean similarity", "Validity"], deltas), "",
        "## Per-field agreement (structured)",
        md_table(["Prompt", "Temp", "Field", "Agreement"], field_rows) if field_rows else "_none_", "",
        "## Interpretation guide",
        "- At T=0.0 decoding is greedy: expect 1 unique output. Residual variation there comes from "
        "non-deterministic GPU kernels / batching, not sampling.",
        "- At T=0.7 free text diverges quickly; enum / boolean JSON fields should stay stable while "
        "free-text fields (summaries, key_points) drift. Low agreement on an *enum* field signals an "
        "ambiguous prompt or a weak model.",
        "- A fixed `--seed` makes T>0 reproducible run-to-run on the same machine/build.", "",
        "## Sample outputs", *samples,
    ]) + "\n"


async def main(args: argparse.Namespace) -> int:
    async with OllamaClient() as client:
        if not await client.is_available(args.model):
            print(f"Model {args.model} not pulled. Run: ollama pull {args.model}")
            return 1
        env = await capture_environment(client)
        out_dir = make_run_dir("phase2")
        log = out_dir / "runs.jsonl"
        await client.chat(args.model, [{"role": "user", "content": "hi"}], options={"num_predict": 1})  # warm load

        results = []
        if args.mode in ("text", "both"):
            for pid, prompt in TEXT_PROMPTS.items():
                for t in args.temperatures:
                    results.append(await run_text(client, args.model, pid, prompt, t, args, log))
        if args.mode in ("structured", "both"):
            for pid, (schema_name, prompt) in STRUCTURED_PROMPTS.items():
                for t in args.temperatures:
                    results.append(await run_structured(client, args.model, pid, schema_name, prompt, t, args, log))

    md = render_report(env, args.model, args, results)
    (out_dir / "report.md").write_text(md)
    write_json(out_dir / "summary.json", {"environment": env, "args": vars(args), "results": results})
    print("\n" + md.split("## Sample outputs")[0])
    print(f"Saved to {out_dir}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.7])
    p.add_argument("--mode", choices=["text", "structured", "both"], default="both")
    p.add_argument("--enforcement", choices=["schema", "json", "prompt"], default="schema")
    p.add_argument("--lenient", action="store_true", help="Salvage JSON embedded in prose")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--num-ctx", type=int, default=4096)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
