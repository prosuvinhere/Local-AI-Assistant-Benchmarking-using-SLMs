"""Phase 1 - Local inference & measurement.

Measures TTFT, decode tokens/sec, prefill tokens/sec, inter-token latency and
end-to-end latency for one or more models across prompts of different input
lengths (short / medium / long) so prefill and decode costs can be separated.

Examples
--------
python -m bench.phase1_latency --model llama3.2:3b
python -m bench.phase1_latency --model llama3.2:3b --model phi4-mini --iterations 20 --cold-start
python -m bench.phase1_latency --model llama3.2:3b --prompt-set long --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Dict, List

from app.benchmarking import LatencyReport, run_latency_benchmark
from app.ollama_client import ModelNotFoundError, OllamaClient, OllamaError
from bench.common import (
    append_jsonl, capture_environment, env_table, environment_warnings, fmt,
    make_run_dir, md_table, write_json,
)

_PASSAGE = (
    "Small language models (SLMs) with one to eight billion parameters can now run on consumer "
    "laptops thanks to 4-bit and 5-bit weight quantization, memory-mapped model files and GPU "
    "back-ends such as Apple Metal and CUDA. Running inference locally removes network latency, "
    "keeps sensitive data on the device and eliminates per-token API costs. The trade-offs are "
    "limited context windows, lower reasoning ability than frontier models and a hard ceiling set "
    "by available RAM. Inference has two phases: prefill, where the whole prompt is processed in "
    "parallel and which is compute-bound, and decode, where tokens are produced one at a time and "
    "which is memory-bandwidth-bound. Time to first token is dominated by prefill, while tokens per "
    "second during generation is dominated by how quickly the weights can be streamed from memory. "
)

PROMPT_SETS: Dict[str, str] = {
    "short": "Explain what a hash map is in two sentences.",
    "medium": f"Read the passage and summarise it in exactly three bullet points.\n\n{_PASSAGE}",
    "long": "Read the report and list its five most important claims.\n\n"
    + "\n\n".join(f"Section {i}. {_PASSAGE}" for i in range(1, 9)),
}


def _progress(model: str, label: str):
    def cb(done: int, total: int, m) -> None:
        sys.stdout.write(
            f"\r  {model:<24} {label:<7} {done:>3}/{total}  "
            f"ttft={fmt(m.ttft_ms, 0)}ms  decode={fmt(m.decode_tps)} tok/s   "
        )
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\n")
    return cb


def render_report(env: Dict, reports: List[LatencyReport], args: argparse.Namespace) -> str:
    rows = []
    for r in reports:
        rows.append([
            r.model, r.prompt_label,
            fmt(r.prompt_tokens.mean if r.prompt_tokens else None, 0),
            fmt(r.completion_tokens.mean if r.completion_tokens else None, 0),
            fmt(r.ttft_ms.median if r.ttft_ms else None, 0),
            fmt(r.ttft_ms.p95 if r.ttft_ms else None, 0),
            r.decode_tps.fmt() if r.decode_tps else "–",
            fmt(r.decode_tps.cv * 100 if r.decode_tps else None, 1, "%"),
            fmt(r.prompt_tps.median if r.prompt_tps else None, 0),
            fmt(r.itl_ms.median if r.itl_ms else None, 1),
            fmt(r.total_latency_ms.median if r.total_latency_ms else None, 0),
            fmt(r.total_latency_ms.p95 if r.total_latency_ms else None, 0),
            fmt(r.e2e_tps.median if r.e2e_tps else None),
        ])
    cold_rows = [
        [r.model, r.prompt_label, fmt(r.cold_start.load_ms, 0), fmt(r.cold_start.ttft_ms, 0),
         fmt(r.ttft_ms.median if r.ttft_ms else None, 0)]
        for r in reports if r.cold_start
    ]
    conc_rows = [
        [r.model, r.prompt_label, r.concurrency, r.aggregate_tps.fmt(), r.decode_tps.fmt() if r.decode_tps else "–"]
        for r in reports if r.aggregate_tps
    ]

    parts = [
        "# Phase 1 - Local Inference Latency Report", "",
        "## Environment", env_table(env), "",
        "## Methodology",
        f"- {args.warmup} warm-up iteration(s) discarded, then {args.iterations} measured iterations per (model, prompt).",
        f"- temperature={args.temperature}, seed={args.seed}, num_predict={args.max_tokens}, num_ctx={args.num_ctx}.",
        f"- Prefix-cache busting: {'ON (unique nonce per request)' if not args.no_cache_bust else 'OFF'}.",
        "- **TTFT** = client wall clock from request send to first streamed token (includes HTTP + prefill).",
        "- **Decode tok/s** = Ollama `eval_count / eval_duration` (pure generation speed). Shown as mean ± 95% CI.",
        "- **CV** = coefficient of variation of decode tok/s (run-to-run stability; <5% is a stable rig).",
        "- **Prefill tok/s** = `prompt_eval_count / prompt_eval_duration`.",
        "- **ITL** = median inter-token latency between streamed chunks.",
        "- **E2E tok/s** = completion tokens / total wall latency (what a user perceives).", "",
        "## Results",
        md_table(["Model", "Prompt", "In tok", "Out tok", "TTFT p50 (ms)", "TTFT p95 (ms)",
                  "Decode tok/s", "CV", "Prefill tok/s", "ITL p50 (ms)", "Latency p50 (ms)",
                  "Latency p95 (ms)", "E2E tok/s"], rows),
    ]
    if cold_rows:
        parts += ["", "## Cold start (model unloaded first)",
                  md_table(["Model", "Prompt", "Load (ms)", "Cold TTFT (ms)", "Warm TTFT p50 (ms)"], cold_rows)]
    if conc_rows:
        parts += ["", "## Concurrency",
                  "Aggregate throughput = total completion tokens / batch wall time. Requires "
                  "`OLLAMA_NUM_PARALLEL` >= concurrency on the server to actually run in parallel.",
                  md_table(["Model", "Prompt", "Concurrency", "Aggregate tok/s", "Per-request decode tok/s"], conc_rows)]
    warnings = environment_warnings(env)
    if warnings:
        parts += ["", "## Warnings"] + [f"- {w}" for w in warnings]
    return "\n".join(parts) + "\n"


async def main(args: argparse.Namespace) -> int:
    labels = list(PROMPT_SETS) if args.prompt_set == "all" else [args.prompt_set]
    prompts = {"custom": args.prompt} if args.prompt else {k: PROMPT_SETS[k] for k in labels}
    options = {"temperature": args.temperature, "seed": args.seed,
               "num_predict": args.max_tokens, "num_ctx": args.num_ctx}

    async with OllamaClient() as client:
        env = await capture_environment(client)
        for w in environment_warnings(env):
            print(f"WARNING: {w}")
        out_dir = make_run_dir("phase1")
        reports: List[LatencyReport] = []
        for model in args.model:
            if not await client.is_available(model):
                print(f"SKIP {model}: not pulled (run `ollama pull {model}`)")
                continue
            for i, (label, prompt) in enumerate(prompts.items()):
                try:
                    report = await run_latency_benchmark(
                        client, model, [{"role": "user", "content": prompt}], options,
                        iterations=args.iterations, warmup=args.warmup,
                        cold_start=args.cold_start and i == 0,
                        cache_bust=not args.no_cache_bust, concurrency=args.concurrency,
                        prompt_label=label, on_progress=_progress(model, label),
                    )
                except (ModelNotFoundError, OllamaError) as exc:
                    print(f"\nERROR {model}/{label}: {exc}")
                    continue
                reports.append(report)
                for run in report.runs:
                    append_jsonl(out_dir / "runs.jsonl", {"prompt": label, **run.model_dump()})
            if args.unload_after:
                await client.unload(model)

    if not reports:
        print("No results collected.")
        return 1
    md = render_report(env, reports, args)
    (out_dir / "report.md").write_text(md)
    write_json(out_dir / "summary.json", {"environment": env, "args": vars(args), "reports": reports})
    print("\n" + md)
    print(f"Saved to {out_dir}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", help="Repeatable. Default: llama3.2:3b")
    p.add_argument("--prompt", help="Custom prompt (overrides --prompt-set)")
    p.add_argument("--prompt-set", choices=[*PROMPT_SETS, "all"], default="all")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--cold-start", action="store_true", help="Unload model first and record cold TTFT")
    p.add_argument("--no-cache-bust", action="store_true", help="Allow Ollama prompt-prefix cache reuse")
    p.add_argument("--unload-after", action="store_true", help="Free memory after each model")
    args = p.parse_args(argv)
    args.model = args.model or ["llama3.2:3b"]
    return args


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
