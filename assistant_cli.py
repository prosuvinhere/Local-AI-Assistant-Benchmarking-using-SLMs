"""Interactive terminal assistant (CLI alternative to the FastAPI service).

    python assistant_cli.py --model llama3.2:3b
Commands: /reset  clear history · /stats  toggle metrics · /quit
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import settings
from app.ollama_client import ChatResult, OllamaClient, OllamaError


async def repl(model: str, system: str, temperature: float) -> None:
    history = [{"role": "system", "content": system}]
    show_stats = True
    async with OllamaClient() as client:
        if not await client.is_available(model):
            print(f"Model '{model}' not found. Run: ollama pull {model}")
            return
        print(f"Local assistant on {model} (100% on-device). /reset /stats /quit")
        while True:
            try:
                user = input("\nyou › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not user:
                continue
            if user == "/quit":
                return
            if user == "/reset":
                history = history[:1]
                print("(history cleared)")
                continue
            if user == "/stats":
                show_stats = not show_stats
                continue
            history.append({"role": "user", "content": user})
            print("ai  › ", end="", flush=True)
            try:
                async for ev in client.stream_chat(model, history, options={"temperature": temperature}):
                    if isinstance(ev, ChatResult):
                        history.append({"role": "assistant", "content": ev.content})
                        m = ev.metrics
                        if show_stats:
                            print(f"\n      [ttft {m.ttft_ms:.0f} ms · {m.decode_tps or 0:.1f} tok/s · "
                                  f"{m.completion_tokens} tok · {m.total_latency_ms / 1000:.2f} s]")
                    elif not ev.is_thinking:
                        print(ev.text, end="", flush=True)
            except OllamaError as exc:
                history.pop()
                print(f"\n[error] {exc}")


def main() -> None:
    p = argparse.ArgumentParser(description="Local SLM assistant (terminal)")
    p.add_argument("--model", default=settings.default_model)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--system", default="You are a concise, helpful assistant running fully offline.")
    args = p.parse_args()
    asyncio.run(repl(args.model, args.system, args.temperature))


if __name__ == "__main__":
    sys.exit(main())
