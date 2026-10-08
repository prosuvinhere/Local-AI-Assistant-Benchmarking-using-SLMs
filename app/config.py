"""Runtime configuration, driven by environment variables (no extra deps)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse


def normalize_host(raw: str) -> str:
    """Turn OLLAMA_HOST-style values ("0.0.0.0", "localhost:11434") into a client URL."""
    raw = raw.strip().rstrip("/")
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    # 0.0.0.0 is a *bind* address for the server; clients must use loopback.
    raw = raw.replace("://0.0.0.0", "://127.0.0.1")
    if urlparse(raw).port is None:
        raw += ":11434"
    return raw


@dataclass(frozen=True)
class Settings:
    ollama_host: str = field(
        default_factory=lambda: normalize_host(os.getenv("OLLAMA_HOST", "127.0.0.1:11434"))
    )
    default_model: str = field(default_factory=lambda: os.getenv("SLM_DEFAULT_MODEL", "llama3.2:3b"))
    request_timeout_s: float = field(default_factory=lambda: float(os.getenv("SLM_TIMEOUT_S", "600")))
    # How long Ollama keeps a model resident after a request. Benchmarks rely on
    # this to avoid paying load cost on every iteration.
    keep_alive: str = field(default_factory=lambda: os.getenv("SLM_KEEP_ALIVE", "10m"))
    default_num_ctx: int = field(default_factory=lambda: int(os.getenv("SLM_NUM_CTX", "4096")))


settings = Settings()
