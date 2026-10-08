"""Memory measurement for locally served models.

Three complementary signals (none is perfect on its own):

1. ``/api/ps`` ``size`` / ``size_vram`` - Ollama's own accounting of weights +
   KV cache + compute buffers for the loaded model. **Most reliable** number,
   and the one used for cross-model comparison.
2. Peak RSS of all ``ollama`` processes, sampled at 10 Hz in a background
   thread. On Apple Silicon (unified memory) Metal buffers and mmapped weights
   are only partially reflected in RSS, so treat it as a lower bound.
3. System-wide used-memory delta (``total - available``) - catches everything
   but is noisy if other apps are active.
"""
from __future__ import annotations

import os
import threading
import time
from typing import List, Optional

import psutil


def ollama_processes() -> List[psutil.Process]:
    me = os.getpid()
    procs = []
    for p in psutil.process_iter(["name"]):
        try:
            if p.pid != me and "ollama" in (p.info["name"] or "").lower():
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs


def ollama_rss_bytes() -> int:
    total = 0
    for p in ollama_processes():
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def system_used_bytes() -> int:
    vm = psutil.virtual_memory()
    return vm.total - vm.available


class MemorySampler:
    """Context manager that samples Ollama RSS + system memory in a thread."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.baseline_rss = 0
        self.peak_rss = 0
        self.baseline_sys = 0
        self.peak_sys = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, ollama_rss_bytes())
            self.peak_sys = max(self.peak_sys, system_used_bytes())
            self.samples += 1
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "MemorySampler":
        self.baseline_rss = self.peak_rss = ollama_rss_bytes()
        self.baseline_sys = self.peak_sys = system_used_bytes()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def rss_delta(self) -> int:
        return max(0, self.peak_rss - self.baseline_rss)

    @property
    def sys_delta(self) -> int:
        return max(0, self.peak_sys - self.baseline_sys)


def wait_for_memory_settle(seconds: float = 2.0) -> None:
    """Give the OS a moment to reclaim pages after an unload."""
    time.sleep(seconds)
