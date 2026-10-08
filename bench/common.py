"""Shared helpers for the benchmark CLIs: environment capture, output, tables."""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import psutil
from pydantic import BaseModel

from app.ollama_client import OllamaClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"


def make_run_dir(phase: str, root: Optional[Path] = None) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = (root or RESULTS_ROOT) / f"{phase}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cpu_brand() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


async def capture_environment(client: OllamaClient) -> Dict[str, Any]:
    """Everything needed to reproduce / interpret a benchmark run."""
    battery = None
    try:
        b = psutil.sensors_battery()
        if b is not None:
            battery = {"percent": b.percent, "power_plugged": b.power_plugged}
    except Exception:
        pass
    try:
        ollama_version = await client.version()
    except Exception as exc:  # pragma: no cover
        ollama_version = f"unavailable ({exc})"
    vm = psutil.virtual_memory()
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_brand(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "ram_available_gb_at_start": round(vm.available / 1e9, 2),
        "python": platform.python_version(),
        "ollama_version": ollama_version,
        "ollama_host": client.host,
        "battery": battery,
    }


def environment_warnings(env: Dict[str, Any]) -> List[str]:
    warnings = []
    bat = env.get("battery")
    if bat and not bat.get("power_plugged"):
        warnings.append("Running on battery: macOS/laptops throttle CPU/GPU - plug in for stable numbers.")
    if env.get("ram_available_gb_at_start", 99) < 3:
        warnings.append("Less than 3 GB RAM free at start: expect swapping and inflated latencies.")
    return warnings


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=2, default=str))


def append_jsonl(path: Path, obj: Any) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(to_jsonable(obj), default=str) + "\n")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    rows = [["" if c is None else str(c) for c in r] for r in rows]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def fmt(value: Optional[float], digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "–"
    return f"{value:,.{digits}f}{suffix}"


def fmt_bytes(n: Optional[float]) -> str:
    if not n:
        return "–"
    return f"{n / 1e9:.2f} GB"


def env_table(env: Dict[str, Any]) -> str:
    keys = ["timestamp", "os", "cpu", "cpu_cores_physical", "ram_total_gb",
            "ram_available_gb_at_start", "ollama_version", "python", "battery"]
    return md_table(["Property", "Value"], [(k, env.get(k)) for k in keys])
