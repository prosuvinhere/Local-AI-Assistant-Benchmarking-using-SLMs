"""Structured-output layer: schema-forced JSON + validation + one-shot repair.

Flow for every request::

    attempt 1 ──► parse JSON ──► Pydantic (strict) ──► OK ──► return
                     │ fail
                     ▼
    attempt 2 (repair prompt containing the exact validator error)
                     │
                     ├─► OK   ──► return (retried=True)
                     └─► fail ──► graceful failure object (never raises)

Three enforcement levels let you measure how much each layer contributes:

* ``schema`` - pass the JSON Schema to Ollama's ``format`` parameter
  (grammar-constrained decoding: syntactically valid, schema-shaped output).
* ``json``   - ``format="json"``: valid JSON guaranteed, shape is not.
* ``prompt`` - instructions only; the model can emit anything. This is where
  the validator + retry earn their keep.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Type

from pydantic import BaseModel, ValidationError

from .metrics import GenerationMetrics
from .ollama_client import ChatResult

Enforcement = Literal["schema", "json", "prompt"]

# The spec: re-prompt exactly once before failing gracefully.
MAX_RETRIES = 1


class ChatCapable(Protocol):  # lets tests inject a fake client
    async def chat(self, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> ChatResult: ...


class AttemptRecord(BaseModel):
    attempt: int
    raw_output: str
    valid: bool
    error_type: Optional[Literal["json_syntax", "schema"]] = None
    error: Optional[str] = None
    metrics: GenerationMetrics


class StructuredResult(BaseModel):
    success: bool
    schema_name: str
    enforcement: Enforcement
    data: Optional[Dict[str, Any]] = None
    retried: bool = False
    attempts: List[AttemptRecord]
    error: Optional[str] = None
    total_latency_ms: float = 0.0

    @property
    def first_attempt_valid(self) -> bool:
        return bool(self.attempts) and self.attempts[0].valid


def build_system_prompt(schema_cls: Type[BaseModel]) -> str:
    schema = json.dumps(schema_cls.model_json_schema(), indent=2)
    return (
        "You are a precise assistant that replies ONLY with a single JSON object.\n"
        "The JSON object MUST validate against this JSON Schema:\n"
        f"{schema}\n\n"
        "Rules:\n"
        "- Output raw JSON only: no markdown fences, no comments, no text before or after.\n"
        "- Include every required field, use exactly the property names shown, add no other keys.\n"
        "- Respect types, enums and numeric ranges exactly (numbers are not strings, booleans are true/false)."
    )


def build_repair_prompt(error: str) -> str:
    return (
        "Your previous reply was rejected by the validator with these errors:\n"
        f"{error}\n\n"
        "Reply again with ONLY the corrected JSON object that satisfies the schema. "
        "No explanation, no markdown."
    )


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.S)


def extract_json_candidate(text: str, lenient: bool = False) -> str:
    """Strip markdown code fences (a formatting habit, not a content error).

    With ``lenient=True`` also slice from the first '{' to the last '}', which
    salvages "Sure! Here is the JSON: {...}" replies. Off by default so that the
    retry path is exercised honestly.
    """
    candidate = text.strip()
    match = _FENCE_RE.match(candidate)
    if match:
        candidate = match.group(1).strip()
    if lenient and not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]
    return candidate


def _format_validation_error(exc: ValidationError, limit: int = 8) -> str:
    lines = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        lines.append(f"- {loc}: {err.get('msg')}")
    extra = len(exc.errors()) - limit
    if extra > 0:
        lines.append(f"- ... and {extra} more")
    return "\n".join(lines)


def validate_output(
    raw: str, schema_cls: Type[BaseModel], lenient: bool = False
) -> Tuple[Optional[BaseModel], Optional[str], Optional[str]]:
    """Return (instance, error_type, error_message)."""
    candidate = extract_json_candidate(raw, lenient=lenient)
    try:
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        snippet = candidate[:120].replace("\n", " ")
        return None, "json_syntax", f"- Invalid JSON: {exc.msg} at line {exc.lineno} col {exc.colno}. Output began: {snippet!r}"
    try:
        # strict=True: "0.9" is NOT a float, "true" is NOT a bool.
        return schema_cls.model_validate_json(candidate, strict=True), None, None
    except ValidationError as exc:
        return None, "schema", _format_validation_error(exc)


async def generate_structured(
    client: ChatCapable,
    model: str,
    prompt: str,
    schema_cls: Type[BaseModel],
    *,
    schema_name: Optional[str] = None,
    enforcement: Enforcement = "schema",
    options: Optional[Dict[str, Any]] = None,
    lenient: bool = False,
    max_retries: int = MAX_RETRIES,
) -> StructuredResult:
    """Generate a schema-valid object, re-prompting at most ``max_retries`` times."""
    fmt: Any = None
    if enforcement == "schema":
        fmt = schema_cls.model_json_schema()
    elif enforcement == "json":
        fmt = "json"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(schema_cls)},
        {"role": "user", "content": prompt},
    ]
    attempts: List[AttemptRecord] = []
    last_error = "unknown error"
    t0 = time.perf_counter()

    for attempt in range(1, max_retries + 2):
        result = await client.chat(model, messages, options=options, format=fmt)
        instance, err_type, err = validate_output(result.content, schema_cls, lenient=lenient)
        attempts.append(
            AttemptRecord(
                attempt=attempt,
                raw_output=result.content,
                valid=instance is not None,
                error_type=err_type,  # type: ignore[arg-type]
                error=err,
                metrics=result.metrics,
            )
        )
        if instance is not None:
            return StructuredResult(
                success=True,
                schema_name=schema_name or schema_cls.__name__,
                enforcement=enforcement,
                data=instance.model_dump(),
                retried=attempt > 1,
                attempts=attempts,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
            )
        last_error = err or last_error
        # Keep the bad answer in context so the model can see what to fix.
        messages = messages + [
            {"role": "assistant", "content": result.content},
            {"role": "user", "content": build_repair_prompt(last_error)},
        ]

    return StructuredResult(
        success=False,
        schema_name=schema_name or schema_cls.__name__,
        enforcement=enforcement,
        retried=max_retries > 0,
        attempts=attempts,
        error=f"Output failed validation after {len(attempts)} attempt(s):\n{last_error}",
        total_latency_ms=(time.perf_counter() - t0) * 1000,
    )
