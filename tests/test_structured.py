"""Offline tests for the validation + exactly-one-retry layer (no Ollama needed)."""
import asyncio
import json
from typing import Any, Dict, List

import pytest

from app.metrics import GenerationMetrics
from app.ollama_client import ChatResult
from app.output_schemas import AssistantAnswer, TicketExtraction
from app.structured import extract_json_candidate, generate_structured, validate_output

VALID = {"answer": "A fruit.", "category": "factual", "confidence": 0.9, "key_points": ["botany"]}


class FakeClient:
    """Returns canned outputs in order and records every call."""

    def __init__(self, outputs: List[str]) -> None:
        self.outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    async def chat(self, model, messages, **kwargs) -> ChatResult:
        self.calls.append({"messages": [dict(m) for m in messages], **kwargs})
        return ChatResult(model=model, content=self.outputs.pop(0), metrics=GenerationMetrics(model=model))


def run(client, schema=AssistantAnswer, **kw):
    return asyncio.run(generate_structured(client, "fake", "q?", schema, **kw))


def test_valid_first_try_no_retry():
    c = FakeClient([json.dumps(VALID)])
    res = run(c)
    assert res.success and not res.retried and res.first_attempt_valid
    assert len(c.calls) == 1
    assert res.data["answer"] == "A fruit."


def test_invalid_json_then_valid_retries_once():
    c = FakeClient(["Sure! {answer: oops", json.dumps(VALID)])
    res = run(c)
    assert res.success and res.retried and not res.first_attempt_valid
    assert len(c.calls) == 2
    assert res.attempts[0].error_type == "json_syntax"
    # repair prompt must contain the bad output and the validator error
    second = c.calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": "Sure! {answer: oops"}
    assert "Invalid JSON" in second[-1]["content"]


def test_fails_gracefully_after_exactly_one_retry():
    c = FakeClient(["not json", "still not json", json.dumps(VALID)])
    res = run(c)
    assert not res.success
    assert len(c.calls) == 2, "must re-prompt exactly once"
    assert res.data is None and res.error and len(res.attempts) == 2
    assert len(c.outputs) == 1  # third output never requested


def test_schema_violation_triggers_retry():
    bad = dict(VALID, confidence=1.7, extra_key=1)
    c = FakeClient([json.dumps(bad), json.dumps(VALID)])
    res = run(c)
    assert res.success and res.retried
    assert res.attempts[0].error_type == "schema"
    assert "confidence" in res.attempts[0].error and "extra_key" in res.attempts[0].error


def test_strict_types_reject_stringly_values():
    bad = dict(VALID, confidence="0.9")
    inst, etype, err = validate_output(json.dumps(bad), AssistantAnswer)
    assert inst is None and etype == "schema"


def test_int_accepted_for_float_field():
    inst, _, _ = validate_output(json.dumps(dict(VALID, confidence=1)), AssistantAnswer)
    assert inst is not None


def test_code_fences_are_stripped():
    fenced = "```json\n" + json.dumps(VALID) + "\n```"
    assert json.loads(extract_json_candidate(fenced)) == VALID
    c = FakeClient([fenced])
    assert run(c).success and len(c.calls) == 1


def test_lenient_mode_salvages_prose():
    text = "Here you go: " + json.dumps(VALID) + " Hope that helps!"
    assert validate_output(text, AssistantAnswer)[0] is None
    assert validate_output(text, AssistantAnswer, lenient=True)[0] is not None


@pytest.mark.parametrize("enforcement,expected", [
    ("schema", dict), ("json", str), ("prompt", type(None)),
])
def test_enforcement_sets_ollama_format(enforcement, expected):
    c = FakeClient([json.dumps({
        "customer_name": "A", "product": "B", "issue_summary": "broken thing",
        "priority": "low", "sentiment": "neutral", "requires_followup": False,
    })])
    res = run(c, schema=TicketExtraction, enforcement=enforcement)
    assert res.success
    assert isinstance(c.calls[0]["format"], expected)
