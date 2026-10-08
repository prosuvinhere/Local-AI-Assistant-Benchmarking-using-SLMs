"""FastAPI integration tests using TestClient and a mock Ollama client."""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.metrics import GenerationMetrics
from app.ollama_client import ChatResult, TokenChunk


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client



def test_health_check(client):
    with patch.object(app.state.client, "version", AsyncMock(return_value="0.33.1")), \
         patch.object(app.state.client, "ps", AsyncMock(return_value=[])):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ollama_version"] == "0.33.1"


def test_schemas_endpoint(client):

    resp = client.get("/v1/schemas")
    assert resp.status_code == 200
    schemas = resp.json()
    assert "assistant_answer" in schemas
    assert "ticket_extraction" in schemas
    assert "additionalProperties" in schemas["assistant_answer"]
    assert schemas["assistant_answer"]["additionalProperties"] is False


def test_chat_endpoint(client):
    mock_metrics = GenerationMetrics(
        model="llama3.2:3b",
        ttft_ms=250.0,
        total_latency_ms=1000.0,
        decode_tps=25.0,
        completion_tokens=25,
    )
    mock_result = ChatResult(
        model="llama3.2:3b",
        content="Testing response",
        metrics=mock_metrics,
    )

    with patch.object(app.state.client, "chat", AsyncMock(return_value=mock_result)):
        resp = client.post("/v1/chat", json={
            "model": "llama3.2:3b",
            "prompt": "Hello",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Testing response"
        assert data["metrics"]["decode_tps"] == 25.0


def test_structured_endpoint_success(client):

    mock_metrics = GenerationMetrics(
        model="llama3.2:3b",
        total_latency_ms=800.0,
        decode_tps=30.0,
        completion_tokens=40,
    )
    valid_json = (
        '{"answer": "Canberra is the capital.", '
        '"category": "factual", '
        '"confidence": 1.0, '
        '"key_points": ["Federal territory"]}'
    )
    mock_result = ChatResult(
        model="llama3.2:3b",
        content=valid_json,
        metrics=mock_metrics,
    )

    with patch.object(app.state.client, "chat", AsyncMock(return_value=mock_result)):
        resp = client.post("/v1/structured", json={
            "model": "llama3.2:3b",
            "prompt": "Capital of Australia?",
            "schema_name": "assistant_answer",
            "enforcement": "schema"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["category"] == "factual"
        assert data["retried"] is False
