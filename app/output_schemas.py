"""Pydantic schemas the model is *forced* to produce (Phase 2).

``extra="forbid"`` turns into ``"additionalProperties": false`` in the JSON
schema, so hallucinated keys are rejected both by Ollama's grammar-constrained
decoder and by our validator.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Type

from pydantic import BaseModel, ConfigDict, Field


class AssistantAnswer(BaseModel):
    """General-purpose structured assistant reply."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1, description="Direct, concise answer to the user's question.")
    category: Literal["factual", "reasoning", "coding", "creative", "advice", "other"] = Field(
        ..., description="Type of question asked."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Self-reported confidence between 0 and 1.")
    key_points: List[str] = Field(
        ..., min_length=1, max_length=5, description="1-5 short supporting points."
    )


class TicketExtraction(BaseModel):
    """Information extraction from a customer-support message."""

    model_config = ConfigDict(extra="forbid")

    customer_name: str = Field(..., min_length=1, description="Full name of the customer.")
    product: str = Field(..., min_length=1, description="Product or service the message is about.")
    issue_summary: str = Field(..., min_length=5, max_length=300, description="One-sentence summary of the issue.")
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        ..., description="urgent = outage/financial loss now; high = blocked; medium = degraded; low = question."
    )
    sentiment: Literal["negative", "neutral", "positive"]
    requires_followup: bool = Field(..., description="True if a human agent must respond.")


SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "assistant_answer": AssistantAnswer,
    "ticket_extraction": TicketExtraction,
}


def get_schema(name: str) -> Type[BaseModel]:
    try:
        return SCHEMA_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown schema '{name}'. Available: {sorted(SCHEMA_REGISTRY)}") from None
