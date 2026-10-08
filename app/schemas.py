"""HTTP request/response models for the FastAPI service."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .metrics import GenerationMetrics
from .output_schemas import SCHEMA_REGISTRY


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerationOptions(BaseModel):
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)
    seed: Optional[int] = Field(None, description="Fix for reproducible sampling at temperature > 0.")
    max_tokens: int = Field(512, ge=1, le=8192, description="Mapped to Ollama's num_predict.")
    num_ctx: Optional[int] = Field(None, ge=256, le=131072, description="Context window (affects memory).")

    def to_ollama(self) -> Dict[str, Any]:
        opts: Dict[str, Any] = {"temperature": self.temperature, "num_predict": self.max_tokens}
        for key in ("top_p", "top_k", "seed", "num_ctx"):
            val = getattr(self, key)
            if val is not None:
                opts[key] = val
        return opts


class ChatRequest(BaseModel):
    model: Optional[str] = Field(None, description="Defaults to SLM_DEFAULT_MODEL.")
    prompt: Optional[str] = None
    system: Optional[str] = None
    messages: Optional[List[Message]] = None
    options: GenerationOptions = Field(default_factory=GenerationOptions)

    @model_validator(mode="after")
    def _need_input(self) -> "ChatRequest":
        if not self.prompt and not self.messages:
            raise ValueError("Provide either 'prompt' or 'messages'.")
        return self

    def to_messages(self) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        if self.messages:
            msgs.extend(m.model_dump() for m in self.messages)
        if self.prompt:
            msgs.append({"role": "user", "content": self.prompt})
        return msgs


class ChatResponse(BaseModel):
    model: str
    content: str
    thinking: Optional[str] = None
    metrics: GenerationMetrics


class StructuredRequest(BaseModel):
    model: Optional[str] = None
    prompt: str = Field(..., min_length=1)
    schema_name: str = Field("assistant_answer", description=f"One of: {sorted(SCHEMA_REGISTRY)}")
    enforcement: Literal["schema", "json", "prompt"] = "schema"
    lenient: bool = Field(False, description="Salvage JSON embedded in prose before validating.")
    options: GenerationOptions = Field(default_factory=lambda: GenerationOptions(temperature=0.0))

    @field_validator("schema_name")
    @classmethod
    def _known_schema(cls, v: str) -> str:
        if v not in SCHEMA_REGISTRY:
            raise ValueError(f"Unknown schema '{v}'. Available: {sorted(SCHEMA_REGISTRY)}")
        return v


class BenchmarkRequest(BaseModel):
    model: Optional[str] = None
    prompt: str = "Explain in three sentences why small language models are useful on laptops."
    iterations: int = Field(5, ge=1, le=50)
    warmup: int = Field(1, ge=0, le=5)
    cache_bust: bool = True
    options: GenerationOptions = Field(
        default_factory=lambda: GenerationOptions(temperature=0.0, seed=42, max_tokens=256)
    )
