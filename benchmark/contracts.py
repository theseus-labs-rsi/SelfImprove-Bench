"""Shared data contracts between benchmarks and RSI methods.

This module owns data only.  Model-provider behavior and dispatch remain in the
backend layer, while benchmarks and RSI methods exchange the types below.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


# ---------- Model call data ----------

@dataclass(frozen=True, slots=True)
class BackendRequest:
    prompt: str
    model: str
    messages: tuple[Mapping[str, Any], ...] = ()
    image_paths: tuple[Path, ...] = ()
    tools: tuple[Mapping[str, Any], ...] = ()
    tool_choice: str | Mapping[str, Any] | None = None
    parallel_tool_calls: bool = False
    api_mode: str | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    output_format: str | None = None
    temperature: float | None = None


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str | None
    name: str
    arguments: str | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BackendResponse:
    text: str
    reasoning: str | None = None
    usage: TokenUsage | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    provider: str | None = None
    model: str | None = None
    status: str | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    metadata: Mapping[str, Any] | None = None


# ---------- Benchmark feedback data ----------

class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: str
    content_hash: str | None = None


@dataclass(frozen=True, slots=True)
class AgentCallRecord:
    """Provider-neutral model response retained for later refinement."""

    response_text: str
    reasoning: str | None = None
    usage: TokenUsage | None = None
    provider: str | None = None
    model: str | None = None
    status: str | None = None
    finish_reason: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class StepFeedback:
    step_index: int
    prompt: str
    image_records: tuple[ImageRecord, ...]
    action: str
    environment_response: Any
    success: bool | None
    score: float | None
    feedback: str | None
    agent_call: AgentCallRecord | None = None


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    execution_status: ExecutionStatus
    success: bool
    score: float
    feedback: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FeedbackResult:
    task_id: str
    task: Any
    trajectory: tuple[StepFeedback, ...]
    evaluation: TaskEvaluation


FeedbackResults = list[FeedbackResult]

@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Common benchmark result, including native multimodal feedback."""

    record: dict[str, Any]
    feedback: FeedbackResult | None = None


@dataclass(frozen=True, slots=True)
class SelectionEvaluation:
    """Compact benchmark result used only to compare parent and candidates."""

    status: str
    relative_score: float | None
    success: bool
    result_path: str
    error_type: str | None = None
    error: str | None = None


def to_jsonable(value: Any) -> Any:
    """Convert shared contract values into JSON-compatible data."""
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [to_jsonable(item) for item in value]
    return value


__all__ = [
    "AgentCallRecord",
    "BackendRequest",
    "BackendResponse",
    "ExecutionStatus",
    "EvaluationOutcome",
    "FeedbackResult",
    "FeedbackResults",
    "ImageRecord",
    "SelectionEvaluation",
    "StepFeedback",
    "TaskEvaluation",
    "TokenUsage",
    "ToolCall",
    "to_jsonable",
]
