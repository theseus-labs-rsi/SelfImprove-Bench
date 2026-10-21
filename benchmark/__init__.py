"""Unified deterministic multi-game RSI benchmark."""

from .contracts import (
    AgentCallRecord,
    BackendRequest,
    BackendResponse,
    ExecutionStatus,
    FeedbackResult,
    FeedbackResults,
    ImageRecord,
    StepFeedback,
    TaskEvaluation,
    TokenUsage,
    ToolCall,
    to_jsonable,
)


def __getattr__(name: str):
    """Keep method-side contract imports independent of game implementations."""
    if name in {"BenchmarkConfig", "load_config"}:
        from . import config

        return getattr(config, name)
    if name == "run_benchmark":
        from .runner import run_benchmark

        return run_benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentCallRecord",
    "BackendRequest",
    "BackendResponse",
    "BenchmarkConfig",
    "ExecutionStatus",
    "FeedbackResult",
    "FeedbackResults",
    "ImageRecord",
    "StepFeedback",
    "TaskEvaluation",
    "TokenUsage",
    "ToolCall",
    "load_config",
    "run_benchmark",
    "to_jsonable",
]
