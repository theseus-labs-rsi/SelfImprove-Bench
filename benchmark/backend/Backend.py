from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from benchmark.contracts import BackendRequest, BackendResponse, TokenUsage, ToolCall


# ---------- Concrete backend contract ----------

class ModelBackend(Protocol):
    def generate(self, request: BackendRequest) -> BackendResponse:
        ...


# ---------- Backend distribution ----------

class BackendDispatcher:
    def __init__(self, backends: Mapping[str, ModelBackend]) -> None:
        self._backends = dict(backends)

    def distribute(
        self,
        backend_name: str,
        prompt: str,
        image_paths: Sequence[str | Path] | None = None,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        parallel_tool_calls: bool = False,
        api_mode: str | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        output_format: str | None = None,
        temperature: float | None = None,
    ) -> BackendResponse:
        try:
            backend = self._backends[backend_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._backends)) or "<none>"
            raise ValueError(
                f"Unknown backend {backend_name!r}; available: {available}"
            ) from exc

        request = BackendRequest(
            prompt=prompt,
            model=model,
            messages=tuple(dict(message) for message in messages or ()),
            image_paths=tuple(Path(path) for path in image_paths or ()),
            tools=tuple(dict(tool) for tool in tools or ()),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            api_mode=api_mode,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            output_format=output_format,
            temperature=temperature,
        )
        return backend.generate(request)


__all__ = [
    "BackendDispatcher",
    "BackendRequest",
    "BackendResponse",
    "ModelBackend",
    "TokenUsage",
    "ToolCall",
]
