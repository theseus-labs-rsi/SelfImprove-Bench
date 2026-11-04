"""Reusable interface plumbing for benchmark-integrated text-state RSI methods.

This module contains transport and persistence only.  Each method adapter owns
its state schema, acting prompt, refinement prompt, candidate normalization,
and acceptance policy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backend import BackendDispatcher, OpenAIBackend
from .contracts import BackendResponse, FeedbackResult, TokenUsage, to_jsonable
from .interfaces import Candidate, CandidateBatch, MethodContext, SelectionResult
from .resume_state import atomic_json


COMMON_OPTIONS = frozenset({
    "actor_model", "optimizer_model", "api_key_env", "base_url_env", "api_mode",
    "actor_max_output_tokens", "optimizer_max_output_tokens", "actor_reasoning_effort",
    "optimizer_reasoning_effort", "temperature", "selection_margin",
})


def _json_text(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)


def _complete_truncated_json_object(text: str) -> str | None:
    """Close an object cut off at EOF without repairing interior syntax errors."""
    start = text.find("{")
    if start < 0:
        return None
    candidate = text[start:].strip()
    stack: list[str] = []
    in_string = False
    escaped = False
    incomplete = False
    pairs = {"}": "{", "]": "["}

    for index, char in enumerate(candidate):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            elif char in "\r\n":
                return None
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack[-1] != pairs[char]:
                return None
            stack.pop()
            if not stack and candidate[index + 1:].strip():
                return None

    if escaped:
        return None
    if in_string:
        candidate += '"'
        incomplete = True
    if stack:
        candidate += "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        incomplete = True
    return candidate if incomplete else None


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("optimizer response did not contain a JSON object")
    candidates = [stripped]
    end = stripped.rfind("}")
    if start > 0 or (end >= start and end < len(stripped) - 1):
        candidates.append(stripped[start:end + 1])
    repaired = _complete_truncated_json_object(stripped)
    if repaired is not None:
        candidates.append(repaired)

    last_error: json.JSONDecodeError | None = None
    value: Any = None
    for candidate in dict.fromkeys(candidates):
        try:
            value = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
    else:
        assert last_error is not None
        raise ValueError(f"optimizer response contained invalid JSON: {last_error}") from last_error
    if not isinstance(value, dict):
        raise ValueError("optimizer response must be a JSON object")
    return value


def _usage_report(usage: TokenUsage | None) -> dict[str, Any]:
    if usage is None:
        return {
            "prompt_tokens": None, "reasoning_tokens": None,
            "answer_tokens": None, "total_tokens": None, "complete": False,
        }
    answer = usage.output_tokens
    if answer is not None and usage.reasoning_tokens is not None:
        answer = max(answer - usage.reasoning_tokens, 0)
    return {
        "prompt_tokens": usage.input_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "answer_tokens": answer,
        "total_tokens": usage.total_tokens,
        "complete": usage.total_tokens is not None,
    }


def feedback_evidence(feedback: Sequence[FeedbackResult]) -> list[dict[str, Any]]:
    """Preserve the complete public feedback contract for method-side refinement."""
    return [to_jsonable(item) for item in feedback]


@dataclass(frozen=True, slots=True)
class TextStateAgent:
    """Immutable acting snapshot backed by one method-owned structured state."""

    dispatcher: BackendDispatcher
    model: str
    method_name: str
    state: Mapping[str, Any]
    system_prompt: str
    api_mode: str | None
    max_output_tokens: int
    reasoning_effort: str | None
    temperature: float | None

    def act(
        self, prompt: str, *, image_paths: Sequence[str | Path] = (),
    ) -> BackendResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("task prompt must be a non-empty string")
        rendered = (
            f"{self.system_prompt.strip()}\n\n"
            f"# Current {self.method_name} state\n{_json_text(self.state)}\n\n"
            f"# Task\n{prompt.strip()}\n\n"
            "Return only the answer/action requested by the task. Do not describe the "
            "method state unless the task explicitly asks for it."
        )
        response = self.dispatcher.distribute(
            "openai", rendered, image_paths=image_paths, model=self.model,
            api_mode=self.api_mode, max_output_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort, temperature=self.temperature,
        )
        metadata = dict(response.metadata or {})
        metadata["rsi_method"] = self.method_name
        metadata["state_digest"] = hashlib.sha256(
            _json_text(self.state).encode("utf-8")
        ).hexdigest()
        return replace(response, metadata=metadata)


class StructuredStateAdapter:
    """Base lifecycle for methods whose deployable artifact is structured text state.

    Subclasses supply method semantics through ``initial_state``,
    ``acting_instructions``, ``refinement_instructions`` and optionally
    ``normalize_candidate`` / ``accept_candidate``.
    """

    method_name = "method"
    state_filename = "state.json"
    allowed_options: frozenset[str] = frozenset()

    def __init__(self, context: MethodContext) -> None:
        if context is None:
            raise TypeError("MethodContext is required")
        unknown = set(context.options) - COMMON_OPTIONS - self.allowed_options
        if unknown:
            raise ValueError(
                f"unknown {self.method_name} adapter_options: {', '.join(sorted(unknown))}"
            )
        self._method_context = context
        self.options = dict(context.options)
        self.artifact_dir = context.artifact_dir.resolve()
        self.state_dir = self.artifact_dir / "state"
        self.actor_model = str(
            self.options.get("actor_model") or context.model or "gpt-5.6-sol"
        )
        self.optimizer_model = str(
            self.options.get("optimizer_model") or context.model or self.actor_model
        )
        self.api_mode = str(self.options.get("api_mode", "responses"))
        self.actor_max_output_tokens = int(self.options.get("actor_max_output_tokens", 4096))
        self.optimizer_max_output_tokens = int(
            self.options.get("optimizer_max_output_tokens", 8192)
        )
        self.actor_reasoning_effort = self.options.get("actor_reasoning_effort")
        self.optimizer_reasoning_effort = self.options.get("optimizer_reasoning_effort")
        self.temperature = self.options.get("temperature")
        self.selection_margin = float(self.options.get("selection_margin", 0.0))
        if self.actor_max_output_tokens < 1 or self.optimizer_max_output_tokens < 1:
            raise ValueError("max output token settings must be positive")
        self.dispatcher = BackendDispatcher({
            "openai": OpenAIBackend(
                api_key_env=str(self.options.get("api_key_env", "OPENAI_API_KEY")),
                base_url_env=str(self.options.get("base_url_env", "OPENAI_BASE_URL")),
                api_mode=self.api_mode,
                max_output_tokens=max(
                    self.actor_max_output_tokens, self.optimizer_max_output_tokens
                ),
            )
        })
        self._state: dict[str, Any] = {}
        self._version = 0
        self._history: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self.last_refinement_report: dict[str, Any] | None = None
        self.last_selection_report: dict[str, Any] | None = None

    def initial_state(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def acting_instructions(self) -> str:
        raise NotImplementedError

    def refinement_instructions(self, evidence: Sequence[Mapping[str, Any]]) -> str:
        raise NotImplementedError

    def normalize_candidate(
        self, payload: Mapping[str, Any], *, current: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = payload.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("optimizer response must contain an object field named 'state'")
        return dict(state)

    def accept_candidate(self, parent: Any, candidate: Any) -> tuple[bool, str]:
        parent_score = getattr(parent, "relative_score", None)
        candidate_score = getattr(candidate, "relative_score", None)
        if getattr(candidate, "status", "") != "ok":
            return False, "candidate evaluation did not complete successfully"
        if not isinstance(candidate_score, (int, float)):
            return False, "candidate evaluation has no numeric relative score"
        if not isinstance(parent_score, (int, float)):
            return True, "parent score unavailable; candidate completed successfully"
        accepted = float(candidate_score) > float(parent_score) + self.selection_margin
        return accepted, (
            "candidate strictly improved the paired score"
            if accepted else "candidate did not clear the paired validation margin"
        )

    def init(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=False)
        self._state = self._validate_state(dict(self.initial_state()))
        self._version = 0
        self._history = []
        self._pending = None
        self._persist_state()

    def load(self) -> TextStateAgent:
        if not self._state:
            raise RuntimeError(f"{self.method_name} adapter is not initialized")
        return self._agent(self._state)

    def _agent(self, state: Mapping[str, Any]) -> TextStateAgent:
        return TextStateAgent(
            dispatcher=self.dispatcher, model=self.actor_model,
            method_name=self.method_name, state=dict(state),
            system_prompt=self.acting_instructions(), api_mode=self.api_mode,
            max_output_tokens=self.actor_max_output_tokens,
            reasoning_effort=self.actor_reasoning_effort,
            temperature=float(self.temperature) if self.temperature is not None else None,
        )

    def refine(
        self, feedback: list[FeedbackResult], *, output_dir: Path,
    ) -> CandidateBatch:
        if self._pending is not None:
            raise RuntimeError("previous candidate still requires paired selection")
        evidence = feedback_evidence(feedback)
        atomic_json(output_dir / "method_feedback.json", evidence)
        prompt = (
            self.refinement_instructions(evidence)
            + "\n\n# Current deployable state\n"
            + _json_text(self._state)
            + "\n\n# Benchmark feedback\n"
            + _json_text(evidence)
            + "\n\nReturn one JSON object only."
        )
        response = self.dispatcher.distribute(
            "openai", prompt, model=self.optimizer_model, api_mode=self.api_mode,
            max_output_tokens=self.optimizer_max_output_tokens,
            reasoning_effort=self.optimizer_reasoning_effort,
            temperature=float(self.temperature) if self.temperature is not None else None,
        )
        (output_dir / "optimizer_response.txt").write_text(response.text, encoding="utf-8")
        payload = _parse_json_object(response.text)
        candidate_state = self._validate_state(
            self.normalize_candidate(payload, current=self._state)
        )
        candidate_id = f"{self.method_name}-v{self._version + 1:04d}"
        candidate_dir = output_dir / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(candidate_dir / self.state_filename, candidate_state)
        report = {
            "method": self.method_name,
            "version_before": self._version,
            "version_candidate": self._version + 1,
            "candidate_id": candidate_id,
            "feedback_count": len(evidence),
            "optimizer_model": self.optimizer_model,
            "optimizer_reasoning": response.reasoning,
            "optimizer_metadata": to_jsonable(response.metadata),
            "proposal": to_jsonable(payload),
            "token_usage_report": _usage_report(response.usage),
        }
        atomic_json(output_dir / "refinement_report.json", report)
        self.last_refinement_report = report
        self._pending = {
            "candidate_id": candidate_id,
            "state": candidate_state,
            "report": report,
        }
        return CandidateBatch(
            candidates=(Candidate(candidate_id, self._agent(candidate_state), candidate_dir),),
            work_dir=output_dir,
        )

    def select(
        self, *, batch: CandidateBatch, parent_evaluation: Any,
        candidate_evaluations: Mapping[str, Any], output_dir: Path,
        selection_mode: str = "paired",
    ) -> SelectionResult:
        if selection_mode != "paired":
            raise ValueError("only paired selection is supported")
        if self._pending is None:
            raise RuntimeError("no pending candidate")
        candidate_id = self._pending["candidate_id"]
        if tuple(item.candidate_id for item in batch.candidates) != (candidate_id,):
            raise ValueError("candidate batch does not match pending method state")
        candidate_eval = candidate_evaluations[candidate_id]
        accepted, reason = self.accept_candidate(parent_evaluation, candidate_eval)
        report = {
            "method": self.method_name,
            "candidate_id": candidate_id,
            "accepted": accepted,
            "reason": reason,
            "parent": to_jsonable(parent_evaluation),
            "candidate": to_jsonable(candidate_eval),
            "version_before": self._version,
            "version_after": self._version + (1 if accepted else 0),
        }
        if accepted:
            self._state = dict(self._pending["state"])
            self._version += 1
        self._history.append(report)
        self._pending = None
        self.last_selection_report = report
        self._persist_state()
        atomic_json(output_dir / "selection.json", report)
        return SelectionResult(
            status="committed" if accepted else "unchanged",
            committed_candidate_id=candidate_id if accepted else None,
        )

    def export_state(self) -> Mapping[str, Any]:
        return {
            "schema_version": 1,
            "method": self.method_name,
            "version": self._version,
            "state": to_jsonable(self._state),
            "history": to_jsonable(self._history),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1 or state.get("method") != self.method_name:
            raise RuntimeError(f"invalid {self.method_name} checkpoint identity")
        version = state.get("version")
        history = state.get("history")
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise RuntimeError("invalid checkpoint version")
        if not isinstance(history, list):
            raise RuntimeError("invalid checkpoint history")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state = self._validate_state(state.get("state"))
        self._version = version
        self._history = list(history)
        self._pending = None
        self._persist_state()

    def close(self) -> None:
        self._pending = None

    def _validate_state(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"{self.method_name} state must be a non-empty object")
        encoded = json.dumps(to_jsonable(value), ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
        if not isinstance(decoded, dict):
            raise ValueError(f"{self.method_name} state must serialize as an object")
        return decoded

    def _persist_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.state_dir / self.state_filename, self._state)
        atomic_json(self.state_dir / "history.json", self._history)
        atomic_json(self.state_dir / "checkpoint.json", self.export_state())


__all__ = [
    "COMMON_OPTIONS", "StructuredStateAdapter", "TextStateAgent", "feedback_evidence",
]
