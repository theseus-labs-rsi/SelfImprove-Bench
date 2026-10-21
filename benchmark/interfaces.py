"""Public, method-neutral boundary between RSI implementations and games.

Third-party methods depend on this module and ``benchmark.contracts``, not on
game implementations or the runner. See README.md#adding-an-rsi-method.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .contracts import BackendResponse, EvaluationOutcome, FeedbackResult, SelectionEvaluation


@dataclass(frozen=True)
class MethodContext:
    """Run-local settings passed to an adapter factory as its only argument."""

    artifact_dir: Path
    task_count: int
    options: Mapping[str, Any]
    model: str | None = None
    timeout: float = 300.0
    resuming: bool = False
    seed: int = 0


class ActingAgent(Protocol):
    def act(self, prompt: str, *, image_paths: Sequence[Path] = ()) -> BackendResponse:
        """Return game action text; image_paths is used by Minecraft/Astray."""
        ...


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    agent: ActingAgent
    artifact_dir: Path


@dataclass(frozen=True)
class CandidateBatch:
    candidates: Sequence[Candidate]
    work_dir: Path


@dataclass(frozen=True)
class SelectionResult:
    status: str
    committed_candidate_id: str | None = None
    batch: CandidateBatch | None = None


class RSIMethod(Protocol):
    """Required lifecycle. Instances are constructed with ``MethodContext``.

    ``refine`` may commit internally and return None. Only methods returning a
    CandidateBatch also need ``select`` (the optional protocol below).
    """

    def init(self) -> None: ...
    def load(self) -> ActingAgent: ...
    def refine(self, feedback: list[FeedbackResult], *, output_dir: Path) -> CandidateBatch | None: ...
    def export_state(self) -> Mapping[str, Any]: ...
    def restore_state(self, state: Mapping[str, Any]) -> None: ...
    def close(self) -> None: ...


class CandidateSelectingMethod(RSIMethod, Protocol):
    def select(
        self, *, batch: CandidateBatch, parent_evaluation: SelectionEvaluation,
        candidate_evaluations: Mapping[str, SelectionEvaluation],
        output_dir: Path, selection_mode: str = "paired",
    ) -> SelectionResult: ...


class GameBench(Protocol):
    """Game-side evaluation: no method initialization or improvement here."""

    def evaluate(self, task: Mapping[str, Any], agent: ActingAgent) -> EvaluationOutcome: ...
