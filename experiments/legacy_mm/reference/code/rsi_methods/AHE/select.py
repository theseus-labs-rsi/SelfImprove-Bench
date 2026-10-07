"""Choose one evaluated AHE variant and commit its harness as the next parent.

The benchmark owns evaluation. This module only compares its numeric scores;
it neither reruns candidates nor interprets task success as a binary reward.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .init import AHEState
from .refine import CandidateBatch, CandidateSpec


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    """Auditable decision returned to the benchmark's selection bridge."""

    status: str
    committed_candidate_id: str | None
    parent_score: float
    selected_score: float
    candidate_scores: dict[str, float | None]
    reason: str
    report_path: Path


def _evaluation(value: Any, label: str) -> tuple[str, float | None]:
    """Read the bridge's SelectionEvaluation without a pass/fail threshold.

    ``relative_score`` is optional for failed evaluations. A successful
    evaluation must have a finite number before it can enter selection.
    """

    field = value.get if isinstance(value, Mapping) else lambda key: getattr(value, key, None)
    status = str(field("status") or "").strip().lower()
    score = field("relative_score")
    if not status:
        raise ValueError(f"{label} evaluation has no status")
    if score is not None and (
        isinstance(score, bool) or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise ValueError(f"{label} relative_score must be finite or null")
    if status == "ok" and score is None:
        raise ValueError(f"{label} succeeded without a relative_score")
    return status, float(score) if score is not None else None


def _candidate_scores(
    batch: CandidateBatch, evaluations: Mapping[str, Any],
) -> tuple[dict[str, float | None], list[tuple[CandidateSpec, float]]]:
    """Require an evaluation for every generated variant, even failed ones."""

    ids = [candidate.candidate_id for candidate in batch.candidates]
    if not ids or len(set(ids)) != len(ids) or set(evaluations) != set(ids):
        raise ValueError("candidate_evaluations must match all pending candidate ids")
    scores: dict[str, float | None] = {}
    eligible: list[tuple[CandidateSpec, float]] = []
    for candidate in batch.candidates:
        if candidate.agent.workspace != candidate.workspace.resolve():
            raise RuntimeError(f"Candidate Agent workspace changed: {candidate.candidate_id}")
        status, score = _evaluation(evaluations[candidate.candidate_id], candidate.candidate_id)
        scores[candidate.candidate_id] = score
        if status == "ok" and score is not None:
            eligible.append((candidate, score))
    return scores, eligible


def select(
    state: AHEState,
    batch: CandidateBatch,
    *,
    parent_evaluation: Any,
    candidate_evaluations: Mapping[str, Any],
    output_dir: str | Path,
) -> SelectionOutcome:
    """Persist a score-based decision, then point state at the accepted copy.

    The parent remains untouched when every valid candidate regresses. On a
    tie, the first candidate is accepted, matching AHE's variant ordering.
    Files for rejected variants stay on disk for auditing; the adaptor closes
    their live Agent processes after this function succeeds.
    """

    parent = state.current_workspace.resolve()
    if batch.parent_workspace.resolve() != parent:
        raise RuntimeError("AHE parent changed after candidate generation")
    if not math.isfinite(batch.parent_score):
        raise ValueError("AHE parent score must be finite")
    parent_status, parent_score = _evaluation(parent_evaluation, "parent")
    if parent_status != "ok" or parent_score is None:
        raise ValueError("AHE selection needs a successful parent evaluation")
    if not math.isclose(parent_score, batch.parent_score, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Parent evaluation score differs from the refinement score")
    if not isinstance(candidate_evaluations, Mapping):
        raise TypeError("candidate_evaluations must be a mapping")
    scores, eligible = _candidate_scores(batch, candidate_evaluations)

    # max() is stable: an equal-scoring variant earlier in the batch wins.
    winner, best_score = max(eligible, key=lambda item: item[1]) if eligible else (None, None)
    accepted = winner is not None and best_score >= parent_score
    chosen = winner if accepted else None
    reason = (
        "candidate_not_below_parent" if accepted else
        "no_successful_candidates" if not eligible else "all_candidates_below_parent"
    )
    report_path = Path(output_dir).expanduser().resolve() / "ahe_selection.json"
    if report_path.exists():
        raise FileExistsError(f"AHE selection report already exists: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    outcome = SelectionOutcome(
        "committed" if accepted else "unchanged",
        chosen.candidate_id if chosen else None,
        parent_score, best_score if accepted else parent_score,
        scores, reason, report_path,
    )
    report_path.write_text(json.dumps({
        "refinement_id": batch.refinement_id,
        "status": outcome.status,
        "reason": reason,
        "parent_workspace": str(parent),
        "parent_score": parent_score,
        "candidate_scores": scores,
        "committed_candidate_id": outcome.committed_candidate_id,
        "selected_score": outcome.selected_score,
        "selected_workspace": str(chosen.workspace) if chosen else str(parent),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if chosen is not None:
        state.current_workspace = chosen.workspace.resolve()
    return outcome
