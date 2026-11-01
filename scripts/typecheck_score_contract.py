"""Small real-contract check; this is not a full runtime type-checking claim."""
from collections.abc import Sequence
from typing import assert_type

from benchmark.contracts import ExecutionStatus, TaskEvaluation


def available_scores(evaluations: Sequence[TaskEvaluation]) -> list[float]:
    scores: list[float] = []
    for evaluation in evaluations:
        score = evaluation.score
        assert_type(score, float | None)
        if evaluation.execution_status == ExecutionStatus.COMPLETED and score is not None:
            scores.append(score)
    return scores


def observed_mean(evaluations: Sequence[TaskEvaluation]) -> float | None:
    scores = available_scores(evaluations)
    return sum(scores) / len(scores) if scores else None
