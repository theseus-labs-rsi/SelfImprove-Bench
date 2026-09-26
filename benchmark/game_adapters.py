"""Small adapters between benchmark task descriptors and game contracts."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from benchmark.contracts import (
    ExecutionStatus,
    FeedbackResult,
    TaskEvaluation,
    to_jsonable,
)

from benches.combat import game as combat_game
from benches.farm import game as farm_game
from benches.heros_journey.benchmark import HeroJourneyBenchmark
from benches.heros_journey.task_factory import HeroJourneyDifficulty
from benches.sweepmine.core import SweepmineCore
from benches.sweepmine.play import play_feedback
from benches.sweepmine_reorganized.play import play_feedback as play_sweepmine_reorganized


@dataclass(frozen=True)
class EvaluationOutcome:
    record: dict[str, Any]
    feedback: FeedbackResult | None = None


GAME_MODULES = {"farm": farm_game, "combat": combat_game}


def _native_feedback_outcome(
    game: str, task: Mapping[str, Any], feedback: FeedbackResult
) -> EvaluationOutcome:
    evaluation = feedback.evaluation
    total_tokens = sum(
        (step.agent_call.usage.total_tokens or 0 if step.agent_call and step.agent_call.usage else 0)
        for step in feedback.trajectory
    )
    record = {
        "status": "ok" if evaluation.execution_status.value == "completed" else "error",
        "game": game, "task_id": feedback.task_id,
        "difficulty": task["difficulty"], "relative_score": evaluation.score,
        "raw_score": evaluation.score, "success": evaluation.success,
        "error": None if evaluation.execution_status.value == "completed" else evaluation.feedback,
        "agent_output": None, "simulation": {"metrics": to_jsonable(evaluation.metrics)},
        "interactions": to_jsonable(feedback.trajectory),
        "llm_metrics": {"tokens": {"total_tokens": total_tokens}},
    }
    if game == "heros_journey":
        record["interaction_protocol"] = evaluation.metrics.get("interaction_protocol")
        record["score_metric"] = evaluation.metrics.get("score_metric")
        record["completion_rate"] = evaluation.metrics.get("suite_completion_rate_mean")
    return EvaluationOutcome(record=record, feedback=feedback)


def evaluate_sweepmine(task: Mapping[str, Any], agent: Any) -> EvaluationOutcome:
    core = SweepmineCore(dict(task))
    return _native_feedback_outcome("sweepmine", task, play_feedback(core, agent))


def evaluate_sweepmine_reorganized(task: Mapping[str, Any], agent: Any) -> EvaluationOutcome:
    feedback = play_sweepmine_reorganized(task, agent)
    return _native_feedback_outcome("sweepmine_reorganized", task, feedback)


def _aggregate_hero_feedback(
    task: Mapping[str, Any], feedbacks: list[FeedbackResult]
) -> FeedbackResult:
    if not feedbacks:
        raise ValueError("Hero's Journey seed suite returned no feedback")
    trajectory = []
    seed_results = []
    completion_rates = []
    protocols = set()
    for feedback in feedbacks:
        seed = feedback.evaluation.metrics.get("seed")
        completion_rate = feedback.evaluation.metrics.get("completion_rate")
        protocol = feedback.evaluation.metrics.get("interaction_protocol", "legacy_unlabeled")
        protocols.add(protocol)
        if isinstance(completion_rate, (int, float)) and not isinstance(completion_rate, bool):
            completion_rates.append(float(completion_rate))
        seed_results.append({
            "seed": seed,
            "task_id": feedback.task_id,
            "status": feedback.evaluation.execution_status.value,
            "success": feedback.evaluation.success,
            "score": feedback.evaluation.score,
            "completion_rate": completion_rate,
            "interaction_protocol": protocol,
            "score_metric": feedback.evaluation.metrics.get("score_metric"),
            "parse_error": feedback.evaluation.metrics.get("parse_error"),
        })
        for step in feedback.trajectory:
            trajectory.append(replace(
                step,
                step_index=len(trajectory),
                environment_response={
                    "suite_seed": seed,
                    "suite_task_id": feedback.task_id,
                    "response": step.environment_response,
                },
            ))

    evaluations = [feedback.evaluation for feedback in feedbacks]
    score = statistics.fmean(evaluation.score for evaluation in evaluations)
    execution_status = (
        ExecutionStatus.ERROR
        if any(evaluation.execution_status == ExecutionStatus.ERROR for evaluation in evaluations)
        else ExecutionStatus.COMPLETED
    )
    messages = [
        f"[seed {item['seed']}] {feedback.evaluation.feedback or ''}"
        for item, feedback in zip(seed_results, feedbacks)
    ]
    return FeedbackResult(
        task_id=str(task["task_id"]),
        task=dict(task),
        trajectory=tuple(trajectory),
        evaluation=TaskEvaluation(
            execution_status=execution_status,
            success=all(evaluation.success for evaluation in evaluations),
            score=score,
            feedback="\n".join(messages),
            metrics={
                "task_type": task["task_type"],
                "suite_size": len(feedbacks),
                "suite_score_mean": score,
                "suite_completion_rate_mean": (
                    statistics.fmean(completion_rates) if completion_rates else None
                ),
                "suite_completion_rate_count": len(completion_rates),
                "interaction_protocol": next(iter(protocols)) if len(protocols) == 1 else "mixed",
                "score_metric": "reference_steps_over_executed_steps_v1",
                "suite_success_rate": statistics.fmean(
                    float(evaluation.success) for evaluation in evaluations
                ),
                "seed_results": seed_results,
            },
        ),
    )


def evaluate_hero_journey(task: Mapping[str, Any], agent: Any) -> EvaluationOutcome:
    settings = HeroJourneyDifficulty.from_mapping({
        **task,
        "episode_indices": (task["episode_idx"],),
    })
    benchmark = HeroJourneyBenchmark(
        difficulties={int(task["difficulty"]): settings},
        seeds=[int(seed) for seed in task["seeds"]],
        max_workers=3 if getattr(agent, "supports_parallel_calls", False) else 1,
    )
    episodes = [
        benchmark.construct_one(
            int(task["difficulty"]),
            seed=int(seed),
            task_id=f"{task['task_id']}-seed-{index + 1}",
            episode_idx=int(task["episode_idx"]),
        )
        for index, seed in enumerate(task["seeds"])
    ]
    feedback = _aggregate_hero_feedback(task, benchmark.play(episodes, agent))
    return _native_feedback_outcome("heros_journey", task, feedback)


def evaluate_task(
    game: str, task: Mapping[str, Any], *, request: Callable[..., dict[str, Any]] | None = None,
    agent: Any | None = None,
) -> EvaluationOutcome:
    if game in {"sweepmine", "sweepmine_reorganized", "heros_journey"}:
        if agent is None:
            raise TypeError(f"{game} evaluation requires an explicit agent")
        if game == "sweepmine":
            return evaluate_sweepmine(task, agent)
        if game == "sweepmine_reorganized":
            return evaluate_sweepmine_reorganized(task, agent)
        return evaluate_hero_journey(task, agent)
    if request is None:
        raise TypeError(f"{game} evaluation requires a request callback")
    return EvaluationOutcome(GAME_MODULES[game].evaluate(task, request))
