"""Small adapters between benchmark task descriptors and game contracts."""

from __future__ import annotations

import statistics
from dataclasses import replace
from typing import Any, Callable, Mapping

from benchmark.contracts import (
    ExecutionStatus,
    EvaluationOutcome,
    FeedbackResult,
    TaskEvaluation,
    to_jsonable,
)

from benches.combat import game as combat_game
from benches.farm import game as farm_game
try:
    from benches.heros_journey.benchmark import HeroJourneyBenchmark
    from benches.heros_journey.task_factory import HeroJourneyDifficulty
except ModuleNotFoundError:
    HeroJourneyBenchmark = None
    HeroJourneyDifficulty = None
from benches.sweepmine.core import SweepmineCore
from benches.sweepmine.play import play_feedback
try:
    from benches.sweepmine_reorganized.play import play_feedback as play_sweepmine_reorganized
except ModuleNotFoundError:
    play_sweepmine_reorganized = None


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
    return EvaluationOutcome(record=record, feedback=feedback)


def evaluate_sweepmine(task: Mapping[str, Any], agent: Any) -> EvaluationOutcome:
    core = SweepmineCore(dict(task))
    return _native_feedback_outcome("sweepmine", task, play_feedback(core, agent))


def evaluate_sweepmine_reorganized(task: Mapping[str, Any], agent: Any) -> EvaluationOutcome:
    if play_sweepmine_reorganized is None:
        raise RuntimeError("The optional sweepmine_reorganized bench is not present in this checkout")
    feedback = play_sweepmine_reorganized(task, agent)
    return _native_feedback_outcome("sweepmine_reorganized", task, feedback)


def _aggregate_hero_feedback(
    task: Mapping[str, Any], feedbacks: list[FeedbackResult]
) -> FeedbackResult:
    if not feedbacks:
        raise ValueError("Hero's Journey seed suite returned no feedback")
    trajectory = []
    seed_results = []
    for feedback in feedbacks:
        seed = feedback.evaluation.metrics.get("seed")
        seed_results.append({
            "seed": seed,
            "task_id": feedback.task_id,
            "status": feedback.evaluation.execution_status.value,
            "success": feedback.evaluation.success,
            "score": feedback.evaluation.score,
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
                "suite_success_rate": statistics.fmean(
                    float(evaluation.success) for evaluation in evaluations
                ),
                "seed_results": seed_results,
            },
        ),
    )


def evaluate_hero_journey(task: Mapping[str, Any], agent: Any) -> EvaluationOutcome:
    if HeroJourneyBenchmark is None or HeroJourneyDifficulty is None:
        raise RuntimeError("Hero's Journey dependencies are not present in this checkout")
    settings = HeroJourneyDifficulty.from_mapping({
        **task,
        "episode_indices": (task["episode_idx"],),
    })
    benchmark = HeroJourneyBenchmark(
        difficulties={int(task["difficulty"]): settings},
        seeds=[int(seed) for seed in task["seeds"]],
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
    if game == "gameworld":
        if agent is None:
            raise TypeError("gameworld evaluation requires an explicit RSI agent")
        from benches.gameworld.benchmark import GameWorldBenchmark

        return GameWorldBenchmark().evaluate(task, agent)
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
