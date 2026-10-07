"""Combat implementation of the shared benchmark game contract."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .combat_planner import CombatSession, generate_task


GAME_NAME = "combat"
PROMPT_FILE = Path(__file__).parent / "prompts" / "combat_planning_prompt.md"
TOKEN_FIELDS = ("prompt_tokens", "reasoning_tokens", "answer_tokens", "total_tokens")


def generate(difficulty: int, *, seed: int, task_id: str) -> dict[str, Any]:
    return generate_task(difficulty, seed=seed, task_id=task_id)


def build_prompt(stage: Mapping[str, Any]) -> str:
    visible = {
        key: value
        for key, value in stage.items()
        if key not in {"reference_score", "reference_method", "solvability"}
    }
    instructions = PROMPT_FILE.read_text(encoding="utf-8").rstrip()
    return (
        f"{instructions}\n\n## Current stage state\n```json\n"
        f"{json.dumps(visible, ensure_ascii=False, indent=2)}\n```\n\n"
        "This is a non-interactive call. Return only one current-stage JSON decision.\n"
    )


def validate_action(action: dict[str, Any], stage: Mapping[str, Any]) -> dict[str, Any]:
    if action.get("task_id") not in (None, stage.get("task_id")):
        raise ValueError("decision task_id does not match the task")
    if action.get("stage_id") != stage.get("stage_id"):
        raise ValueError(f"decision must target stage {stage.get('stage_id')}")
    if stage.get("decision_type") == "combat":
        if not isinstance(action.get("turns"), list):
            raise ValueError("combat decision must contain a turns array")
    elif stage.get("decision_type") == "shop":
        if not isinstance(action.get("buy_cards"), list) or not isinstance(action.get("heal"), bool):
            raise ValueError("shop decision must contain buy_cards and boolean heal")
    else:
        raise ValueError("unknown combat decision type")
    return action


def _aggregate_metrics(stage_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    tokens: dict[str, Any] = {}
    for field in TOKEN_FIELDS:
        values = [item.get("tokens", {}).get(field) for item in stage_metrics]
        tokens[field] = (
            sum(values)
            if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            else None
        )
    durations = [
        float(item["duration_seconds"])
        for item in stage_metrics
        if isinstance(item.get("duration_seconds"), (int, float))
        and not isinstance(item.get("duration_seconds"), bool)
    ]
    return {
        "stage_call_count": len(stage_metrics),
        "duration_seconds": round(sum(durations), 6),
        "tokens": tokens,
        "stages": stage_metrics,
    }


def evaluate(
    task: Mapping[str, Any],
    request: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Run sequential decisions while exposing only the current reached stage."""
    interactions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    stage_metrics: list[dict[str, Any]] = []
    session: CombatSession | None = None
    try:
        session = CombatSession(task)
        while not session.finished:
            stage = session.current_request()
            metrics: dict[str, Any] = {}
            index = len(interactions)
            interaction = {
                "step_index": index,
                "input": stage,
                "action": None,
                "environment_response": None,
                "success": False,
                "relative_score": None,
                "feedback": None,
                "llm_metrics": metrics,
            }
            interactions.append(interaction)
            try:
                action = request(
                    stage,
                    prompt=build_prompt(stage),
                    validator=validate_action,
                    metrics=metrics,
                )
                interaction["action"] = action
                outcome = session.apply(action)
                interaction["environment_response"] = outcome
                interaction["success"] = (
                    outcome.get("won") if stage["decision_type"] == "combat" else True
                )
                interaction["feedback"] = "Stage decision executed."
            except Exception as exc:
                interaction["environment_response"] = {"error": str(exc)}
                interaction["feedback"] = str(exc)
                raise
            finally:
                stage_metrics.append(metrics)
            decisions.append(action)
        simulation = session.result()
        relative_score = simulation.get("relative_score")
        if interactions:
            interactions[-1]["relative_score"] = relative_score
        return {
            "status": "ok",
            "game": GAME_NAME,
            "task_id": task.get("task_id"),
            "difficulty": task.get("difficulty"),
            "relative_score": relative_score,
            "raw_score": simulation.get("objective_score"),
            "reference_score": simulation.get("reference_score"),
            "success": True,
            "agent_output": {
                "task_id": task.get("task_id"),
                "mode": "sequential_stage_decisions",
                "stages": decisions,
            },
            "simulation": simulation,
            "interactions": interactions,
            "llm_metrics": _aggregate_metrics(stage_metrics),
        }
    except Exception as exc:
        return {
            "status": "error",
            "game": GAME_NAME,
            "task_id": task.get("task_id"),
            "difficulty": task.get("difficulty"),
            "relative_score": 0.0,
            "raw_score": None,
            "reference_score": task.get("reference_score"),
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "agent_output": {
                "task_id": task.get("task_id"),
                "mode": "sequential_stage_decisions",
                "stages": decisions,
            },
            "simulation": session.result() if session is not None else None,
            "interactions": interactions,
            "llm_metrics": _aggregate_metrics(stage_metrics),
        }
