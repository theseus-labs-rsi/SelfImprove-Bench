"""Farm implementation of the shared benchmark game contract."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .farm_planner import generate_task, simulate_plan


GAME_NAME = "farm"
PROMPT_FILE = Path(__file__).parent / "prompts" / "farm_planning_prompt.md"


def generate(difficulty: int, *, seed: int, task_id: str) -> dict[str, Any]:
    return generate_task(difficulty, seed=seed, task_id=task_id)


def build_prompt(task: Mapping[str, Any]) -> str:
    visible = {
        key: value
        for key, value in task.items()
        if key not in {"reference_daily_profits", "reference_profit", "reference_method"}
    }
    instructions = PROMPT_FILE.read_text(encoding="utf-8").rstrip()
    return (
        f"{instructions}\n\n## Current task\n```json\n"
        f"{json.dumps(visible, ensure_ascii=False, indent=2)}\n```\n\n"
        "This is a non-interactive call. Return only the requested JSON object.\n"
    )


def validate_action(action: dict[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    if action.get("task_id") not in (None, task.get("task_id")):
        raise ValueError("plan task_id does not match the task")
    if not isinstance(action.get("days"), list):
        raise ValueError("farm plan must contain a days array")
    return action


def evaluate(
    task: Mapping[str, Any],
    request: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Request one complete plan and normalize the simulation result."""
    metrics: dict[str, Any] = {}
    interactions: list[dict[str, Any]] = []
    try:
        action = request(
            task,
            prompt=build_prompt(task),
            validator=validate_action,
            metrics=metrics,
        )
        simulation = simulate_plan(dict(task), action)
        relative_score = simulation.get("relative_score")
        interactions.append(
            {
                "step_index": 0,
                "input": dict(task),
                "action": action,
                "environment_response": simulation,
                "success": True,
                "relative_score": relative_score,
                "feedback": "Farm plan executed successfully.",
                "llm_metrics": metrics,
            }
        )
        return {
            "status": "ok",
            "game": GAME_NAME,
            "task_id": task.get("task_id"),
            "difficulty": task.get("difficulty"),
            "relative_score": relative_score,
            "raw_score": simulation.get("profit"),
            "reference_score": simulation.get("reference_profit"),
            "success": True,
            "agent_output": action,
            "simulation": simulation,
            "interactions": interactions,
            "llm_metrics": metrics,
        }
    except Exception as exc:
        interactions.append(
            {
                "step_index": 0,
                "input": dict(task),
                "action": None,
                "environment_response": {"error": str(exc)},
                "success": False,
                "relative_score": 0.0,
                "feedback": str(exc),
                "llm_metrics": metrics,
            }
        )
        return {
            "status": "error",
            "game": GAME_NAME,
            "task_id": task.get("task_id"),
            "difficulty": task.get("difficulty"),
            "relative_score": 0.0,
            "raw_score": None,
            "reference_score": task.get("reference_profit"),
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "agent_output": None,
            "simulation": None,
            "interactions": interactions,
            "llm_metrics": metrics,
        }
