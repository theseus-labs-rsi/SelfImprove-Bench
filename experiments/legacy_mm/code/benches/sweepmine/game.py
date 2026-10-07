"""Sweepmine implementation of the shared task-generation contract."""

from __future__ import annotations

from typing import Any

from .task_factory import build_task_descriptor, get_difficulty


GAME_NAME = "sweepmine"


def generate(difficulty: int, *, seed: int, task_id: str) -> dict[str, Any]:
    return build_task_descriptor(
        difficulty,
        seed=seed,
        task_id=task_id,
        options=get_difficulty(difficulty),
    )


__all__ = ["GAME_NAME", "generate"]
