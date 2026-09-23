"""Hero's Journey implementation of the shared task-generation contract."""

from __future__ import annotations

from typing import Any

from .task_factory import build_task_descriptor


GAME_NAME = "heros_journey"


def generate(difficulty: int, *, seed: int, task_id: str) -> dict[str, Any]:
    return build_task_descriptor(difficulty, seed=seed, task_id=task_id)


__all__ = ["GAME_NAME", "generate"]
