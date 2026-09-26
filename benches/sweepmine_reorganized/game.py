"""Deterministic task descriptor for the one-call Sweepmine induction game."""

from typing import Any

from .core_generator import core_size


GAME_NAME = "sweepmine_reorganized"


def generate(difficulty: int, *, seed: int, task_id: str) -> dict[str, Any]:
    """Keep hidden boards out of the persisted task descriptor."""
    core_size(difficulty)
    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "seed": seed,
        "rule_version": "layered_pointwise_v1",
        "task_counts": 1,
        "example_counts": 5,
        "core_mines": 4,
    }
