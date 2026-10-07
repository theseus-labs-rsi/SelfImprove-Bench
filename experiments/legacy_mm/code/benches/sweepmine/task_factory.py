"""Canonical Sweepmine difficulty schema and task construction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


MANIFEST_PATH = Path(__file__).with_name("difficulty_manifest.json")


@dataclass(frozen=True, slots=True)
class SweepmineDifficulty:
    rows: int
    cols: int
    mine_count: int
    max_steps: int

    def __post_init__(self) -> None:
        values = (self.rows, self.cols, self.mine_count, self.max_steps)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("Sweepmine difficulty values must be integers")
        if self.rows <= 0 or self.cols <= 0 or self.max_steps <= 0:
            raise ValueError("rows, cols, and max_steps must be positive")
        if self.mine_count <= 0 or self.mine_count >= self.rows * self.cols:
            raise ValueError("mine_count must be within the board capacity")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SweepmineDifficulty":
        return cls(
            rows=value["rows"],
            cols=value["cols"],
            mine_count=value["mine_count"],
            max_steps=value["max_steps"],
        )


@lru_cache(maxsize=1)
def _cached_manifest() -> dict[int, SweepmineDifficulty]:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("difficulties"), dict):
        raise ValueError("invalid Sweepmine difficulty manifest")
    return {
        int(level): SweepmineDifficulty.from_mapping(options)
        for level, options in raw["difficulties"].items()
    }


def load_difficulty_manifest() -> dict[int, SweepmineDifficulty]:
    """Return a copy of the validated static difficulty table."""
    return dict(_cached_manifest())


def get_difficulty(difficulty: int) -> SweepmineDifficulty:
    try:
        return _cached_manifest()[difficulty]
    except KeyError as exc:
        available = ", ".join(str(level) for level in sorted(_cached_manifest()))
        raise ValueError(
            f"Unknown Sweepmine difficulty {difficulty}; available: {available}"
        ) from exc


def build_task_descriptor(
    difficulty: int,
    *,
    seed: int,
    task_id: str,
    options: SweepmineDifficulty | Mapping[str, Any],
) -> dict[str, Any]:
    """Build the descriptor consumed by ``SweepmineCore``."""
    config = options if isinstance(options, SweepmineDifficulty) else SweepmineDifficulty.from_mapping(options)
    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "seed": seed,
        "rows": config.rows,
        "cols": config.cols,
        "mine_count": config.mine_count,
        "max_steps": config.max_steps,
    }


__all__ = [
    "SweepmineDifficulty",
    "build_task_descriptor",
    "get_difficulty",
    "load_difficulty_manifest",
]
