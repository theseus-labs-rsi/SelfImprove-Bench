"""Static Hero's Journey difficulty schema and task descriptors."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


MANIFEST_PATH = Path(__file__).with_name("difficulty_manifest.json")


@dataclass(frozen=True, slots=True)
class HeroJourneyDifficulty:
    task_type: str
    logic_version: str | None = None
    seeds_per_task: int = 1
    episode_indices: tuple[int, ...] = (0,)
    action_budget_multiplier: int = 2
    max_runs: int | None = None
    interaction_protocol: str = "action_sequence_v1"
    truncate_window: int | None = None
    episode_mode: str = "standard"
    no_reasoning: bool = False
    initial_currency: int = 1000

    def __post_init__(self) -> None:
        if not self.task_type or not self.episode_indices:
            raise ValueError("task_type and episode_indices must not be empty")
        if self.seeds_per_task <= 0 or self.action_budget_multiplier <= 0 or self.initial_currency <= 0:
            raise ValueError("seeds_per_task, action_budget_multiplier, and initial_currency must be positive")
        if self.max_runs is not None and self.max_runs <= 0:
            raise ValueError("max_runs must be positive when provided")
        if self.interaction_protocol not in {"stepwise_v1", "action_sequence_v1"}:
            raise ValueError("unknown Hero's Journey interaction protocol")
        if self.interaction_protocol == "action_sequence_v1" and self.episode_mode != "standard":
            raise ValueError("action_sequence_v1 requires standard episode_mode")
        if self.truncate_window is not None and self.truncate_window <= 0:
            raise ValueError("truncate_window must be positive when provided")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HeroJourneyDifficulty":
        return cls(
            task_type=str(value["task_type"]),
            logic_version=(
                str(value["logic_version"]) if value.get("logic_version") is not None else None
            ),
            seeds_per_task=int(value.get("seeds_per_task", 1)),
            episode_indices=tuple(int(index) for index in value.get("episode_indices", (0,))),
            action_budget_multiplier=int(value.get("action_budget_multiplier", value.get("num_tries", 2))),
            max_runs=cls._optional_int(value.get("max_runs")),
            interaction_protocol=str(value.get("interaction_protocol", "action_sequence_v1")),
            truncate_window=cls._optional_int(value.get("truncate_window")),
            episode_mode=str(value.get("episode_mode", "standard")),
            no_reasoning=bool(value.get("no_reasoning", False)),
            initial_currency=int(value.get("initial_currency", 1000)),
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)


@lru_cache(maxsize=1)
def _cached_manifest() -> dict[int, HeroJourneyDifficulty]:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("difficulties"), dict):
        raise ValueError("invalid Hero's Journey difficulty manifest")
    return {
        int(level): HeroJourneyDifficulty.from_mapping(options)
        for level, options in raw["difficulties"].items()
    }


def load_difficulty_manifest() -> dict[int, HeroJourneyDifficulty]:
    return dict(_cached_manifest())


def get_difficulty(difficulty: int) -> HeroJourneyDifficulty:
    try:
        return _cached_manifest()[difficulty]
    except KeyError as exc:
        available = ", ".join(str(level) for level in sorted(_cached_manifest()))
        raise ValueError(
            f"Unknown Hero's Journey difficulty {difficulty}; available: {available}"
        ) from exc


def derive_task_seeds(base_seed: int, count: int) -> tuple[int, ...]:
    seeds: list[int] = []
    index = 0
    while len(seeds) < count:
        payload = f"heros_journey\x1f{base_seed}\x1f{index}".encode("utf-8")
        candidate = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        if candidate not in seeds:
            seeds.append(candidate)
        index += 1
    return tuple(seeds)


def build_task_descriptor(difficulty: int, *, seed: int, task_id: str) -> dict[str, Any]:
    settings = get_difficulty(difficulty)
    payload = f"heros_journey\x1f{seed}\x1fepisode".encode("utf-8")
    selection = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    descriptor = asdict(settings)
    descriptor["episode_idx"] = settings.episode_indices[selection % len(settings.episode_indices)]
    descriptor["seeds"] = list(derive_task_seeds(seed, settings.seeds_per_task))
    descriptor.pop("episode_indices")
    return {"task_id": task_id, "difficulty": difficulty, "seed": seed, **descriptor}


__all__ = [
    "HeroJourneyDifficulty",
    "build_task_descriptor",
    "derive_task_seeds",
    "get_difficulty",
    "load_difficulty_manifest",
]
