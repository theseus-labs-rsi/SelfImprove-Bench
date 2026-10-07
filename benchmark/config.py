"""Configuration schema for deterministic task orchestration."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benches.heros_journey.task_factory import load_difficulty_manifest as load_hero_difficulties
from benches.sweepmine.task_factory import load_difficulty_manifest


SUPPORTED_GAMES = ("farm", "combat", "sweepmine", "heros_journey")
SUPPORTED_PROVIDERS = (
    "codex",
    "claude",
    "deepseek",
    "openai",
    "ace",
    "baseline",
    "hyperagents",
    "self_harness",
    "ahe",
)
EVOLUTION_WINDOW = 1
HOLDOUT_TASKS_PER_DIFFICULTY = 5
# 静态基线（provider="baseline"）可选的执行后端
BACKEND_CHOICES = ("codex", "claude", "dsh")


@dataclass(frozen=True)
class ScheduleItem:
    game: str
    difficulty: int
    count: int
    replay: bool = False


@dataclass(frozen=True)
class ReplayConfig:
    every: int = 0
    lag: int = 0

    @property
    def enabled(self) -> bool:
        return self.every > 0 and self.lag > 0


@dataclass(frozen=True)
class HoldoutItem:
    game: str
    difficulty: int
    count: int = 1


@dataclass(frozen=True)
class AgentConfig:
    provider: str = "codex"
    backend: str = "codex"
    model: str | None = None
    timeout: float = 300.0
    max_attempts: int = 3
    show_output: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int
    schedule: tuple[ScheduleItem, ...]
    holdout: tuple[HoldoutItem, ...]
    replay: ReplayConfig = ReplayConfig()
    agent: AgentConfig = AgentConfig()
    evolution_window: int = EVOLUTION_WINDOW
    output_dir: str = "benchmark_results"
    continue_on_error: bool = True
    holdout_seed: int = 100
    holdout_tasks_per_difficulty: int = HOLDOUT_TASKS_PER_DIFFICULTY
    game_options: dict[str, Any] = field(default_factory=dict)
    online_evolution: bool = False
    holdout_manifest: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive(value: Any, label: str, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must not exceed {maximum}")
    return value


def _game(value: Any, label: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_GAMES:
        raise ValueError(f"{label} must be one of: {', '.join(SUPPORTED_GAMES)}")
    return normalized


def _schedule_item(value: Any, index: int) -> ScheduleItem:
    if not isinstance(value, dict):
        raise ValueError(f"schedule[{index}] must be an object")
    return ScheduleItem(
        game=_game(value.get("game"), f"schedule[{index}].game"),
        difficulty=_positive(value.get("difficulty"), f"schedule[{index}].difficulty", maximum=5),
        count=_positive(value.get("count"), f"schedule[{index}].count"),
        replay=bool(value.get("replay", False)),
    )


def _holdout_item(value: Any, index: int, pool_size: int) -> HoldoutItem:
    if not isinstance(value, dict):
        raise ValueError(f"holdout[{index}] must be an object")
    return HoldoutItem(
        game=_game(value.get("game"), f"holdout[{index}].game"),
        difficulty=_positive(value.get("difficulty"), f"holdout[{index}].difficulty", maximum=5),
        count=_positive(value.get("count", 1), f"holdout[{index}].count", maximum=pool_size),
    )


def config_from_mapping(value: dict[str, Any]) -> BenchmarkConfig:
    if not isinstance(value, dict):
        raise ValueError("configuration must be an object")
    seed = value.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    pool_size = _positive(
        value.get("holdout_tasks_per_difficulty", HOLDOUT_TASKS_PER_DIFFICULTY),
        "holdout_tasks_per_difficulty",
    )
    schedule_raw = value.get("schedule")
    if not isinstance(schedule_raw, list):
        raise ValueError("schedule must be an array")
    holdout_raw = value.get("holdout")
    if not isinstance(holdout_raw, list) or not holdout_raw:
        raise ValueError("holdout must be a non-empty array")

    replay_raw = value.get("replay", {})
    if not isinstance(replay_raw, dict):
        raise ValueError("replay must be an object")
    every = replay_raw.get("every", 0)
    lag = replay_raw.get("lag", 0)
    for raw, label in ((every, "replay.every"), (lag, "replay.lag")):
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    if bool(every) != bool(lag):
        raise ValueError("replay.every and replay.lag must both be zero or both be positive")
    schedule = tuple(_schedule_item(item, index) for index, item in enumerate(schedule_raw))
    holdout = tuple(_holdout_item(item, index, pool_size) for index, item in enumerate(holdout_raw))
    if any(item.replay for item in schedule) and not (every and lag):
        raise ValueError(
            "schedule enables replay, so replay.every and replay.lag must be positive"
        )

    online_evolution = bool(value.get("online_evolution", False))

    agent_raw = value.get("agent", {})
    if not isinstance(agent_raw, dict):
        raise ValueError("agent must be an object")
    provider = str(agent_raw.get("provider", "codex")).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"agent.provider must be one of: {', '.join(SUPPORTED_PROVIDERS)}")
    if not schedule_raw and provider != "baseline":
        raise ValueError("schedule must be non-empty unless agent.provider is baseline")
    holdout_manifest = value.get("holdout_manifest")
    if holdout_manifest is not None:
        if not isinstance(holdout_manifest, str) or not holdout_manifest.strip():
            raise ValueError("holdout_manifest must be a non-empty path")
        if provider != "baseline" or schedule_raw:
            raise ValueError("holdout_manifest requires baseline with an empty schedule")
    timeout = agent_raw.get("timeout", 300)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("agent.timeout must be positive")
    max_attempts = _positive(agent_raw.get("max_attempts", 3), "agent.max_attempts", maximum=3)
    backend = str(agent_raw.get("backend", "codex")).strip().lower()
    if backend not in BACKEND_CHOICES:
        raise ValueError(f"agent.backend must be one of: {', '.join(BACKEND_CHOICES)}")
    window = _positive(value.get("evolution_window", EVOLUTION_WINDOW), "evolution_window")
    game_options_raw = value.get("game_options", {})
    if game_options_raw:
        raise ValueError(
            "game_options is no longer configurable; edit the bench difficulty manifest"
        )
    sweepmine = load_difficulty_manifest()
    hero = load_hero_difficulties()
    requested_games = {item.game for item in (*schedule, *holdout)}
    interactive_providers = {"ace", "baseline", "hyperagents", "self_harness", "ahe"}
    if "sweepmine" in requested_games and provider not in interactive_providers:
        raise ValueError(
            "Sweepmine requires an RSI agent provider: ace, baseline, hyperagents, self_harness, or ahe"
        )
    if "heros_journey" in requested_games and provider not in interactive_providers:
        raise ValueError(
            "Hero's Journey requires an RSI agent provider: ace, baseline, hyperagents, self_harness, or ahe"
        )
    if online_evolution:
        if provider not in {"ace", "ahe", "hyperagents"}:
            raise ValueError(
                "online_evolution currently supports ace, ahe, or hyperagents"
            )
        if len(schedule) != 1:
            raise ValueError("online_evolution requires exactly one schedule item")
    for item in (*schedule, *holdout):
        if item.game == "sweepmine" and item.difficulty not in sweepmine:
            available = ", ".join(str(level) for level in sorted(sweepmine))
            raise ValueError(
                f"unknown Sweepmine difficulty {item.difficulty}; available: {available}"
            )
        if item.game == "heros_journey" and item.difficulty not in hero:
            available = ", ".join(str(level) for level in sorted(hero))
            raise ValueError(
                f"unknown Hero's Journey difficulty {item.difficulty}; available: {available}"
            )

    return BenchmarkConfig(
        seed=seed,
        schedule=schedule,
        holdout=holdout,
        replay=ReplayConfig(every=every, lag=lag),
        agent=AgentConfig(
            provider=provider,
            backend=backend,
            model=agent_raw.get("model"),
            timeout=float(timeout),
            max_attempts=max_attempts,
            show_output=bool(agent_raw.get("show_output", False)),
        ),
        evolution_window=window,
        output_dir=str(value.get("output_dir", "benchmark_results")),
        continue_on_error=bool(value.get("continue_on_error", True)),
        holdout_seed=int(value.get("holdout_seed", 100)),
        holdout_tasks_per_difficulty=pool_size,
        game_options={
            "sweepmine": {"difficulties": sweepmine},
            "heros_journey": {"difficulties": hero},
        },
        online_evolution=online_evolution,
        holdout_manifest=holdout_manifest,
    )


def load_config(path: str | Path) -> BenchmarkConfig:
    source = Path(path)
    if source.suffix.lower() == ".toml":
        with source.open("rb") as handle:
            value = tomllib.load(handle)
    else:
        value = json.loads(source.read_text(encoding="utf-8"))
    return config_from_mapping(value)
