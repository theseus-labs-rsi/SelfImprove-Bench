"""Deterministic holdout generation and cross-game orchestration."""

from __future__ import annotations

import hashlib
import json
from random import Random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from benches.combat import game as combat_game
from benches.farm import game as farm_game
from benches.heros_journey import game as hero_game
from benches.sweepmine import game as sweepmine_game

from .config import BenchmarkConfig


GAMES = {
    "farm": farm_game,
    "combat": combat_game,
    "sweepmine": sweepmine_game,
    "heros_journey": hero_game,
}


@dataclass(frozen=True)
class PlannedTask:
    run_id: str
    phase: str
    game: str
    difficulty: int
    task: dict[str, Any]
    task_seed: int
    group_index: int
    replay: bool = False
    replay_of: str | None = None
    replay_gap: int | None = None
    generation_index: int | None = None

    def manifest_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["task_fingerprint"] = task_fingerprint(self.task)
        return value


@dataclass(frozen=True)
class TaskManifest:
    config_hash: str
    holdout_pool: tuple[PlannedTask, ...]
    training: tuple[PlannedTask, ...]
    holdout: tuple[PlannedTask, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config_hash": self.config_hash,
            "holdout_pool": [item.manifest_record() for item in self.holdout_pool],
            "training": [item.manifest_record() for item in self.training],
            "holdout": [item.manifest_record() for item in self.holdout],
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "TaskManifest":
        """Restore the original task identities when resuming a saved run."""
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported task manifest schema")

        def restore(raw: dict[str, Any]) -> PlannedTask:
            record = dict(raw)
            fingerprint = record.pop("task_fingerprint", None)
            task = PlannedTask(**record)
            if fingerprint != task_fingerprint(task.task):
                raise ValueError(f"Task fingerprint changed: {task.run_id}")
            return task

        return cls(
            config_hash=str(value["config_hash"]),
            holdout_pool=tuple(restore(item) for item in value["holdout_pool"]),
            training=tuple(restore(item) for item in value["training"]),
            holdout=tuple(restore(item) for item in value["holdout"]),
        )


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(_canonical(part) for part in parts).encode("utf-8")).hexdigest()


def _seed(master: int, *parts: Any) -> int:
    return int(_hash(master, *parts)[:16], 16)


def task_fingerprint(task: dict[str, Any]) -> str:
    payload = {key: value for key, value in task.items() if key != "task_id"}
    return _hash(payload)


def _make_task(
    *,
    phase: str,
    game: str,
    difficulty: int,
    index: int,
    group_index: int,
    used: set[str],
    seed_root: int,
    generation_index: int | None = None,
) -> PlannedTask:
    attempt = 0
    while True:
        task_seed = _seed(seed_root, phase, game, difficulty, index, attempt)
        task_id = f"{phase}-{game}-d{difficulty}-{index + 1:04d}"
        task = GAMES[game].generate(difficulty, seed=task_seed, task_id=task_id)
        fingerprint = task_fingerprint(task)
        if fingerprint not in used:
            used.add(fingerprint)
            return PlannedTask(
                run_id=task_id,
                phase=phase,
                game=game,
                difficulty=difficulty,
                task=task,
                task_seed=task_seed,
                group_index=group_index,
                generation_index=generation_index,
            )
        attempt += 1
        if attempt > 10_000:
            raise RuntimeError(f"unable to generate a unique {game} difficulty {difficulty} task")


def _stored_holdout(
    config: BenchmarkConfig,
    game: str,
    difficulty: int,
    local_index: int,
    used: set[str],
) -> PlannedTask | None:
    """Reuse the original seed-100 holdout files when they match the config."""
    if config.holdout_seed != 100 or config.holdout_tasks_per_difficulty > 5:
        return None
    if game not in {"farm", "combat"}:
        return None
    game_dir = Path(GAMES[game].__file__).resolve().parent
    path = game_dir / "task" / f"difficulty_{difficulty}_task_{local_index + 1:02d}.json"
    if not path.is_file():
        return None
    task = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(task, dict) or task.get("difficulty") != difficulty:
        return None
    fingerprint = task_fingerprint(task)
    if fingerprint in used:
        return None
    used.add(fingerprint)
    return PlannedTask(
        run_id=f"holdout-{game}-d{difficulty}-{local_index + 1:02d}",
        phase="holdout",
        game=game,
        difficulty=difficulty,
        task=task,
        task_seed=100 + (difficulty - 1) * 5 + local_index,
        group_index=difficulty - 1,
    )


def build_manifest(config: BenchmarkConfig) -> TaskManifest:
    """Build the full benchmark deterministically, creating holdouts first."""
    config_hash = _hash(config.as_dict())
    if config.holdout_manifest is not None:
        source_path = Path(config.holdout_manifest)
        source = TaskManifest.from_mapping(json.loads(source_path.read_text(encoding="utf-8")))
        selected: list[PlannedTask] = []
        for group_index, spec in enumerate(config.holdout):
            matching = [task for task in source.holdout
                        if (task.game, task.difficulty) == (spec.game, spec.difficulty)]
            if len(matching) < spec.count:
                raise ValueError(f"Holdout manifest has too few {spec.game} d{spec.difficulty} tasks")
            selected.extend(replace(task, group_index=group_index)
                            for task in matching[:spec.count])
        return TaskManifest(config_hash, tuple(selected), (), tuple(selected))
    used: set[str] = set()
    pool: list[PlannedTask] = []
    pool_lookup: dict[tuple[str, int], list[PlannedTask]] = {}
    pool_index = 0
    requested = {(item.game, item.difficulty) for item in config.holdout}
    for game, difficulty in sorted(requested):
            key = (game, difficulty)
            pool_lookup[key] = []
            for local_index in range(config.holdout_tasks_per_difficulty):
                item = _stored_holdout(
                    config, game, difficulty, local_index, used
                ) or _make_task(
                    phase="holdout",
                    game=game,
                    difficulty=difficulty,
                    index=pool_index,
                    group_index=difficulty - 1,
                    used=used,
                    seed_root=config.holdout_seed,
                )
                pool.append(item)
                pool_lookup[key].append(item)
                pool_index += 1

    normal: list[PlannedTask] = []
    orchestrated: list[PlannedTask] = []
    generated_index = 0
    replay_index = 0
    for group_index, spec in enumerate(config.schedule):
        for _ in range(spec.count):
            if config.repeat and normal:
                first = normal[0]
                item = replace(first, run_id=f"repeat-{generated_index + 1:04d}-{first.run_id}")
            else:
                item = _make_task(
                    phase="train",
                    game=spec.game,
                    difficulty=spec.difficulty,
                    index=generated_index,
                    group_index=group_index,
                    used=used,
                    seed_root=config.seed,
                )
            generated_index += 1
            normal.append(item)
            orchestrated.append(item)
            if (
                spec.replay
                and config.replay.enabled
                and len(normal) % config.replay.every == 0
                and len(normal) > config.replay.lag
            ):
                original = normal[-config.replay.lag - 1]
                replay_index += 1
                orchestrated.append(
                    PlannedTask(
                        run_id=f"replay-{replay_index:04d}-{original.run_id}",
                        phase="train",
                        game=original.game,
                        difficulty=original.difficulty,
                        task=original.task,
                        task_seed=original.task_seed,
                        group_index=group_index,
                        replay=True,
                        replay_of=original.run_id,
                        replay_gap=config.replay.lag,
                        generation_index=original.generation_index,
                    )
                )

    if config.random:
        Random(config.seed).shuffle(orchestrated)

    selected_holdout: list[PlannedTask] = []
    for group_index, spec in enumerate(config.holdout):
        for original in pool_lookup[(spec.game, spec.difficulty)][: spec.count]:
            selected_holdout.append(
                PlannedTask(
                    run_id=f"eval-{group_index + 1:02d}-{original.run_id}",
                    phase="holdout",
                    game=original.game,
                    difficulty=original.difficulty,
                    task=original.task,
                    task_seed=original.task_seed,
                    group_index=group_index,
                )
            )
    return TaskManifest(
        config_hash=config_hash,
        holdout_pool=tuple(pool),
        training=tuple(orchestrated),
        holdout=tuple(selected_holdout),
    )
