"""Task descriptors for external GameWorld browser-game episodes."""

from __future__ import annotations

from typing import Any, Mapping

from .asset_paths import ASSETS_ROOT

GAME_NAME = "gameworld"


def generate(
    difficulty: int,
    *,
    seed: int,
    task_id: str,
    phase: str = "train",
    index: int = 0,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one explicit GameWorld task from the CSD benchmark manifest.

    GameWorld tasks are not synthesized from a numeric difficulty. The CSD
    manifest therefore supplies disjoint train and holdout pools explicitly.
    ``difficulty`` remains in the outer CSD contract for shared metrics.
    ``model_spec`` selects a GameWorld catalog profile; the RSI API model
    is configured in the method TOML.
    """

    raw_options = dict(options or {})
    settings = dict(raw_options.get("gameworld", raw_options))
    task_roots = settings.get("task_assets_roots", {})
    if not isinstance(task_roots, Mapping):
        raise ValueError("gameworld.task_assets_roots must be an object")
    pool = settings.get(phase)
    if not isinstance(pool, list) or not pool:
        raise ValueError(f"gameworld.{phase} must be a non-empty task list")
    if index >= len(pool):
        raise ValueError(
            f"gameworld {phase} pool has {len(pool)} tasks but index {index} was requested"
        )
    selected = pool[index]
    if not isinstance(selected, Mapping):
        raise ValueError(f"gameworld.{phase}[{index}] must be an object")
    game_id = str(selected.get("game") or "").strip()
    game_task_id = str(selected.get("task") or "").strip()
    if not game_id or not game_task_id:
        raise ValueError(f"gameworld.{phase}[{index}] requires game and task")
    result = {
        "task_id": task_id,
        "difficulty": int(difficulty),
        "seed": int(seed),
        "game_id": game_id,
        "game_task_id": game_task_id,
        "framework": str(settings.get("framework") or "").strip().lower() or None,
        "model_spec": str(settings.get("model_spec") or "").strip() or None,
        "max_steps": int(settings.get("max_steps", 100)),
        # Shared game code and phase-specific YAML assets are selected separately.
        "gameworld_root": str(selected.get("gameworld_root") or settings.get("gameworld_root") or "").strip() or None,
        "task_assets_root": str(
            selected.get("task_assets_root") or task_roots.get(phase) or settings.get("task_assets_root")
            or (ASSETS_ROOT / "tasks" / phase
                if not (selected.get("gameworld_root") or settings.get("gameworld_root")) else "")
        ).strip() or None,
        "reasoning_effort": str(settings.get("reasoning_effort") or "").strip() or None,
        "resumable_steps": settings.get("resumable_steps", False) is True,
    }
    # Preserve the opt-in integrity audit interval in the task/source fingerprint.
    if "step_integrity_audit_interval" in settings:
        result["step_integrity_audit_interval"] = int(settings["step_integrity_audit_interval"])
    return result


__all__ = ["GAME_NAME", "generate"]
