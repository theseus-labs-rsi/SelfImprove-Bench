"""Locate shared GameWorld code and phase-specific task assets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


ASSETS_ROOT = Path(__file__).resolve().parent / "assets"
CORE_ROOT = ASSETS_ROOT / "core"


def resolve_gameworld_root(task: Mapping[str, Any]) -> Path:
    """Explicit task roots and the environment override the bundled core."""
    root = task.get("gameworld_root") or os.environ.get("GAMEWORLD_ROOT") or CORE_ROOT
    return Path(str(root)).expanduser().resolve()


def task_definition_path(task: Mapping[str, Any]) -> Path:
    """Use the same task resolver as GameWorld execution and checkpoint hashing."""
    from catalog.tasks import _get_task_definition_path

    options = (
        {"task_assets_root": task["task_assets_root"]}
        if task.get("task_assets_root") else {}
    )
    return _get_task_definition_path(
        str(task["game_id"]), str(task["game_task_id"]), **options,
    )
