"""Task catalog loader."""

from __future__ import annotations

from pathlib import Path

from .._yaml import load_yaml_mapping
from ..games import resolve_game_id
from ._base import TaskSpec

_TASKS_DIR = Path(__file__).parent
_GAME_FOLDERS = {"03_astray": "astray", "18_minecraft-clone-glm": "minecraft"}


def _tasks_directory(task_assets_root: str | Path | None = None) -> Path:
    """Separate task YAMLs from code; an external checkout keeps its old default."""
    if task_assets_root is not None:
        return Path(task_assets_root).expanduser().resolve()
    bundled = _TASKS_DIR.parents[3] / "tasks" / "holdout"
    return bundled if bundled.is_dir() else _TASKS_DIR


def _game_tasks_directory(game_id: str, task_assets_root: str | Path | None = None) -> Path:
    root = _tasks_directory(task_assets_root)
    named = root / _GAME_FOLDERS.get(game_id, game_id)
    return named if named.is_dir() else root / game_id


def _get_task_definition_path(
    game_id: str, task_id: str, *, task_assets_root: str | Path | None = None,
) -> Path:
    game_tasks_dir = _game_tasks_directory(game_id, task_assets_root)
    path = game_tasks_dir / f"{task_id}.yaml"
    if not path.exists():
        known = (
            sorted(item.stem for item in game_tasks_dir.glob("*.yaml"))
            if game_tasks_dir.exists()
            else []
        )
        examples = ", ".join(known[:8])
        suffix = ", ..." if len(known) > 8 else ""
        known_text = f" Known examples: {examples}{suffix}" if examples else ""
        raise FileNotFoundError(
            f"Task '{task_id}' not found for game '{game_id}'. "
            f"Task ids must exactly match catalog YAML stems. Searched: {path}.{known_text}"
        )
    return path


def load_task(
    game_id: str, task_id: str, *, task_assets_root: str | Path | None = None,
) -> TaskSpec:
    """Load one task definition for a game."""
    resolved_game_id = resolve_game_id(game_id)
    normalized_task_id = str(task_id or "").strip()
    path = _get_task_definition_path(
        resolved_game_id, normalized_task_id, task_assets_root=task_assets_root,
    )
    task = TaskSpec.from_mapping(load_yaml_mapping(path))
    if task.game_id != resolved_game_id or task.task_id != normalized_task_id:
        raise ValueError(
            "Task YAML id mismatch: "
            f"path expects game_id='{resolved_game_id}', task_id='{normalized_task_id}', "
            f"but file declares game_id='{task.game_id}', task_id='{task.task_id}'."
        )
    return task


def list_tasks(
    game_id: str | None = None, *, task_assets_root: str | Path | None = None,
) -> list[str]:
    """List available task ids."""
    if game_id is None:
        return sorted({path.stem for path in _tasks_directory(task_assets_root).rglob("*.yaml")})

    resolved_game_id = resolve_game_id(game_id)
    game_tasks_dir = _game_tasks_directory(resolved_game_id, task_assets_root)
    if not game_tasks_dir.exists():
        return []
    return sorted(path.stem for path in game_tasks_dir.glob("*.yaml"))
