from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .archive_utils import setup_initial_candidate


@dataclass(frozen=True)
class CandidateSnapshot:
    """Persistent HyperAgent candidate state outside a running container."""

    candidate_id: str | int
    run_dir: Path
    workspace: Path
    root_commit: str
    parent_id: str | int | None = None
    lineage_patches: tuple[Path, ...] = ()


def initialize_agent(
    run_dir: str | Path,
    *,
    initial_source: str | Path | None = None,
    run_config: Mapping[str, Any] | None = None,
    reuse_existing: bool = False,
) -> CandidateSnapshot:
    """Create or restore the persistent initial HyperAgent candidate."""
    resolved_run_dir = Path(run_dir).resolve()
    workspace, root_commit = setup_initial_candidate(
        str(resolved_run_dir),
        copy_root_dir=str(Path(initial_source).resolve()) if initial_source else None,
        resume=reuse_existing,
        run_config=dict(run_config or {}),
    )
    return CandidateSnapshot(
        candidate_id="initial",
        run_dir=resolved_run_dir,
        workspace=Path(workspace),
        root_commit=root_commit,
    )


def with_lineage(
    initial: CandidateSnapshot,
    *,
    candidate_id: str | int,
    parent_id: str | int | None = None,
    lineage_patches: list[str | Path] | tuple[str | Path, ...],
) -> CandidateSnapshot:
    """Describe a candidate reconstructed from the initial workspace and patches."""
    return CandidateSnapshot(
        candidate_id=candidate_id,
        run_dir=initial.run_dir,
        workspace=initial.workspace,
        root_commit=initial.root_commit,
        parent_id=parent_id,
        lineage_patches=tuple(Path(path).resolve() for path in lineage_patches),
    )
