"""Prepare an isolated AHE lineage for the benchmark's four-stage lifecycle.

This module extracts the directory setup from upstream ``evolve.py``.  It
copies the two agent surfaces that later stages need, while leaving the
cloned upstream repository as a read-only source of initial files.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


UPSTREAM_ROOT = Path(__file__).resolve().parent / "agentic-harness-engineering"


@dataclass(slots=True)
class AHEState:
    """Paths owned by one benchmark run; ``current_workspace`` is the parent state.

    Later ``select()`` may point ``current_workspace`` at an accepted candidate.
    The original source directory is never used as a writable workspace.
    """

    run_dir: Path
    current_workspace: Path
    evolve_agent_dir: Path


def apply_agent_limits(config_path: Path, limits: dict[str, int]) -> None:
    """Apply XBR's cost bounds to a run-local YAML, never to upstream sources."""
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid AHE agent configuration: {config_path}")
    for field in ("max_context_tokens", "max_iterations"):
        value = limits[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid AHE limit: {field}")
        config[field] = value
    max_tokens = limits["max_tokens"]
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("Invalid AHE limit: max_tokens")
    llm_config = config.get("llm_config")
    if not isinstance(llm_config, dict):
        raise ValueError(f"Invalid AHE LLM configuration: {config_path}")
    llm_config["max_tokens"] = max_tokens
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def initialize(run_dir: str | Path, *, source_root: str | Path = UPSTREAM_ROOT) -> AHEState:
    """Copy upstream agent surfaces into a new, exclusively owned run directory.

    Upstream ``evolve.py`` creates ``workspace/``, ``evolve_agent/`` and
    ``runs/`` inside an experiment.  We retain that layout because its Evolve
    Agent prompt refers to those paths.  The caller supplies a fresh run path;
    existing paths are rejected so initialization cannot overwrite a lineage.
    """

    source = Path(source_root).expanduser().resolve()
    target = Path(run_dir).expanduser().resolve()
    code_source = source / "agents" / "code_agent_simple"
    evolve_source = source / "agents" / "evolve_agent"
    if not code_source.is_dir() or not evolve_source.is_dir():
        raise FileNotFoundError(f"AHE agent source directories are missing under {source}")
    if target == source or source in target.parents or target.exists():
        raise FileExistsError(f"AHE run directory must be new and outside the source: {target}")

    target.mkdir(parents=True)
    workspace = target / "workspace"
    evolve_agent = target / "evolve_agent"
    shutil.copytree(code_source, workspace)
    shutil.copytree(evolve_source, evolve_agent)
    (target / "runs").mkdir()
    return AHEState(target, workspace, evolve_agent)
