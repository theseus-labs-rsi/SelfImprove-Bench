"""Expose the benchmark's build interface for one AHE harness state."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
from uuid import uuid4

from .agent import AHEAgent
from .init import AHEState


def build(
    state: AHEState,
    *,
    workspace: str | Path | None = None,
    model: str | None = None,
    timeout_seconds: float = 300,
    llm_environment: Mapping[str, str] | None = None,
) -> AHEAgent:
    """Initialize a NexAU Agent for the selected state or a pending candidate.

    The returned agent binds to this workspace even if ``select()`` later
    updates ``state.current_workspace``.  Rebuilding after selection is how
    the benchmark obtains the next generation.  All workspaces must belong
    to the run created by ``init.initialize()``.
    """

    selected = Path(workspace or state.current_workspace).expanduser().resolve()
    if state.run_dir not in selected.parents:
        raise ValueError(f"AHE workspace must be inside {state.run_dir}: {selected}")
    if not (selected / "code_agent.yaml").is_file():
        raise FileNotFoundError(f"AHE agent config is missing in {selected}")
    artifact_dir = state.run_dir / "agent_calls" / uuid4().hex
    return AHEAgent(
        selected, artifact_dir, model=model, timeout_seconds=timeout_seconds,
        llm_environment=llm_environment,
    )
