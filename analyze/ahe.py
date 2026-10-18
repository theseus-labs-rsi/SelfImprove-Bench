"""Analyze AHE candidates against their actual retained parent snapshots."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .ace import refine_snapshot, summarize_analysis
from .llm import analyze_artifact
from .reader import collect_ahe_artifacts


def _write_state(path: Path, state: dict[str, Any]) -> None:
    """Save an inspectable intermediate state after each iteration finishes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_node(source: dict[str, Any], snapshot_ids: list[str]) -> dict[str, Any]:
    """Create the fields shared by every AHE node."""
    return {
        "node_id": source["node_id"],
        "parent_id": source["parent_id"],
        "accepted": source["accepted"],
        "selection_status": source["selection_status"],
        "iteration": source["iteration"],
        "variant": source["variant"],
        "artifact_dir": source["artifact_dir"],
        **{key: source[key] for key in ("parent_score", "current_score", "score_delta")},
        "snapshot_opinion_ids": list(snapshot_ids),
        "created_opinion_ids": [],
        "unresolved_opinions": [],
    }


def _analyze_node(
    source: dict[str, Any],
    state: dict[str, Any],
    parent_snapshot: list[str],
    call_options: dict[str, Any],
) -> dict[str, Any]:
    """Analyze one round of the AHE summary and the workspace incremental diff."""
    node = _base_node(source, parent_snapshot)
    telemetry: dict[str, Any] = {}
    prior = [state["opinions"][item] for item in parent_snapshot]
    result = analyze_artifact(
        "AHE", source["node_id"], source["artifact"], prior,
        telemetry=telemetry, **call_options,
    )
    refined = refine_snapshot(
        state["opinions"], parent_snapshot, result, source["node_id"],
    )
    node.update(refined)
    node.update({
        "status": "completed",
        "analysis": result.model_dump(mode="json"),
        "summary": summarize_analysis(result),
        "telemetry": telemetry,
    })
    return node


def run_ahe_analysis(
    experiment_dir: str | Path,
    output_path: str | Path,
    *,
    model: str | None = None,
    limit: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    client: Any = None,
    on_node: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Analyze all proposals while inheriting only accepted candidate snapshots."""
    sources = collect_ahe_artifacts(experiment_dir)
    if limit is not None:
        sources = sources[:limit]
    destination = Path(output_path).resolve()
    state: dict[str, Any] = {
        "schema_version": 7,
        "method": "AHE",
        "experiment_dir": str(Path(experiment_dir).resolve()),
        "run_id": sources[0]["run_id"] if sources else None,
        "model": model,
        "status": "running",
        "opinions": {},
        "nodes": [],
        "final_snapshot_opinion_ids": [],
    }
    snapshots: dict[str, list[str]] = {}
    options = {
        "model": model, "api_key": api_key, "base_url": base_url,
        "timeout": timeout, "client": client,
    }

    for source in sources:
        parent_id = source["parent_id"]
        snapshot_ids = snapshots[parent_id] if parent_id is not None else []
        node = _base_node(source, snapshot_ids)
        try:
            node = _analyze_node(source, state, snapshot_ids, options)
        except Exception as exc:
            node.update({"status": "failed", "error": str(exc)})
            state["nodes"].append(node)
            state["status"] = "failed"
            _write_state(destination, state)
            raise
        state["nodes"].append(node)
        snapshots[source["node_id"]] = node["snapshot_opinion_ids"]
        if source["accepted"]:
            state["final_snapshot_opinion_ids"] = list(node["snapshot_opinion_ids"])
        _write_state(destination, state)
        if on_node:
            on_node(node)
    state["status"] = "completed"
    _write_state(destination, state)
    return state
