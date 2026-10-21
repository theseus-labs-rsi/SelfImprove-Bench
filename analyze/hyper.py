"""Analyze patches along the HyperAgent parent-child tree and save a refined snapshot per candidate."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .ace import refine_snapshot, summarize_analysis
from .llm import AnalysisLLMError, analyze_artifact
from .reader import collect_hyperagent_artifacts


def _write_state(path: Path, state: dict[str, Any]) -> None:
    """Save an inspectable tree state after each candidate finishes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_node(source: dict[str, Any], snapshot_ids: list[str]) -> dict[str, Any]:
    """Create the fields shared by every HyperAgent node."""
    return {
        "node_id": source["node_id"],
        "parent_id": source["parent_id"],
        "task_id": source["task_id"],
        "selected": source["selected"],
        "archived_candidate_id": source.get("archived_candidate_id"),
        "next_parent_id": source.get("next_parent_id"),
        "accepted": source["accepted"],
        "changed": source["changed"],
        "artifact_path": source["artifact_path"],
        **{key: source[key] for key in ("parent_score", "current_score", "score_delta")},
        "snapshot_opinion_ids": list(snapshot_ids),
        "created_opinion_ids": [],
        "unresolved_opinions": [],
    }


def _update_leaves(state: dict[str, Any]) -> None:
    """Record the leaf nodes of the currently processed tree."""
    parent_ids = {
        node["parent_id"] for node in state["nodes"]
        if node["parent_id"] is not None
    }
    state["leaf_node_ids"] = [
        node["node_id"] for node in state["nodes"]
        if node["node_id"] not in parent_ids
    ]


def _parent_snapshot(
    source: dict[str, Any],
    snapshots: dict[str, list[str]],
) -> list[str]:
    """The root starts from an empty snapshot; other nodes strictly inherit from their explicit parent."""
    parent_id = source["parent_id"]
    if parent_id is None:
        return []
    if parent_id not in snapshots:
        raise ValueError(f"Parent node not processed yet: {source['node_id']} -> {parent_id}")
    return list(snapshots[parent_id])


def _analyze_node(
    source: dict[str, Any],
    state: dict[str, Any],
    parent_snapshot: list[str],
    call_options: dict[str, Any],
    method_name: str,
) -> dict[str, Any]:
    """Analyze one candidate; a node with no change directly inherits the parent snapshot."""
    node = _base_node(source, parent_snapshot)
    if not source["changed"]:
        node.update({
            "status": "unchanged", "analysis": None,
            "summary": None, "telemetry": {},
        })
        return node
    if source["artifact"] is None:
        node.update({
            "status": "missing_artifact", "analysis": None,
            "summary": None, "telemetry": {},
        })
        return node

    prior = [state["opinions"][item] for item in parent_snapshot]
    telemetry: dict[str, Any] = {}
    result = analyze_artifact(
        method_name, source["node_id"], source["artifact"], prior,
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


def run_hyper_analysis(
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
    method_name: str = "HyperAgent",
) -> dict[str, Any]:
    """Analyze a HyperAgent-style candidate tree in topological order."""
    collectors = {
        "HyperAgent": collect_hyperagent_artifacts,
    }
    if method_name not in collectors:
        raise ValueError(f"Unsupported tree analysis method: {method_name}")
    sources = collectors[method_name](experiment_dir)
    if limit is not None:
        sources = sources[:limit]
    destination = Path(output_path).resolve()
    state: dict[str, Any] = {
        "schema_version": 6,
        "method": method_name,
        "experiment_dir": str(Path(experiment_dir).resolve()),
        "run_id": sources[0]["run_id"] if sources else None,
        "model": model,
        "status": "running",
        "opinions": {},
        "nodes": [],
        "leaf_node_ids": [],
    }
    snapshots: dict[str, list[str]] = {}
    options = {
        "model": model, "api_key": api_key, "base_url": base_url,
        "timeout": timeout, "client": client,
    }

    for source in sources:
        parent_snapshot = _parent_snapshot(source, snapshots)
        try:
            node = _analyze_node(
                source, state, parent_snapshot, options, method_name,
            )
        except AnalysisLLMError as exc:
            node = _base_node(source, parent_snapshot)
            node.update({
                "status": "skipped_analysis_error",
                "analysis": None,
                "summary": None,
                "telemetry": {},
                "error": str(exc),
            })
        except Exception as exc:
            node = _base_node(source, parent_snapshot)
            node.update({"status": "failed", "error": str(exc)})
            state["nodes"].append(node)
            state["status"] = "failed"
            _update_leaves(state)
            _write_state(destination, state)
            raise
        state["nodes"].append(node)
        snapshots[source["node_id"]] = node["snapshot_opinion_ids"]
        _update_leaves(state)
        _write_state(destination, state)
        if on_node:
            on_node(node)
    has_skipped = any(
        node["status"] == "skipped_analysis_error" for node in state["nodes"]
    )
    state["status"] = "completed_with_errors" if has_skipped else "completed"
    _write_state(destination, state)
    return state
