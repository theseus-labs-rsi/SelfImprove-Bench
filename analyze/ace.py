"""Analyze artifacts in ACE task order and save a refined opinion snapshot per node."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .llm import AnalysisOutput, ImprovementOpinion, analyze_artifact
from .reader import collect_ace_artifacts, collect_ace_scores


def _empty_counts(values: list[str]) -> dict[str, int]:
    """Create a classification counter dictionary with a stable order."""
    return {value: 0 for value in values}


def summarize_analysis(output: AnalysisOutput) -> dict[str, Any]:
    """Count all opinions in this round, including duplicates that will later be dropped."""
    relations = _empty_counts([
        "semantically_new", "contradiction_or_revision", "duplicate",
    ])
    relation_unknown = relations.copy()
    scope = _empty_counts([
        "task_specific", "capability_category_general", "fully_general", "undetermined",
    ])
    function = _empty_counts([
        "exploration", "reasoning", "planning", "execution", "undetermined",
    ])
    surface = _empty_counts([
        "prompt_memory", "scaffold_codebase", "meta", "undetermined",
    ])

    for opinion in output.opinions:
        _count_relations(opinion, relations, relation_unknown)
        scope[opinion.scope.value if opinion.scope else "undetermined"] += 1
        surface[opinion.surface.value if opinion.surface else "undetermined"] += 1
        labels = opinion.function or ["undetermined"]
        for label in labels:
            function[label.value if hasattr(label, "value") else label] += 1
    return {
        "opinion_count": len(output.opinions),
        "relations": {**relations, "undetermined": relation_unknown},
        "scope": scope,
        "function": function,
        "surface": surface,
    }


def _count_relations(
    opinion: ImprovementOpinion,
    counts: dict[str, int],
    unknown: dict[str, int],
) -> None:
    """Count the three parallel relations independently; their sum need not equal the number of opinions."""
    for name in counts:
        value = getattr(opinion, name)
        if value is True:
            counts[name] += 1
        elif value is None:
            unknown[name] += 1


def _opinion_record(
    opinion_id: str,
    opinion: ImprovementOpinion,
    node_id: str,
    source_index: int,
    replaced_ids: list[str],
) -> dict[str, Any]:
    """Build the normalized record that enters the global opinion store."""
    return {
        "opinion_id": opinion_id,
        "normalized_opinion": opinion.normalized_opinion,
        "source_evidence": list(opinion.source_evidence),
        "scope": opinion.scope.value if opinion.scope else None,
        "function": [label.value for label in opinion.function],
        "surface": opinion.surface.value if opinion.surface else None,
        "created_at_node": node_id,
        "source_index": source_index,
        "replaces_opinion_ids": replaced_ids,
    }


def refine_snapshot(
    opinion_store: dict[str, dict[str, Any]],
    parent_snapshot_ids: list[str],
    output: AnalysisOutput,
    node_id: str,
) -> dict[str, Any]:
    """Build a child snapshot from the parent snapshot; do not modify the parent list."""
    snapshot_ids = list(parent_snapshot_ids)
    created_ids: list[str] = []
    current_ids: dict[int, str] = {}
    unresolved: list[dict[str, Any]] = []

    for index, opinion in enumerate(output.opinions):
        if opinion.duplicate is True:
            continue
        accepted = (
            opinion.semantically_new is True
            or opinion.contradiction_or_revision is True
        )
        if not accepted:
            unresolved.append({"source_index": index, **opinion.model_dump(mode="json")})
            continue
        replaced_ids = _resolve_replacements(opinion, current_ids)
        snapshot_ids = [item for item in snapshot_ids if item not in replaced_ids]
        opinion_id = f"{node_id}:opinion_{index:04d}"
        opinion_store[opinion_id] = _opinion_record(
            opinion_id, opinion, node_id, index, replaced_ids,
        )
        snapshot_ids.append(opinion_id)
        created_ids.append(opinion_id)
        current_ids[index] = opinion_id
    return {
        "snapshot_opinion_ids": snapshot_ids,
        "created_opinion_ids": created_ids,
        "unresolved_opinions": unresolved,
    }


def _resolve_replacements(
    opinion: ImprovementOpinion,
    current_ids: dict[int, str],
) -> list[str]:
    """Convert this round's indices into stable IDs and merge them with the parent-snapshot targets."""
    targets = list(opinion.replaces_prior_opinion_ids)
    targets.extend(current_ids[index] for index in opinion.replaces_current_opinion_indices)
    return list(dict.fromkeys(targets))


def _write_state(path: Path, state: dict[str, Any]) -> None:
    """Save an inspectable intermediate state after each node finishes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_node(node_id: str, parent_id: str | None, snapshot_ids: list[str]) -> dict[str, Any]:
    """Create the fields shared by every node."""
    return {
        "node_id": node_id,
        "parent_id": parent_id,
        "snapshot_opinion_ids": list(snapshot_ids),
        "created_opinion_ids": [],
        "unresolved_opinions": [],
    }


def run_ace_analysis(
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
    """Analyze ACE nodes in order and persist the full trace state after each round."""
    artifacts = collect_ace_artifacts(experiment_dir)
    scores = collect_ace_scores(experiment_dir)
    if limit is not None:
        artifacts = artifacts[:limit]
    destination = Path(output_path).resolve()
    state: dict[str, Any] = {
        "schema_version": 6,
        "method": "ACE",
        "experiment_dir": str(Path(experiment_dir).resolve()),
        "model": model,
        "status": "running",
        "opinions": {},
        "nodes": [],
        "final_snapshot_opinion_ids": [],
    }

    snapshot_ids: list[str] = []
    parent_id: str | None = None
    for node_id, artifact in artifacts:
        node = _base_node(node_id, parent_id, snapshot_ids)
        node.update(scores[node_id])
        try:
            if artifact is None:
                node.update({
                    "status": "missing_artifact", "analysis": None,
                    "summary": None, "telemetry": {},
                })
            elif not artifact:
                empty = AnalysisOutput(method="ACE", round=node_id, opinions=[])
                node.update({
                    "status": "unchanged",
                    "analysis": empty.model_dump(mode="json"),
                    "summary": summarize_analysis(empty),
                    "telemetry": {},
                })
            else:
                prior = [state["opinions"][item] for item in snapshot_ids]
                telemetry: dict[str, Any] = {}
                result = analyze_artifact(
                    "ACE", node_id, artifact, prior, model=model, api_key=api_key,
                    base_url=base_url, timeout=timeout, client=client, telemetry=telemetry,
                )
                refined = refine_snapshot(state["opinions"], snapshot_ids, result, node_id)
                node.update(refined)
                node.update({
                    "status": "completed",
                    "analysis": result.model_dump(mode="json"),
                    "summary": summarize_analysis(result),
                    "telemetry": telemetry,
                })
                snapshot_ids = node["snapshot_opinion_ids"]
        except Exception as exc:
            node.update({"status": "failed", "error": str(exc)})
            state["nodes"].append(node)
            state["status"] = "failed"
            _write_state(destination, state)
            raise

        state["nodes"].append(node)
        state["final_snapshot_opinion_ids"] = list(snapshot_ids)
        parent_id = node_id
        _write_state(destination, state)
        if on_node:
            on_node(node)
    missing = sum(node["status"] == "missing_artifact" for node in state["nodes"])
    state["missing_artifact_count"] = missing
    state["status"] = "incomplete" if missing else "completed"
    _write_state(destination, state)
    return state
