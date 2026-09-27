"""按 AHE iteration 顺序分析候选 workspace，并保存逐结点精炼快照。"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .ace import refine_snapshot, summarize_analysis
from .llm import analyze_artifact
from .reader import collect_ahe_artifacts


def _write_state(path: Path, state: dict[str, Any]) -> None:
    """在每个 iteration 结束后保存可检查的中间状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_node(source: dict[str, Any], snapshot_ids: list[str]) -> dict[str, Any]:
    """创建 AHE 结点共有字段。"""
    return {
        "node_id": source["node_id"],
        "parent_id": source["parent_id"],
        "iteration": source["iteration"],
        "variant": source["variant"],
        "artifact_dir": source["artifact_dir"],
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
    """分析一轮 AHE 概述与 workspace 增量差异。"""
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
    """顺序分析 AHE 单 variant iteration 链。"""
    sources = collect_ahe_artifacts(experiment_dir)
    if limit is not None:
        sources = sources[:limit]
    destination = Path(output_path).resolve()
    state: dict[str, Any] = {
        "schema_version": 5,
        "method": "AHE",
        "experiment_dir": str(Path(experiment_dir).resolve()),
        "run_id": sources[0]["run_id"] if sources else None,
        "model": model,
        "status": "running",
        "opinions": {},
        "nodes": [],
        "final_snapshot_opinion_ids": [],
    }
    snapshot_ids: list[str] = []
    options = {
        "model": model, "api_key": api_key, "base_url": base_url,
        "timeout": timeout, "client": client,
    }

    for source in sources:
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
        snapshot_ids = node["snapshot_opinion_ids"]
        state["final_snapshot_opinion_ids"] = list(snapshot_ids)
        _write_state(destination, state)
        if on_node:
            on_node(node)
    state["status"] = "completed"
    _write_state(destination, state)
    return state
