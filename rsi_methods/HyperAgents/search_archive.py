"""Persistent search state for the official HyperAgents archive policy."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
from typing import Any

from .archive_utils import select_parent_from_candidates
from .candidate import CandidateSnapshot


class SearchArchive:
    def __init__(self, *, seed: int | None = None) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.rng = random.Random(seed)

    def add(
        self, snapshot: CandidateSnapshot, *, score: float | None,
        valid_parent: bool, artifact_dir: Path | None = None,
    ) -> None:
        key = str(snapshot.candidate_id)
        if key in self.nodes:
            raise ValueError(f"Duplicate HyperAgents archive candidate: {key}")
        if Path(key).name != key or key in {".", ".."}:
            raise ValueError("Unsafe HyperAgents candidate identifier")
        if not isinstance(valid_parent, bool):
            raise ValueError("HyperAgents valid_parent must be bool")
        if score is not None and (
            isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("HyperAgents archive score must be finite or null")
        self.nodes[key] = {
            "snapshot": snapshot, "score": score, "valid_parent": valid_parent,
            "artifact_dir": artifact_dir,
        }

    def choose(self, method: str) -> CandidateSnapshot:
        candidates = {
            key: node["score"] for key, node in self.nodes.items()
            if node["valid_parent"] and node["score"] is not None
        }
        if not candidates:
            # Upstream falls back to the initial node before any valid score exists.
            candidates = {next(iter(self.nodes)): 0.0}
        counts = {key: 0 for key in candidates}
        for node in self.nodes.values():
            parent = node["snapshot"].parent_id
            if parent is not None and str(parent) in counts:
                counts[str(parent)] += 1
        chosen = select_parent_from_candidates(
            candidates, method, counts, rng=self.rng,
        )
        return self.nodes[chosen]["snapshot"]

    def to_dict(self) -> dict[str, Any]:
        from .HyperAgentsAdaptor import _snapshot_to_dict

        nodes = []
        for node in self.nodes.values():
            snapshot = node["snapshot"]
            nodes.append({
                "snapshot": _snapshot_to_dict(snapshot),
                "patch_digests": [
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in snapshot.lineage_patches
                ],
                "score": node["score"], "valid_parent": node["valid_parent"],
                "artifact_dir": str(node["artifact_dir"]) if node["artifact_dir"] else None,
            })
        state = self.rng.getstate()
        return {
            "schema_version": 1, "nodes": nodes,
            "random_state": [state[0], list(state[1]), state[2]],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, root: Path) -> SearchArchive:
        from rsi_methods.checkpoint import _path

        if data.get("schema_version") != 1:
            raise RuntimeError("Unsupported HyperAgents archive checkpoint")
        archive = cls()
        for node in data["nodes"]:
            saved = node["snapshot"]
            patches = tuple(_path(p, root) for p in saved["lineage_patches"])
            if [hashlib.sha256(p.read_bytes()).hexdigest() for p in patches] != node["patch_digests"]:
                raise RuntimeError("HyperAgents archived lineage patches changed")
            snapshot = CandidateSnapshot(
                saved["candidate_id"], _path(saved["run_dir"], root, directory=True),
                _path(saved["workspace"], root, directory=True), saved["root_commit"],
                saved["parent_id"], patches,
            )
            archive.add(
                snapshot, score=node["score"], valid_parent=node["valid_parent"],
                artifact_dir=_path(node["artifact_dir"], root, directory=True)
                if node["artifact_dir"] else None,
            )
        if not archive.nodes:
            raise RuntimeError("Empty HyperAgents archive checkpoint")
        for node in archive.nodes.values():
            parent = node["snapshot"].parent_id
            if parent is not None and str(parent) not in archive.nodes:
                raise RuntimeError("HyperAgents archive parent is missing")
        state = data["random_state"]
        archive.rng.setstate((state[0], tuple(state[1]), state[2]))
        return archive

    def stage_history(self, destination: Path) -> None:
        """Expose only method-owned training artifacts, never the batch tree."""
        import json
        import shutil

        destination.mkdir(parents=True, exist_ok=True)
        summary = []
        for key, node in self.nodes.items():
            snapshot = node["snapshot"]
            summary.append({
                "candidate_id": key, "parent_id": snapshot.parent_id,
                "score": node["score"], "valid_parent": node["valid_parent"],
            })
            source = node["artifact_dir"]
            if source is None:
                continue
            target = destination / key
            # An allowlist prevents evaluation/test trees or credentials from leaking.
            for relative in (
                "evidence", "agent_output/model_patch.diff",
                "agent_output/meta_agent_history.json",
                "agent_output/meta_agent_chat_history.md", "evaluation/result.json",
            ):
                path = source / relative
                if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
                    raise ValueError(f"Symlink in HyperAgents training history: {path}")
                if path.is_dir():
                    if any(p.is_symlink() for p in path.rglob("*")):
                        raise ValueError(f"Symlink in HyperAgents training history: {path}")
                    shutil.copytree(path, target / relative)
                elif path.is_file():
                    (target / relative).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target / relative)
        (destination / "archive.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
        )
