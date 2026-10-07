from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .candidate import CandidateSnapshot, initialize_agent, with_lineage


REGISTRY_FILENAME = "candidate_registry.json"


def _candidate_path(run_dir: Path, stored_path: str | Path) -> Path:
    path = Path(stored_path)
    resolved = path.resolve() if path.is_absolute() else (run_dir / path).resolve()
    if not resolved.is_relative_to(run_dir):
        raise ValueError(f"Candidate path leaves the run directory: {stored_path}")
    return resolved


def _relative_path(run_dir: Path, path: str | Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(run_dir):
        raise ValueError(f"Candidate path leaves the run directory: {path}")
    return str(resolved.relative_to(run_dir))


def _selection_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str | int
    parent_id: str | int | None
    root_commit: str
    lineage_patches: tuple[Path, ...]
    train_result_dir: Path
    test_result_dir: Path
    selection_score: float | None
    valid_parent: bool
    source_round: str | int

    def snapshot(self, initial: CandidateSnapshot) -> CandidateSnapshot:
        if self.root_commit != initial.root_commit:
            raise ValueError(
                f"Candidate {self.candidate_id} has a different root commit."
            )
        return with_lineage(
            initial,
            candidate_id=self.candidate_id,
            parent_id=self.parent_id,
            lineage_patches=self.lineage_patches,
        )


class CandidateRegistry:
    """Persistent index connecting candidate IDs to code and evaluation state."""

    def __init__(
        self,
        initial: CandidateSnapshot,
        records: Iterable[CandidateRecord] = (),
    ) -> None:
        self.initial = initial
        self.run_dir = initial.run_dir.resolve()
        self._records = {record.candidate_id: record for record in records}

    @property
    def path(self) -> Path:
        return self.run_dir / REGISTRY_FILENAME

    def get(self, candidate_id: str | int) -> CandidateRecord:
        try:
            return self._records[candidate_id]
        except KeyError as exc:
            raise KeyError(f"Unknown candidate ID: {candidate_id}") from exc

    def snapshot(self, candidate_id: str | int) -> CandidateSnapshot:
        return self.get(candidate_id).snapshot(self.initial)

    def scores(self) -> dict[str | int, float]:
        return {
            candidate_id: record.selection_score
            for candidate_id, record in self._records.items()
            if record.valid_parent and record.selection_score is not None
        }

    def child_counts(self) -> dict[str | int, int]:
        counts = {candidate_id: 0 for candidate_id in self.scores()}
        for record in self._records.values():
            if record.parent_id in counts:
                counts[record.parent_id] += 1
        return counts

    def register(
        self,
        snapshot: CandidateSnapshot,
        *,
        train_result_dir: str | Path,
        test_result_dir: str | Path,
        summary: Mapping[str, Any],
        source_round: str | int,
    ) -> CandidateRecord:
        score = _selection_score(summary.get("train_selection_score"))
        record = CandidateRecord(
            candidate_id=snapshot.candidate_id,
            parent_id=snapshot.parent_id,
            root_commit=snapshot.root_commit,
            lineage_patches=snapshot.lineage_patches,
            train_result_dir=Path(train_result_dir).resolve(),
            test_result_dir=Path(test_result_dir).resolve(),
            selection_score=score,
            valid_parent=summary.get("train_valid") is True and score is not None,
            source_round=source_round,
        )
        self._records[record.candidate_id] = record
        self.save()
        return record

    def save(self) -> None:
        payload = {
            "schema_version": 1,
            "root_commit": self.initial.root_commit,
            "candidates": [self._serialize(record) for record in self._records.values()],
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _serialize(self, record: CandidateRecord) -> dict[str, Any]:
        return {
            "candidate_id": record.candidate_id,
            "parent_id": record.parent_id,
            "root_commit": record.root_commit,
            "lineage_patches": [
                _relative_path(self.run_dir, path) for path in record.lineage_patches
            ],
            "train_result_dir": _relative_path(self.run_dir, record.train_result_dir),
            "test_result_dir": _relative_path(self.run_dir, record.test_result_dir),
            "selection_score": record.selection_score,
            "valid_parent": record.valid_parent,
            "source_round": record.source_round,
        }

    @classmethod
    def load(cls, initial: CandidateSnapshot) -> "CandidateRegistry":
        run_dir = initial.run_dir.resolve()
        payload = json.loads((run_dir / REGISTRY_FILENAME).read_text(encoding="utf-8"))
        records = [cls._deserialize(run_dir, item) for item in payload["candidates"]]
        registry = cls(initial, records)
        if payload.get("root_commit") != initial.root_commit:
            raise ValueError("Candidate registry root commit does not match the run.")
        return registry

    @staticmethod
    def _deserialize(run_dir: Path, item: Mapping[str, Any]) -> CandidateRecord:
        return CandidateRecord(
            candidate_id=item["candidate_id"],
            parent_id=item.get("parent_id"),
            root_commit=str(item["root_commit"]),
            lineage_patches=tuple(
                _candidate_path(run_dir, path)
                for path in item.get("lineage_patches", [])
            ),
            train_result_dir=_candidate_path(run_dir, item["train_result_dir"]),
            test_result_dir=_candidate_path(run_dir, item["test_result_dir"]),
            selection_score=_selection_score(item.get("selection_score")),
            valid_parent=item.get("valid_parent") is True,
            source_round=item["source_round"],
        )

    @classmethod
    def rebuild(
        cls,
        initial: CandidateSnapshot,
        summaries: Iterable[Mapping[str, Any]],
    ) -> "CandidateRegistry":
        registry = cls(initial)
        fallback_lineage: list[Path] = []
        current_id: str | int = initial.candidate_id
        for summary in summaries:
            snapshot, fallback_lineage = _snapshot_from_summary(
                initial, summary, fallback_lineage, current_id
            )
            if snapshot is None:
                continue
            current_id = snapshot.candidate_id
            train_dir, test_dir = _round_result_dirs(
                initial.run_dir, summary["round"]
            )
            registry.register(
                snapshot,
                train_result_dir=train_dir,
                test_result_dir=test_dir,
                summary=summary,
                source_round=summary["round"],
            )
        return registry


def _snapshot_from_summary(
    initial: CandidateSnapshot,
    summary: Mapping[str, Any],
    fallback_lineage: list[Path],
    current_id: str | int,
) -> tuple[CandidateSnapshot | None, list[Path]]:
    round_id = summary.get("round")
    if round_id == "initial":
        return initial, fallback_lineage
    if summary.get("changed") is not True:
        return None, fallback_lineage
    patch = _candidate_path(initial.run_dir, summary["patch_path"])
    fallback_lineage = [*fallback_lineage, patch]
    stored_lineage = summary.get("lineage_patches")
    lineage = (
        [_candidate_path(initial.run_dir, path) for path in stored_lineage]
        if isinstance(stored_lineage, list)
        else fallback_lineage
    )
    snapshot = with_lineage(
        initial,
        candidate_id=summary.get("candidate_id", round_id),
        parent_id=summary.get("candidate_parent_id", summary.get("parent_id", current_id)),
        lineage_patches=lineage,
    )
    return snapshot, fallback_lineage


def _round_result_dirs(run_dir: Path, round_id: str | int) -> tuple[Path, Path]:
    round_dir = run_dir / "initial" if round_id == "initial" else run_dir / f"round_{int(round_id):03d}"
    return round_dir / "train", round_dir / "test"


def load_candidate_snapshot(
    run_dir: str | Path,
    candidate_id: str | int,
) -> CandidateSnapshot:
    initial = initialize_agent(run_dir, reuse_existing=True)
    return CandidateRegistry.load(initial).snapshot(candidate_id)
