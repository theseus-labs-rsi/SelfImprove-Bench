"""Persist benchmark-owned, same-task evidence for the next AHE iteration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.resume_state import atomic_json


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "evolution_history.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def save_history(run_dir: Path, records: list[dict[str, Any]]) -> None:
    atomic_json(run_dir / "evolution_history.json", records)
    lines = [
        "# Evolution History",
        "",
        "Each round evaluates one training task. Only parent/candidate scores within "
        "that round are paired; scores across different tasks are not a fixed-set trend.",
        "The active lineage advances valid candidates, including score regressions.",
        "",
    ]
    for record in records:
        lines.extend([
            f"## {record['refinement_id']}",
            "",
            f"- Task: `{record['task_id']}`",
            f"- Parent score: {record['parent_score']}",
            f"- Status: {record.get('status', 'awaiting candidate evaluation')}",
        ])
        if "candidate_scores" in record:
            lines.extend([
                f"- Candidate scores: {json.dumps(record['candidate_scores'])}",
                f"- Committed: {record.get('committed_candidate_id')}",
                f"- Same-task score delta: {record.get('score_delta')}",
                f"- Change evidence: `{record['change_evaluation']}`",
            ])
        lines.append("")
    (run_dir / "evolution_history.md").write_text("\n".join(lines), encoding="utf-8")
