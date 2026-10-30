"""Multi-dimensional RSI metrics computed from normalized run records."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import fmean, pvariance
from typing import Any


def record_score(record: dict[str, Any]) -> float:
    """Keep every attempted case in the denominator, with failures scored zero."""
    if record.get("status") not in {None, "ok"}:
        return 0.0
    value = record.get("relative_score")
    return (
        float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
        and isfinite(value) else 0.0
    )


def _cost_total(records: list[dict[str, Any]], field: str) -> float | None:
    values = [record.get(field) for record in records]
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and isfinite(value) and value >= 0 for value in values
    ):
        return None
    return sum(values)


def execution_score(record: dict[str, Any]) -> float | None:
    """Read the pre-refinement execution score, counting explicit failures as zero."""
    if record.get("status") in {"error", "skipped"}:
        return 0.0
    value = record.get("relative_score")
    return (
        float(value) if record.get("status") == "ok"
        and isinstance(value, (int, float)) and not isinstance(value, bool)
        and isfinite(value) else None
    )


def post_refinement_score(record: dict[str, Any]) -> float | None:
    """Recover the score of the harness actually retained after same-task selection."""
    if "post_refinement_score" in record:
        value = record["post_refinement_score"]
        return (
            float(value) if isinstance(value, (int, float))
            and not isinstance(value, bool) and isfinite(value) else None
        )
    selection = record.get("selection")
    if selection is None:
        return record_score(record) if "refinement" in record and record["refinement"] is None else None
    if not isinstance(selection, dict):
        return None
    status = selection.get("status")
    if status in {"unchanged", "skipped"}:
        return record_score(record)
    if status not in {"committed", "completed"}:
        return None
    candidate_id = selection.get("committed_candidate_id")
    stages = selection.get("stages")
    if not candidate_id or not isinstance(stages, list) or not stages:
        return None
    final_stage = stages[-1]
    if not isinstance(final_stage, dict) or final_stage.get("committed_candidate_id") != candidate_id:
        return None
    evaluations = final_stage.get("evaluations")
    if not isinstance(evaluations, list):
        return None
    matches = [item for item in evaluations if isinstance(item, dict) and item.get("candidate_id") == candidate_id]
    if len(matches) != 1 or matches[0].get("status") != "ok":
        return None
    value = matches[0].get("relative_score")
    return (
        float(value) if isinstance(value, (int, float))
        and not isinstance(value, bool) and isfinite(value) else None
    )


def calculate_metrics(
    training: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    evolution_window: int,
) -> dict[str, Any]:
    evolution_records = [record for record in training if not record.get("replay")]
    replay_records = [record for record in training if record.get("replay")]
    scores = [record_score(record) for record in evolution_records]
    window = min(evolution_window, len(scores))
    first = fmean(scores[:window]) if window else None
    last = fmean(scores[-window:]) if window else None
    absolute_gain = (last - first) if first is not None and last is not None else None
    relative_gain = absolute_gain / first if absolute_gain is not None and first not in (None, 0) else None

    by_run_id = {record["run_id"]: record for record in evolution_records}
    retention_changes: list[float] = []
    normalized_changes: list[float] = []
    pairs: list[dict[str, Any]] = []
    unavailable_pairs: list[dict[str, Any]] = []
    for replay in replay_records:
        original = by_run_id.get(replay.get("replay_of"))
        gap = replay.get("replay_gap")
        if original is None or not isinstance(gap, int) or gap <= 0:
            continue
        baseline = execution_score(original)
        replay_score = execution_score(replay)
        if baseline is None or replay_score is None:
            unavailable_pairs.append({
                "original_run_id": original["run_id"],
                "replay_run_id": replay["run_id"],
                "reason": "execution_score_unavailable",
            })
            continue
        change = replay_score - baseline
        retention_changes.append(change)
        normalized_changes.append(change / gap)
        pairs.append(
            {
                "original_run_id": original["run_id"],
                "replay_run_id": replay["run_id"],
                "gap": gap,
                "baseline_score": baseline,
                "replay_score": replay_score,
                "score_change": change,
                "change_per_round": change / gap,
            }
        )

    groups: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for record in holdout:
        groups[(record["game"], int(record["difficulty"]), int(record["group_index"]))].append(
            record_score(record)
        )
    generalization = [
        {
            "game": game,
            "difficulty": difficulty,
            "group_index": group_index,
            "count": len(values),
            "average_relative_score": fmean(values),
        }
        for (game, difficulty, group_index), values in sorted(groups.items(), key=lambda item: item[0][2])
    ]

    total_tokens = _cost_total(training, "total_tokens")
    total_seconds = _cost_total(training, "total_seconds")
    declines = sum(current < previous for previous, current in zip(scores, scores[1:]))
    large_declines = sum(
        previous > 0 and current <= 0.8 * previous
        for previous, current in zip(scores, scores[1:])
    )
    return {
        "evolution_gain": {
            "window": evolution_window,
            "first_window_average": first,
            "last_window_average": last,
            "absolute_gain": absolute_gain,
            "relative_gain": relative_gain,
        },
        "capability_retention": {
            "replay_pair_count": len(pairs),
            "average_score_change": fmean(retention_changes) if retention_changes else None,
            "average_change_per_round": fmean(normalized_changes) if normalized_changes else None,
            "pairs": pairs,
            "baseline": "pre_refinement_execution_score",
            "unavailable_pair_count": len(unavailable_pairs),
            "unavailable_pairs": unavailable_pairs,
        },
        "generalization": {
            "overall_average_relative_score": (
                fmean([record_score(record) for record in holdout]) if holdout else None
            ),
            "groups": generalization,
        },
        "evolution_cost": {
            "total_tokens": int(total_tokens) if total_tokens is not None else None,
            "total_seconds": total_seconds,
            "tokens_per_absolute_gain": (
                total_tokens / absolute_gain
                if total_tokens is not None and absolute_gain not in (None, 0) else None
            ),
            "seconds_per_absolute_gain": (
                total_seconds / absolute_gain
                if total_seconds is not None and absolute_gain not in (None, 0) else None
            ),
        },
        "stability": {
            "score_variance": pvariance(scores) if scores else None,
            "declining_round_fraction": declines / (len(scores) - 1) if len(scores) > 1 else 0.0,
            "declining_rounds": declines,
            "drop_20pct_round_count": large_declines,
            "transition_count": max(0, len(scores) - 1),
        },
    }


def format_metrics(metrics: dict[str, Any]) -> str:
    gain = metrics["evolution_gain"]
    retention = metrics["capability_retention"]
    generalization = metrics["generalization"]
    cost = metrics["evolution_cost"]
    stability = metrics["stability"]

    def display(value: Any) -> str:
        return "N/A" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)

    lines = [
        "RSI metrics",
        f"  evolution gain: absolute={display(gain['absolute_gain'])}, relative={display(gain['relative_gain'])}",
        f"  capability retention: mean_delta={display(retention['average_score_change'])}, delta/round={display(retention['average_change_per_round'])}",
        f"  generalization: mean={display(generalization['overall_average_relative_score'])}",
        f"  evolution cost: tokens/gain={display(cost['tokens_per_absolute_gain'])}, seconds/gain={display(cost['seconds_per_absolute_gain'])}",
        f"  stability: variance={display(stability['score_variance'])}, decline_fraction={display(stability['declining_round_fraction'])}, drop_20pct_round_count={display(stability['drop_20pct_round_count'])}",
    ]
    for group in generalization["groups"]:
        lines.append(
            f"    holdout {group['game']} d{group['difficulty']} n={group['count']}: "
            f"{display(group['average_relative_score'])}"
        )
    return "\n".join(lines)
