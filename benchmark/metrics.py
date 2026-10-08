"""Multi-dimensional RSI metrics computed from normalized run records."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean, pvariance
from typing import Any


def _score(record: dict[str, Any]) -> float:
    value = record.get("relative_score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _total(record: dict[str, Any], field: str) -> float:
    value = record.get(field)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def calculate_metrics(
    training: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    evolution_window: int,
) -> dict[str, Any]:
    evolution_records = [record for record in training if not record.get("replay")]
    replay_records = [record for record in training if record.get("replay")]
    scores = [_score(record) for record in evolution_records]
    window = min(evolution_window, len(scores))
    first = fmean(scores[:window]) if window else None
    last = fmean(scores[-window:]) if window else None

    by_run_id = {record["run_id"]: record for record in evolution_records}
    retention_changes: list[float] = []
    normalized_changes: list[float] = []
    pairs: list[dict[str, Any]] = []
    for replay in replay_records:
        original = by_run_id.get(replay.get("replay_of"))
        gap = replay.get("replay_gap")
        if original is None or not isinstance(gap, int) or gap <= 0:
            continue
        change = _score(replay) - _score(original)
        retention_changes.append(change)
        normalized_changes.append(change / gap)
        pairs.append(
            {
                "original_run_id": original["run_id"],
                "replay_run_id": replay["run_id"],
                "gap": gap,
                "score_change": change,
                "change_per_round": change / gap,
            }
        )

    groups: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for record in holdout:
        groups[(record["game"], int(record["difficulty"]), int(record["group_index"]))].append(
            _score(record)
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

    total_tokens = sum(_total(record, "total_tokens") for record in training)
    total_seconds = sum(_total(record, "total_seconds") for record in training)
    declines = sum(current < previous for previous, current in zip(scores, scores[1:]))
    large_declines = sum(
        previous > 0 and current <= 0.8 * previous
        for previous, current in zip(scores, scores[1:])
    )
    return {
        "evolution_score": {
            "window": evolution_window,
            "first_window_average": first,
            "last_window_average": last,
        },
        "capability_retention": {
            "replay_pair_count": len(pairs),
            "average_score_change": fmean(retention_changes) if retention_changes else None,
            "average_change_per_round": fmean(normalized_changes) if normalized_changes else None,
            "pairs": pairs,
        },
        "generalization": {
            "overall_average_relative_score": (
                fmean([_score(record) for record in holdout]) if holdout else None
            ),
            "groups": generalization,
        },
        "evolution_cost": {
            "total_tokens": int(total_tokens),
            "total_seconds": total_seconds,
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
    score = metrics["evolution_score"]
    retention = metrics["capability_retention"]
    generalization = metrics["generalization"]
    cost = metrics["evolution_cost"]
    stability = metrics["stability"]

    def display(value: Any) -> str:
        return "N/A" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)

    lines = [
        "RSI metrics",
        f"  evolution score: first={display(score['first_window_average'])}, last={display(score['last_window_average'])}",
        f"  capability retention: mean_delta={display(retention['average_score_change'])}, delta/round={display(retention['average_change_per_round'])}",
        f"  generalization: mean={display(generalization['overall_average_relative_score'])}",
        f"  evolution cost: total tokens={display(cost['total_tokens'])}, total time (seconds)={display(cost['total_seconds'])}",
        f"  stability: variance={display(stability['score_variance'])}, decline_fraction={display(stability['declining_round_fraction'])}, drop_20pct_round_count={display(stability['drop_20pct_round_count'])}",
    ]
    for group in generalization["groups"]:
        lines.append(
            f"    holdout {group['game']} d{group['difficulty']} n={group['count']}: "
            f"{display(group['average_relative_score'])}"
        )
    return "\n".join(lines)
