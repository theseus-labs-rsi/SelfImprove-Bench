"""Multi-dimensional RSI metrics computed from normalized run records."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean, pvariance
from typing import Any


def _score(record: dict[str, Any]) -> float | None:
    if record.get("status") != "ok":
        return None
    value = record.get("relative_score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _total(record: dict[str, Any], field: str) -> float | None:
    value = record.get(field)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def calculate_metrics(
    training: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    evolution_window: int,
    expected_training_runs: int | None = None,
) -> dict[str, Any]:
    evolution_records = [record for record in training if not record.get("replay")]
    replay_records = [record for record in training if record.get("replay")]
    scored = [_score(record) for record in evolution_records]
    scores = [score for score in scored if score is not None]
    # Window position is scheduled-task order; never move a failed task out of
    # the window and accidentally label a later task the 'first' result.
    first_window = scored[:evolution_window]
    last_window = scored[-evolution_window:]
    first = (fmean(first_window) if len(first_window) == evolution_window
             and all(score is not None for score in first_window) else None)
    last = (fmean(last_window) if len(last_window) == evolution_window
            and (expected_training_runs is None or len(training) == expected_training_runs)
            and all(score is not None for score in last_window) else None)
    absolute_gain = (last - first) if first is not None and last is not None else None
    relative_gain = absolute_gain / first if absolute_gain is not None and first not in (None, 0) else None

    by_run_id = {record["run_id"]: record for record in evolution_records}
    retention_changes: list[float] = []
    normalized_changes: list[float] = []
    pairs: list[dict[str, Any]] = []
    for replay in replay_records:
        original = by_run_id.get(replay.get("replay_of"))
        gap = replay.get("replay_gap")
        if original is None or not isinstance(gap, int) or gap <= 0:
            continue
        replay_score, original_score = _score(replay), _score(original)
        if replay_score is None or original_score is None:
            continue
        change = replay_score - original_score
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
        score = _score(record)
        if score is not None:
            groups[(record["game"], int(record["difficulty"]), int(record["group_index"]))].append(score)
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

    measured_tokens = [_total(record, "total_tokens") for record in training]
    total_tokens = (sum(measured_tokens) if measured_tokens
                    and all(value is not None for value in measured_tokens) else None)
    measured_seconds = [_total(record, "total_seconds") for record in training]
    total_seconds = (sum(measured_seconds) if measured_seconds
                     and all(value is not None for value in measured_seconds) else None)
    valid_holdout_scores = [score for record in holdout if (score := _score(record)) is not None]
    transitions = [(a, b) for a, b in zip(scored, scored[1:]) if a is not None and b is not None]
    declines = sum(current < previous for previous, current in transitions)
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
        },
        "generalization": {
            "overall_average_relative_score": (
                fmean(valid_holdout_scores) if valid_holdout_scores else None
            ),
            "groups": generalization,
        },
        "evolution_cost": {
            "total_tokens": int(total_tokens) if total_tokens is not None else None,
            "total_seconds": total_seconds,
            "tokens_per_absolute_gain": (
                total_tokens / absolute_gain if total_tokens is not None and absolute_gain not in (None, 0) else None
            ),
            "seconds_per_absolute_gain": (
                total_seconds / absolute_gain if total_seconds is not None and absolute_gain not in (None, 0) else None
            ),
        },
        "stability": {
            "score_variance": pvariance(scores) if scores else None,
            "declining_round_fraction": declines / len(transitions) if transitions else None,
            "declining_rounds": declines,
            "transition_count": len(transitions),
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
        f"  stability: variance={display(stability['score_variance'])}, decline_fraction={display(stability['declining_round_fraction'])}",
    ]
    for group in generalization["groups"]:
        lines.append(
            f"    holdout {group['game']} d{group['difficulty']} n={group['count']}: "
            f"{display(group['average_relative_score'])}"
        )
    return "\n".join(lines)
