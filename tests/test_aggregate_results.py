"""Aggregation must use the runner's all-attempts score denominator."""

import json

import pytest

from benchmark.aggregate_results import aggregate, parse_unified
from benchmark.metrics import calculate_metrics


def write_result(root, training, holdout):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "result.json"
    path.write_text(json.dumps({
        "config": {
            "agent": {"provider": "offline-test"},
            "schedule": [{"game": "farm", "difficulty": 3}],
        },
        "training_runs": training,
        "holdout_runs": holdout,
    }))
    return path


def record(run_id, status="ok", score=1.0, **extra):
    return {
        "run_id": run_id, "status": status, "relative_score": score,
        "game": "farm", "difficulty": 3, "group_index": 0,
        **extra,
    }


@pytest.mark.parametrize("status,score", [
    ("error", None), ("error", 999), ("skipped", None), ("skipped", 999),
    ("ok", None), ("ok", True), ("ok", float("nan")), ("ok", float("inf")),
])
def test_failed_or_invalid_cases_stay_in_denominator(tmp_path, status, score):
    records = [record("success"), record("failure", status, score)]
    row = parse_unified(write_result(tmp_path, records, records))[0]
    metrics = calculate_metrics(records, records, evolution_window=1)
    assert row["observed_training_count"] == 2
    assert row["observed_holdout_count"] == 2
    assert row["holdout_same_domain_same_difficulty"] == 0.5
    assert row["holdout_same_domain_same_difficulty"] == (
        metrics["generalization"]["overall_average_relative_score"]
    )
    assert row["last_absolute_score"] == 0.0
    assert row["last_absolute_score"] == metrics["evolution_gain"]["last_window_average"]


def test_all_failed_repeat_is_zero_not_missing(tmp_path):
    training = [record("train")]
    good = write_result(tmp_path / "good", training, [record("success")])
    bad = write_result(tmp_path / "bad", training, [record("failure", "error", None)])
    rows, groups = aggregate([str(good), str(bad)])
    assert rows[1]["observed_holdout_count"] == 1
    metric = groups[0]["holdout_same_domain_same_difficulty"]
    assert metric == {"mean": 0.5, "non_null_n": 2}


def test_holdout_errors_count_in_each_generalization_bucket(tmp_path):
    holdout = [
        record("same-ok"), record("same-error", "error", None),
        record("hard-ok", difficulty=5),
        record("hard-error", "error", None, difficulty=5),
        record("cross-ok", game="combat"),
        record("cross-error", "error", None, game="combat"),
    ]
    row = parse_unified(write_result(tmp_path, [record("train")], holdout))[0]
    assert row["observed_holdout_count"] == 6
    for category in ("same_domain_same_difficulty", "same_domain_higher_difficulty", "cross_domain"):
        assert row[f"holdout_{category}"] == 0.5


def test_failed_replay_uses_zero_not_retained_numeric_value(tmp_path):
    training = [
        record("original"),
        record("replay", "error", 999, replay=True, replay_of="original", replay_gap=2),
    ]
    row = parse_unified(write_result(tmp_path, training, []))[0]
    assert row["observed_training_count"] == 1
    assert row["observed_replay_pair_count"] == 1
    assert row["replay_score_difference"] == -1.0
    assert row["replay_difference_per_gap"] == -0.5


def test_unexecuted_cases_are_not_invented_as_zeros(tmp_path):
    row = parse_unified(write_result(tmp_path, [], []))[0]
    assert row["observed_training_count"] == 0
    assert row["observed_holdout_count"] == 0
    assert row["first_absolute_score"] is None
    assert row["holdout_same_domain_same_difficulty"] is None
