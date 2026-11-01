"""Committed results agree across uninterrupted, paused and retried execution."""

import json
from dataclasses import replace

import pytest

from benchmark import runner
from benchmark.config import config_from_mapping
from benchmark.contracts import EvaluationOutcome
from benchmark.tasks import GAMES


FIELDS = (
    "run_id", "game", "difficulty", "phase", "task_seed", "replay", "replay_of",
    "replay_gap", "status", "relative_score", "success", "post_refinement_score",
    "inference_tokens", "refinement_tokens", "candidate_validation_tokens", "total_tokens",
)


def committed_records(report):
    return [{key: row.get(key) for key in FIELDS}
            for row in report["training_runs"] + report["holdout_runs"]]


def deterministic_metrics(report):
    metrics = json.loads(json.dumps(report["metrics"]))
    # Wall-clock time legitimately varies; token accounting and every score
    # statistic must agree. Retry attempts are retained separately as artifacts.
    for field in ("total_seconds", "seconds_per_absolute_gain"):
        metrics["evolution_cost"].pop(field)
    return metrics


@pytest.mark.parametrize("seed", [17, 42])
@pytest.mark.parametrize("interruption", ["pause", "provider_failure"])
def test_random_schedule_replay_resume_matches_uninterrupted(
    tmp_path, monkeypatch, seed, interruption,
):
    config = config_from_mapping({
        "seed": seed, "random": True, "evolution_window": 2,
        "schedule": [
            {"game": "farm", "difficulty": 1, "count": 3, "replay": True},
            {"game": "farm", "difficulty": 2, "count": 3, "replay": True},
        ],
        "holdout": [{"game": "farm", "difficulty": 2, "count": 2}],
        "replay": {"every": 2, "lag": 1},
        "agent": {"provider": "rsi", "adapter": "examples.minimal_rsi:MinimalAdapter"},
        "output_dir": str(tmp_path / "uninterrupted"),
    })
    monkeypatch.setattr(GAMES["farm"], "generate", lambda difficulty, *, seed, task_id: {
        "difficulty": difficulty, "seed": seed, "task_id": task_id,
    })
    fail_next = False
    calls = 0

    def evaluate(game, task, **kwargs):
        nonlocal fail_next, calls
        calls += 1
        if fail_next and calls == 3:
            fail_next = False
            raise TimeoutError("synthetic provider timeout; no network")
        return EvaluationOutcome({
            "game": game, "difficulty": task["difficulty"], "status": "ok",
            "relative_score": (task["seed"] % 10) / 10, "success": False,
            "llm_metrics": {"tokens": {"total_tokens": 5 + task["seed"] % 17}},
        })

    monkeypatch.setattr(runner, "evaluate_task", evaluate)
    expected = runner.run_benchmark(config)
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    calls = 0
    fail_next = interruption == "provider_failure"
    partial = runner.run_benchmark(
        resumed_config, max_new_train_runs=2 if interruption == "pause" else None,
    )
    assert len(partial["training_runs"]) < len(expected["training_runs"])
    actual = runner.run_benchmark(resumed_config, resume=True)
    assert committed_records(actual) == committed_records(expected)
    assert deterministic_metrics(actual) == deterministic_metrics(expected)
    assert actual["metrics"]["capability_retention"]["replay_pair_count"] > 0
    assert actual["summary"]["checkpoint_chain_complete"]
    assert actual["summary"]["abandoned_attempts"] == int(interruption == "provider_failure")
    checkpoint = json.loads((tmp_path / "resumed/checkpoints/holdout_0002.json").read_text())
    assert checkpoint["state"]["adapter_state"]["rounds"] == 6
    # A second resume is idempotent and does not execute or refine again.
    calls_before = calls
    again = runner.run_benchmark(resumed_config, resume=True)
    assert calls == calls_before
    assert committed_records(again) == committed_records(actual)
