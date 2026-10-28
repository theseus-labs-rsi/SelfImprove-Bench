"""Offline regressions for failed refinement cost and unknown usage."""

import itertools
import json
from types import SimpleNamespace

import pytest

from benchmark import runner
from benchmark.contracts import EvaluationOutcome
from benchmark.metrics import calculate_metrics
from benchmark.tasks import PlannedTask


def run_task(tmp_path, monkeypatch, refine, *, previous=None, ordinal=1):
    session = SimpleNamespace(refinement_telemetry=previous, pending_batch=None)
    controller = SimpleNamespace(
        config=SimpleNamespace(agent=SimpleNamespace(provider="offline-test")),
        provider="offline-test", session=session, active_agent=None,
        begin=lambda task: None, request=lambda *args, **kwargs: None,
        refine=lambda *args, **kwargs: refine(session),
    )
    task = PlannedTask("synthetic-cost-test", "train", "farm", 1, {}, 1, 0)
    monkeypatch.setattr(runner, "_evaluate", lambda *args, **kwargs: EvaluationOutcome({
        "status": "ok", "game": "farm", "relative_score": 0.2, "success": False,
        "llm_metrics": {"tokens": {"total_tokens": 11}},
    }))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    ticks = itertools.count(100, 2)
    record = runner._run_one(task, controller=controller, ordinal=ordinal, output_dir=tmp_path)
    saved = json.loads((tmp_path / f"runs/train/{ordinal:04d}_{task.run_id}.json").read_text())
    assert saved == record
    return record


@pytest.mark.parametrize("tokens", [None, 17, 0])
def test_failed_refinement_preserves_current_telemetry(tmp_path, monkeypatch, tokens):
    def fail(session):
        session.refinement_telemetry = {
            "status": "error", "task_index": 1, "duration_seconds": 88.297587,
            "tokens": {"total_tokens": tokens},
            "error": "synthetic protocol failure",
        }
        raise RuntimeError("synthetic protocol failure")

    record = run_task(tmp_path, monkeypatch, fail)
    assert record["status"] == "error"
    assert record["parent_status"] == "ok"
    assert record["selection"]["status"] == "error"
    assert record["refinement"]["error"] == "synthetic protocol failure"
    assert record["refinement_seconds"] == 88.297587
    assert record["total_seconds"] == pytest.approx(90.297587)
    assert record["inference_tokens"] == 11
    assert record["refinement_tokens"] == tokens
    assert record["total_tokens"] == (None if tokens is None else 11 + tokens)


@pytest.mark.parametrize("previous", [None, {
    "status": "ok", "task_index": 1, "duration_seconds": 12345,
    "tokens": {"total_tokens": 999},
}])
def test_failure_without_new_telemetry_does_not_reuse_previous_costs(
    tmp_path, monkeypatch, previous,
):
    def fail(session):
        raise RuntimeError("synthetic failure before telemetry")

    record = run_task(tmp_path, monkeypatch, fail, previous=previous, ordinal=2)
    assert record["refinement"]["status"] == "error"
    assert record["refinement"]["task_index"] == 2
    assert record["refinement_seconds"] == 2
    assert record["total_seconds"] == 4
    assert record["refinement_tokens"] is None
    assert record["total_tokens"] is None


def test_selection_failure_does_not_replace_completed_refinement_cost(tmp_path, monkeypatch):
    def refine(session):
        session.refinement_telemetry = {
            "status": "ok", "task_index": 1, "duration_seconds": 5,
            "tokens": {"total_tokens": 17},
        }
        session.pending_batch = SimpleNamespace(candidates=[SimpleNamespace(candidate_id="test")])

    def fail_selection(*args, **kwargs):
        raise RuntimeError("synthetic selection failure")

    monkeypatch.setattr(runner, "_evaluate_candidates", fail_selection)
    record = run_task(tmp_path, monkeypatch, refine)
    assert record["selection"]["status"] == "error"
    assert record["refinement"]["status"] == "ok"
    assert record["refinement_seconds"] == 5
    assert record["refinement_tokens"] == 17


@pytest.mark.parametrize("status,tokens", [("ok", 17), ("skipped", 0)])
def test_success_and_explicit_zero_usage_stay_recorded(tmp_path, monkeypatch, status, tokens):
    def refine(session):
        session.refinement_telemetry = {
            "status": status, "task_index": 1, "duration_seconds": 5,
            "tokens": {"total_tokens": tokens},
        }

    record = run_task(tmp_path, monkeypatch, refine)
    assert record["status"] == "ok"
    assert record["refinement_seconds"] == 5
    assert record["refinement_tokens"] == tokens
    assert record["total_tokens"] == 11 + tokens


@pytest.mark.parametrize("tokens,expected", [(None, None), (17, 28), (0, 11)])
def test_cost_metrics_propagate_unknown_usage(tokens, expected):
    records = [
        {"run_id": "known", "relative_score": 0.2, "total_tokens": 11, "total_seconds": 2},
        {"run_id": "failed", "relative_score": 0.7, "total_tokens": tokens, "total_seconds": 88},
    ]
    cost = calculate_metrics(records, [], evolution_window=1)["evolution_cost"]
    assert cost["total_tokens"] == expected
    assert cost["tokens_per_absolute_gain"] == (
        None if expected is None else pytest.approx(expected / 0.5)
    )
    assert cost["total_seconds"] == 90
    assert cost["seconds_per_absolute_gain"] == pytest.approx(180)


def test_missing_cost_is_not_a_known_zero():
    cost = calculate_metrics([{"run_id": "legacy", "relative_score": 0.2}], [], evolution_window=1)[
        "evolution_cost"
    ]
    assert cost["total_tokens"] is None
    assert cost["total_seconds"] is None
