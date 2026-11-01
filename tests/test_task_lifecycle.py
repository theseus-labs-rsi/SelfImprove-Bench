"""Failures at execution, persistence and cleanup must release the active task."""

from dataclasses import replace

import pytest

from benchmark import runner
from benchmark.config import AgentConfig, BenchmarkConfig
from benchmark.contracts import EvaluationOutcome, ExecutionStatus
from benchmark.rsi_session import make_feedback_result
from benchmark.tasks import PlannedTask


@pytest.fixture
def controller(tmp_path):
    config = BenchmarkConfig(
        seed=42, schedule=(), holdout=(),
        agent=AgentConfig(provider="rsi", adapter="examples.minimal_rsi:MinimalAdapter"),
    )
    instance = runner.AgentController(config, tmp_path, 3)
    yield instance
    instance.close()


def task(run_id="first"):
    return PlannedTask(run_id, "train", "farm", 1, {"task_id": run_id}, 42, 0)


def outcome(*args, **kwargs):
    return EvaluationOutcome({
        "status": "ok", "game": "farm", "relative_score": 0.4, "success": False,
        "llm_metrics": {"tokens": {"total_tokens": 10}},
    })


def assert_released(controller):
    assert controller.active_agent is None
    assert controller.session._agent is None
    assert controller.session._task_id is None
    assert controller.session.pending_batch is None


@pytest.mark.parametrize("boundary", ["evaluate_task", "_evaluate"])
def test_execution_exception_keeps_original_error_and_next_task_runs(
    tmp_path, monkeypatch, controller, boundary,
):
    def fail(*args, **kwargs):
        raise ValueError("invalid synthetic action")

    monkeypatch.setattr(runner, boundary, fail)
    first = runner._run_one(task(), controller=controller, ordinal=1, output_dir=tmp_path)
    assert first["error_type"] == "ValueError"
    assert first["error"] == "invalid synthetic action"
    assert first["selection"]["status"] == "completed"
    assert first["total_tokens"] is None
    assert_released(controller)
    monkeypatch.setattr(runner, boundary, outcome)
    second = runner._run_one(task("second"), controller=controller, ordinal=2, output_dir=tmp_path)
    assert second["status"] == "ok"
    assert controller.session.adaptor.rounds == 2
    assert_released(controller)


def test_begin_failure_and_record_write_failure_release_task(tmp_path, monkeypatch, controller):
    begin = controller.begin

    def partial_begin(value):
        begin(value)
        raise ValueError("failed after loading agent")

    monkeypatch.setattr(controller, "begin", partial_begin)
    record = runner._run_one(task(), controller=controller, ordinal=1, output_dir=tmp_path)
    assert record["error"] == "failed after loading agent"
    assert_released(controller)
    monkeypatch.setattr(controller, "begin", begin)
    monkeypatch.setattr(runner, "_evaluate", outcome)

    def fail_write(*args):
        raise OSError("disk full")

    monkeypatch.setattr(runner.resume_state, "atomic_json", fail_write)
    with pytest.raises(OSError, match="disk full"):
        runner._run_one(task("second"), controller=controller, ordinal=2, output_dir=tmp_path)
    assert_released(controller)


def test_refinement_setup_failure_releases_task_without_masking_error(controller):
    session = controller.session
    session.begin_task("first")
    # A previous partial artifact is a real failure before adaptor.refine runs.
    (session.artifact_dir / "task_0001_first").mkdir()
    result = make_feedback_result(
        task_id="first", task={}, trajectory=[], execution_status=ExecutionStatus.ERROR,
        success=False, score=None, feedback="failed", metrics={},
    )
    with pytest.raises(FileExistsError):
        session.refine(result, 1)
    assert session._agent is None and session._task_id is None
    assert session.adaptor.rounds == 0
    assert session.refinement_telemetry["tokens"]["total_tokens"] == 0
    session.begin_task("second")
    session.refine(replace(result, task_id="second"), 2)
    assert session.adaptor.rounds == 1


def test_failed_adapter_does_not_reuse_previous_round_usage(controller, monkeypatch):
    session = controller.session
    result = make_feedback_result(
        task_id="first", task={}, trajectory=[], execution_status=ExecutionStatus.COMPLETED,
        success=False, score=0.0, feedback=None, metrics={},
    )
    session.begin_task("first")
    session.refine(result, 1)
    assert session.refinement_telemetry["tokens"]["total_tokens"] == 0

    def fail(*args, **kwargs):
        raise RuntimeError("adapter stopped before reporting usage")

    monkeypatch.setattr(session.adaptor, "refine", fail)
    session.begin_task("second")
    with pytest.raises(RuntimeError, match="before reporting usage"):
        session.refine(replace(result, task_id="second"), 2)
    assert session.refinement_telemetry["tokens"]["total_tokens"] is None
    assert session._agent is None


def test_close_failure_clears_session_and_controller_state(controller, monkeypatch):
    controller.begin(task())

    def fail_close():
        raise RuntimeError("close failed")

    monkeypatch.setattr(controller.session.adaptor, "close", fail_close)
    with pytest.raises(RuntimeError, match="close failed"):
        controller.close()
    assert_released(controller)
    assert controller.session._closed
    controller.close()  # Idempotent even after the first cleanup failed.


@pytest.mark.parametrize("tokens", [None, -1, True, 0, 12])
def test_runner_preserves_unknown_inference_cost(tmp_path, controller, monkeypatch, tokens):
    value = outcome()
    value.record["llm_metrics"]["tokens"]["total_tokens"] = tokens
    monkeypatch.setattr(runner, "_evaluate", lambda *args, **kwargs: value)
    record = runner._run_one(task(), controller=controller, ordinal=1, output_dir=tmp_path)
    expected = tokens if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0 else None
    assert record["inference_tokens"] == expected
    assert record["total_tokens"] == expected
