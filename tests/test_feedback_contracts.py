"""Missing evidence must not become an observed zero or crash a consumer."""

import json

import pytest

from benchmark.contracts import ExecutionStatus, FeedbackResult, TaskEvaluation, to_jsonable
from benchmark.game_adapters import _aggregate_hero_feedback, _native_feedback_outcome
from benchmark.metrics import record_score
from benchmark.rsi_session import make_feedback_result
from benches.gameworld.benchmark import GameWorldBenchmark
from rsi_methods.AHE.refine import _read_feedback
from rsi_methods.HyperAgents.HyperAgentsAdaptor import HyperAgentsAdaptor


def feedback(score, status=ExecutionStatus.COMPLETED):
    return make_feedback_result(
        task_id="case", task={}, trajectory=[], execution_status=status,
        success=False, score=score, feedback=None, metrics={},
    )


@pytest.mark.parametrize("score,expected", [
    (None, None), (0, 0.0), (0.8, 0.8), (True, None),
    (float("nan"), None), (float("inf"), None),
])
def test_feedback_preserves_missing_vs_zero(score, expected):
    result = feedback(score)
    assert result.evaluation.score == expected
    assert json.loads(json.dumps(to_jsonable(result), allow_nan=False))["evaluation"]["score"] == expected


@pytest.mark.parametrize("summary", [None, {
    "finalized": False, "task_status": "success", "progress": 1.0,
}])
def test_gameworld_missing_or_invalid_summary_is_not_success(tmp_path, summary):
    if summary is not None:
        (tmp_path / "agent_0/evaluation").mkdir(parents=True)
        (tmp_path / "agent_0/evaluation/summary.json").write_text(json.dumps(summary))
    result = GameWorldBenchmark._feedback({"task_id": "case"}, tmp_path)
    assert result.evaluation.execution_status == ExecutionStatus.ERROR
    assert result.evaluation.score is None
    assert not result.evaluation.success
    # This is the actual downstream refinement consumer, without a model call.
    native = HyperAgentsAdaptor._feedback_to_result([result])
    assert not native["valid"]
    assert native["selection_score"] is None
    assert native["metrics"]["score_mean"] is None


@pytest.mark.parametrize("status", list(ExecutionStatus))
def test_hero_missing_seed_score_is_unavailable_and_counts_as_failed_attempt(status):
    task = {"task_id": "suite", "task_type": "synthetic", "difficulty": 1}
    combined = _aggregate_hero_feedback(task, [feedback(0.8), feedback(None, status)])
    assert combined.evaluation.score is None
    assert combined.evaluation.execution_status == ExecutionStatus.ERROR
    outcome = _native_feedback_outcome("heros_journey", task, combined)
    assert outcome.record["relative_score"] is None
    assert record_score(outcome.record) == 0.0
    assert outcome.record["llm_metrics"]["tokens"]["total_tokens"] is None


def test_completed_zero_is_valid_for_all_numeric_consumers():
    result = feedback(0.0)
    hyper = HyperAgentsAdaptor._feedback_to_result([result])
    assert hyper["valid"] and hyper["selection_score"] == 0.0
    payload, baseline = _read_feedback([result])
    assert baseline == payload["evaluation"]["score"] == 0.0


def test_ahe_failed_missing_score_has_explicit_zero_selection_baseline():
    payload, baseline = _read_feedback([feedback(None, ExecutionStatus.ERROR)])
    assert payload["evaluation"]["score"] is None
    assert baseline == 0.0
    with pytest.raises(ValueError, match="finite benchmark score"):
        _read_feedback([feedback(None)])


def test_hyper_rejects_empty_batch_and_nonfinite_scores_without_arithmetic_failure():
    with pytest.raises(ValueError, match="at least one"):
        HyperAgentsAdaptor._feedback_to_result([])
    result = FeedbackResult("case", {}, (), TaskEvaluation(
        ExecutionStatus.COMPLETED, False, float("nan"),
    ))
    assert HyperAgentsAdaptor._feedback_to_result([result])["selection_score"] is None
