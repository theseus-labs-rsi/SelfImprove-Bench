from benchmark.contracts import (
    AgentCallRecord,
    ExecutionStatus,
    FeedbackResult,
    StepFeedback,
    TaskEvaluation,
    TokenUsage,
)
from benchmark.token_ledger import aggregate_feedback_usage


def result(task_id, usages):
    steps = tuple(
        StepFeedback(
            step_index=index, prompt="", image_records=(), action="",
            environment_response={}, success=None, score=None, feedback=None,
            agent_call=AgentCallRecord(
                response_text="answer",
                usage=TokenUsage(**usage) if usage is not None else None,
            ),
        )
        for index, usage in enumerate(usages)
    )
    return FeedbackResult(
        task_id=task_id, task={}, trajectory=steps,
        evaluation=TaskEvaluation(ExecutionStatus.COMPLETED, True, 1.0),
    )


def test_missing_call_usage_uses_mean_from_same_task_only():
    report = aggregate_feedback_usage([
        result("round-a", [
            {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            None,
            {"input_tokens": 200, "output_tokens": 40, "total_tokens": 240},
        ]),
        result("round-b", [None]),
    ])

    assert report["usage"] == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
    }
    assert report["usage_estimation"] == {
        "method": "within_task_observed_call_mean",
        "imputed_task_count": 1,
        "imputed_field_count": 3,
        "unresolved_fields": [
            "input_tokens", "output_tokens", "total_tokens",
            "cached_input_tokens", "reasoning_tokens",
        ],
    }


def test_missing_fields_are_imputed_independently_within_each_task():
    report = aggregate_feedback_usage([
        result("round-a", [
            {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            {"input_tokens": 200, "output_tokens": 40, "total_tokens": 240},
            {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        ]),
    ])

    assert report["usage"] == {
        "input_tokens": 450,
        "output_tokens": 90,
        "total_tokens": 540,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
    }
    assert report["usage_estimation"]["imputed_task_count"] == 1
    assert report["usage_estimation"]["imputed_field_count"] == 3


def test_no_model_calls_have_zero_cost_without_imputation():
    report = aggregate_feedback_usage([result("round-a", [])])
    assert report["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    assert report["usage_estimation"]["imputed_task_count"] == 0


def test_failed_task_without_call_evidence_has_unknown_cost():
    failed = FeedbackResult(
        task_id="failed", task={}, trajectory=(),
        evaluation=TaskEvaluation(ExecutionStatus.ERROR, False, None),
    )
    report = aggregate_feedback_usage([failed])
    assert all(value is None for value in report["usage"].values())
    assert report["usage_estimation"]["unresolved_fields"] == [
        "input_tokens", "output_tokens", "total_tokens",
        "cached_input_tokens", "reasoning_tokens",
    ]
