import json

from benchmark.client import _usage, request_json


def test_incomplete_provider_usage_stays_unknown_instead_of_text_estimate():
    usage = _usage({"input_tokens": 20}, "a long prompt", "a long answer")
    assert usage["total_tokens"] is None
    assert usage["prompt_tokens"] is None
    assert usage["answer_tokens"] is None
    assert usage["method"] == "unavailable"


def test_zero_provider_total_is_preserved():
    usage = _usage({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    assert usage["total_tokens"] == 0
    assert usage["method"] == "provider_reported"


def test_success_without_usage_is_unknown_not_locally_estimated():
    metrics = {}
    request_json(
        {}, prompt="large prompt", provider="codex",
        validator=lambda value, task: value,
        callers={"codex": lambda *args, **kwargs: json.dumps({"ok": True})},
        metrics=metrics,
    )
    assert metrics["tokens"]["total_tokens"] is None
    assert metrics["tokens"]["method"] == "unavailable"
