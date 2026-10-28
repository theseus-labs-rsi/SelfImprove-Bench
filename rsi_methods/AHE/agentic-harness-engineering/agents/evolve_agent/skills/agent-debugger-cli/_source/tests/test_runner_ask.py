import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent_debugger_core.runtime.runner import run_agent, RunnerResult, RunnerError


def _wrap(payload):
    inner = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return json.dumps({
        "success": True,
        "message": "Result submitted and task completed.",
        "status": "TASK_COMPLETED",
        "task_completed": True,
        "output": {"result": inner},
    }, ensure_ascii=False)


def _scripted_agent(payload: dict):
    fake_agent = MagicMock()
    fake_agent.run.return_value = _wrap(payload)
    return fake_agent


def test_ask_mode_happy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_BASE_URL", "u")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AHE_HOME", str(tmp_path))
    (tmp_path / "evolve_agent" / "tools").mkdir(parents=True)
    (tmp_path / "evolve_agent" / "tools" / "__init__.py").write_text("")

    fake_agent = _scripted_agent({"mode": "ask", "answer": "42"})

    with patch("agent_debugger_core.runtime.runner._build_agent", return_value=fake_agent):
        result = run_agent(
            trace_paths=[Path("/fake/trace.json")],
            mode="ask",
            question="why?",
        )

    assert isinstance(result, RunnerResult)
    assert result.mode == "ask"
    assert result.answer == "42"
    assert result.budget_exceeded is False
    request = fake_agent.run.call_args.kwargs["message"]
    assert "Task mode: ask" in request
    assert '`mode="ask"`' in request


@pytest.mark.parametrize("budget_exceeded", [False, True])
def test_runner_exports_tracer_usage(monkeypatch, budget_exceeded):
    from agent_debugger_core.runtime import runner

    traces = {"test": "provider spans"}
    usage = {"total_tokens": 13, "input_tokens": 8, "output_tokens": 5, "model_calls": 1}

    class Tracer:
        def dump_traces(self):
            return traces

    monkeypatch.setitem(sys.modules, "nexau.archs.tracer.adapters.in_memory",
                        SimpleNamespace(InMemoryTracer=Tracer))
    fake_agent = _scripted_agent({"mode": "ask", "answer": "diagnosis"})
    fake_agent.config = SimpleNamespace(tracers=[Tracer()])
    if budget_exceeded:
        fake_agent.run.return_value = runner.BUDGET_MARKER + " unfinished"
    with patch.object(runner, "resolve_llm_settings", return_value={}), patch.object(
        runner, "_build_agent", return_value=fake_agent,
    ), patch("benchmark.token_ledger.nexau_usage_snapshot", return_value=usage) as export:
        result = runner.run_agent(trace_paths=[Path("/fake/trace.json")], mode="ask")
    export.assert_called_once_with(traces)
    assert result.usage_snapshot == usage
    assert result.budget_exceeded is budget_exceeded


@pytest.mark.parametrize("payload", [
    {"mode": "check", "issues": [], "response": "wrong mode"},
    {"mode": "check", "answer": "still the wrong mode"},
    {"mode": "ask", "answer": " "},
    {"mode": "ask", "answer": 42},
    {"answer": "mode is missing"},
])
def test_ask_rejects_protocol_mismatch_without_extra_model_calls(payload):
    from agent_debugger_core.runtime import runner

    fake_agent = _scripted_agent(payload)
    with patch.object(runner, "resolve_llm_settings", return_value={}), patch.object(
        runner, "_build_agent", return_value=fake_agent,
    ), pytest.raises(RunnerError) as error:
        runner.run_agent(trace_paths=[Path("/fake/trace.json")], mode="ask")
    assert error.value.raw_output == fake_agent.run.return_value
    fake_agent.run.assert_called_once()


@pytest.mark.parametrize("payload", [
    '{"mode":"ask","answer":"literal\nnewline"}',
    '{"mode":"ask","answer":"unterminated}',
    '[]', 'null', '42',
])
def test_malformed_inner_result_is_a_runner_error(payload):
    from agent_debugger_core.runtime import runner

    with pytest.raises(RunnerError, match="complete_task result"):
        runner._parse_run_output(_wrap(payload))


def test_escaped_newlines_and_double_encoded_outer_result_remain_valid():
    from agent_debugger_core.runtime import runner

    payload = {"mode": "ask", "answer": "first line\nsecond line"}
    assert runner._parse_run_output(json.dumps(_wrap(payload))) == payload
