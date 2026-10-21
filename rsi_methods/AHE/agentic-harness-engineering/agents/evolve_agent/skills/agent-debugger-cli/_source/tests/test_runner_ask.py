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
