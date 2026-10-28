import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


FIX = Path(__file__).parent / "fixtures"


def _setup_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_BASE_URL", "u")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("AHE_HOME", str(tmp_path))
    (tmp_path / "evolve_agent" / "tools").mkdir(parents=True)
    (tmp_path / "evolve_agent" / "tools" / "__init__.py").write_text("")


def test_cli_ask_json_format(tmp_path, monkeypatch, capsys):
    _setup_env(tmp_path, monkeypatch)
    trace = FIX / "openai_messages_sample.json"

    from agent_debugger_core.runtime.runner import RunnerResult
    from agent_debugger_core.cli import adb as adb_cli

    usage = {"total_tokens": 13, "input_tokens": 8, "output_tokens": 5, "model_calls": 1}
    fake_result = RunnerResult(mode="ask", answer="because", iterations=3, usage_snapshot=usage)
    with patch("agent_debugger_core.cli.adb.run_agent", return_value=fake_result):
        rc = adb_cli.main(["ask", "-t", str(trace), "-q", "why?", "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["status"] == "success"
    assert payload["command"] == "ask"
    assert payload["response"] == "because"
    assert payload["question"] == "why?"
    assert payload["usage_snapshot"] == usage


def test_cli_ask_text_format(tmp_path, monkeypatch, capsys):
    _setup_env(tmp_path, monkeypatch)
    trace = FIX / "openai_messages_sample.json"

    from agent_debugger_core.runtime.runner import RunnerResult
    from agent_debugger_core.cli import adb as adb_cli

    fake_result = RunnerResult(mode="ask", answer="because", iterations=3)
    with patch("agent_debugger_core.cli.adb.run_agent", return_value=fake_result):
        rc = adb_cli.main(["ask", "-t", str(trace), "-q", "why?"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "because" in captured.out


@pytest.mark.parametrize("inner_result,expected_error", [
    ('{"mode":"ask","answer":"literal\nnewline"}', "not valid JSON"),
    (json.dumps({"mode": "check", "issues": [], "response": "report"}), "expected mode=ask"),
])
def test_cli_returns_structured_failure_with_original_output(
    tmp_path, monkeypatch, capsys, inner_result, expected_error,
):
    _setup_env(tmp_path, monkeypatch)
    from agent_debugger_core.cli import adb as adb_cli

    raw = json.dumps({
        "status": "TASK_COMPLETED", "task_completed": True,
        "output": {"result": inner_result},
    })
    fake_agent = MagicMock()
    fake_agent.run.return_value = raw
    with patch("agent_debugger_core.runtime.runner._build_agent", return_value=fake_agent):
        rc = adb_cli.main([
            "ask", "-t", str(FIX / "openai_messages_sample.json"), "-q", "why?", "--format", "json",
        ])
    assert rc == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "failed"
    assert expected_error in payload["error"]
    assert payload["raw_output"] == raw
    assert "Traceback" not in captured.err
    fake_agent.run.assert_called_once()
