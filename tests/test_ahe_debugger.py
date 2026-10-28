"""AHE must diagnose successfully before it can generate candidates."""

import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmark.rsi_session import is_fatal_ahe_error

refinement = importlib.import_module("rsi_methods.AHE.refine")


def test_uses_bundled_debugger_and_method_environment(tmp_path):
    completed = SimpleNamespace(
        returncode=0, stdout=json.dumps({"status": "success", "response": " diagnosis "}), stderr="",
    )
    with patch.dict(refinement.os.environ, {}, clear=True), patch.object(
        refinement.subprocess, "run", return_value=completed,
    ) as run:
        assert refinement._ask_debugger(
            tmp_path / "trace.json", 0.2, "feedback", model="method-model",
            llm_environment={"OPENAI_BASE_URL": "https://example.invalid/v1", "OPENAI_API_KEY": "test-key"},
        ) == "diagnosis"
    args, kwargs = run.call_args
    command = args[0][args[0].index("--") + 1:]
    assert command[:3] == [refinement.sys.executable, "-m", "agent_debugger_core.cli.adb"]
    assert "--ro-bind" in args[0]
    env = kwargs["env"]
    assert env["LLM_MODEL"] == "method-model"
    assert env["LLM_BASE_URL"] == "https://example.invalid/v1"
    assert env["LLM_API_KEY"] == "test-key"
    assert (Path(env["AHE_HOME"]) / "agents/evolve_agent/tools/__init__.py").is_file()
    assert "agent-debugger-cli/_source" in env["PYTHONPATH"]


@pytest.mark.parametrize("payload", [
    "not json", "[]", '{"status":"failed","response":"error"}',
    '{"status":"success","response":" "}', '{"status":"success","response":{}}',
    '{"status":"success","response":"[budget-exceeded] unfinished analysis"}',
    '{"response":"missing success status"}',
])
def test_rejects_invalid_or_failed_diagnosis(tmp_path, payload):
    with patch.object(refinement.subprocess, "run", return_value=SimpleNamespace(
        returncode=0, stdout=payload, stderr="",
    )), pytest.raises(RuntimeError, match="AHE Agent Debugger") as error:
        refinement._ask_debugger(tmp_path / "trace.json", 0, "")
    assert is_fatal_ahe_error(str(error.value))


@pytest.mark.parametrize("failure", [
    SimpleNamespace(returncode=1, stdout="", stderr="missing LLM configuration"),
    subprocess.TimeoutExpired("debugger", 600),
    OSError("cannot start interpreter"),
])
def test_process_failure_is_fatal(tmp_path, failure):
    kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
    with patch.object(refinement.subprocess, "run", **kwargs), pytest.raises(RuntimeError) as error:
        refinement._ask_debugger(tmp_path / "trace.json", 0, "")
    assert is_fatal_ahe_error(str(error.value))


def test_diagnosis_failure_prevents_evolver_and_candidate_build(tmp_path):
    parent = tmp_path / "workspace"
    parent.mkdir()
    (parent / "code_agent.yaml").write_text("name: test\n")
    state = SimpleNamespace(run_dir=tmp_path, current_workspace=parent)
    agent = SimpleNamespace(workspace=parent, trace_paths=[tmp_path / "trace.json"])
    feedback = {"task_id": "test", "evaluation": {"score": 0.2}}
    with patch.object(refinement, "_read_feedback", return_value=(feedback, 0.2)), patch.object(
        refinement, "_trace_from_feedback", return_value=agent.trace_paths[0],
    ), patch.object(refinement, "_prepare_evidence", return_value=(tmp_path / "trial", "test")), patch.object(
        refinement, "_ask_debugger", side_effect=RuntimeError("AHE Agent Debugger failed"),
    ), patch.object(refinement, "_run_evolver") as evolve, patch.object(refinement, "build") as build:
        with pytest.raises(RuntimeError, match="Debugger failed"):
            refinement.refine(state, agent, [], output_dir=tmp_path / "output", iteration=1)
    evolve.assert_not_called()
    build.assert_not_called()
    assert not (tmp_path / "candidates").exists()


def test_debugger_usage_is_saved_with_the_diagnosis(tmp_path):
    usage = {"total_tokens": 13, "input_tokens": 8, "output_tokens": 5, "model_calls": 1}
    completed = SimpleNamespace(returncode=0, stderr="E2B SDK not installed\n", stdout=json.dumps({
        "status": "success", "response": "diagnosis", "usage_snapshot": usage,
    }))
    with patch.object(refinement.subprocess, "run", return_value=completed):
        refinement._ask_debugger(tmp_path / "trace.json", 0.2, "", output_dir=tmp_path / "analysis")
    assert json.loads((tmp_path / "analysis/debugger_usage_snapshot.json").read_text()) == usage
    assert (tmp_path / "analysis/debugger_stderr.txt").read_text() == completed.stderr
    assert json.loads((tmp_path / "analysis/debugger_process.json").read_text())["returncode"] == 0


@pytest.mark.parametrize("returncode", [0, 1])
def test_debugger_reports_structured_error_despite_stderr_warning(tmp_path, returncode):
    secret = "synthetic-api-key"
    error_text = f"expected mode=ask, got 'check'; credential={secret}"
    stdout = json.dumps({"status": "failed", "error": error_text, "raw_output": "original model result"})
    stderr = "E2B SDK not installed\n"
    completed = SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    with patch.object(refinement.subprocess, "run", return_value=completed) as run:
        with pytest.raises(RuntimeError) as error:
            refinement._ask_debugger(
                tmp_path / "trace.json", 0.2, "", output_dir=tmp_path / "analysis",
                llm_environment={"LLM_API_KEY": secret},
            )
    run.assert_called_once()
    assert "expected mode=ask" in str(error.value)
    assert "E2B SDK not installed" in str(error.value)
    assert secret not in str(error.value)
    assert is_fatal_ahe_error(str(error.value))
    saved = json.loads((tmp_path / "analysis/debugger_stdout.txt").read_text())
    assert saved["raw_output"] == "original model result"
    assert secret not in saved["error"]
    assert (tmp_path / "analysis/debugger_stderr.txt").read_text() == stderr
    assert not (tmp_path / "analysis/debugger_result.json").exists()


def test_debugger_timeout_preserves_partial_bytes_and_redacts_credentials(tmp_path):
    secret = "synthetic-timeout-key"
    failure = subprocess.TimeoutExpired(
        "debugger", 600, output=f"partial output {secret}".encode(), stderr=b"warning\xff",
    )
    with patch.object(refinement.subprocess, "run", side_effect=failure) as run:
        with pytest.raises(RuntimeError, match="Debugger timed out"):
            refinement._ask_debugger(
                tmp_path / "trace.json", 0.2, f"note {secret}", output_dir=tmp_path / "analysis",
                llm_environment={"LLM_API_KEY": secret},
            )
    run.assert_called_once()
    assert (tmp_path / "analysis/debugger_stdout.txt").read_text() == "partial output <redacted>"
    assert (tmp_path / "analysis/debugger_stderr.txt").read_text().startswith("warning")
    metadata_text = (tmp_path / "analysis/debugger_process.json").read_text()
    assert secret not in metadata_text
    metadata = json.loads(metadata_text)
    assert metadata["timed_out"] is True
    assert metadata["returncode"] is None


def test_debugger_start_failure_is_recorded(tmp_path):
    with patch.object(refinement.subprocess, "run", side_effect=OSError("interpreter unavailable")):
        with pytest.raises(RuntimeError, match="interpreter unavailable"):
            refinement._ask_debugger(tmp_path / "trace.json", 0.2, "", output_dir=tmp_path / "analysis")
    metadata = json.loads((tmp_path / "analysis/debugger_process.json").read_text())
    assert metadata["returncode"] is None
    assert metadata["error"] == "OSError: interpreter unavailable"


@pytest.mark.parametrize("debugger_usage,expected", [
    ({"total_tokens": 13, "input_tokens": 8, "output_tokens": 5, "model_calls": 2}, 43),
    (None, None), ({"total_tokens": None}, None), ({"total_tokens": True}, None),
    ({"total_tokens": -1}, None), ({"total_tokens": 0, "model_calls": 0}, None),
])
def test_refinement_cost_includes_debugger_exactly_once(tmp_path, debugger_usage, expected):
    report = refinement._evolve_token_report([
        {"candidate_id": "first", "usage": {"total_tokens": 10, "input_tokens": 6, "output_tokens": 4}},
        {"candidate_id": "second", "usage": {"total_tokens": 20, "input_tokens": 12, "output_tokens": 8}},
    ], debugger_invoked=True, debugger_usage=debugger_usage, source_path=tmp_path / "report.json")
    assert report["total_tokens"] == expected
    assert report["complete"] is (expected is not None)
    if expected is not None:
        assert report["prompt_num_tokens"] == 26
        assert report["response_num_tokens"] == 17
        assert report["reasoning_num_tokens"] is None


def test_missing_evolver_usage_does_not_report_debugger_only_total(tmp_path):
    report = refinement._evolve_token_report([{"usage": None}], debugger_invoked=True,
        debugger_usage={"total_tokens": 13}, source_path=tmp_path / "report.json")
    assert not report["complete"]
    assert report["total_tokens"] is None
