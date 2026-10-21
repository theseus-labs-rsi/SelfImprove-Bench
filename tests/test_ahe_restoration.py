"""Offline checks for upstream-style AHE iteration and runtime construction."""

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rsi_methods.AHE.history import load_history, save_history
from rsi_methods.AHE.init import AHEState, initialize
from rsi_methods.AHE.refine import CandidateBatch, CandidateSpec, _prepare_evolve_runtime
from rsi_methods.AHE.select import select
from rsi_methods.checkpoint import export_state, restore_state


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "rsi_methods/AHE/agentic-harness-engineering/agents/evolve_agent"


def test_initial_actor_uses_game_prompt_without_changing_evolver(tmp_path):
    from jinja2 import StrictUndefined, Template

    state = initialize(tmp_path / "run")
    config = yaml.safe_load((state.current_workspace / "code_agent.yaml").read_text())
    prompt_path = state.current_workspace / config["system_prompt"]
    prompt = prompt_path.read_text()
    source_actor = SOURCE.parent / "code_agent_simple"
    assert prompt == (source_actor / "systemprompt.md").read_text()
    rendered = Template(prompt, undefined=StrictUndefined).render(
        date="2026-10-21", username="offline-test", working_directory="/sandbox",
    )
    assert "You are a game-playing agent." in rendered
    assert "Shell commands are not in-game actions" in rendered
    assert "screenshots" in rendered
    assert "complete plan or action sequence" in rendered
    assert "You solve software tasks" not in rendered
    assert "{{" not in rendered
    assert any(tool["name"] == "run_shell_command" for tool in config["tools"])
    assert (state.evolve_agent_dir / "evolve_prompt.md").read_bytes() == (
        SOURCE / "evolve_prompt.md"
    ).read_bytes()


@pytest.mark.parametrize("status,score,accepted", [
    ("ok", 0.1, True), ("ok", 0.9, True), ("error", 0.0, False),
])
def test_active_lineage_and_same_task_evidence(tmp_path, status, score, accepted):
    parent = tmp_path / "workspace"
    parent.mkdir()
    work = tmp_path / "candidates/iteration_001"
    candidate = work / "variant_0/workspace"
    candidate.mkdir(parents=True)
    (candidate.parent / "change_manifest.json").write_text(
        json.dumps({"iteration": 1, "changes": [{"id": "chg-1"}]}),
    )
    state = AHEState(tmp_path, parent, tmp_path / "evolve_agent")
    batch = CandidateBatch("iteration_001", work, parent, 0.8, (
        CandidateSpec("variant_0", candidate, SimpleNamespace(workspace=candidate.resolve())),
    ))
    save_history(tmp_path, [{
        "refinement_id": "iteration_001", "task_id": "task-A", "parent_score": 0.8,
    }])
    outcome = select(
        state, batch, parent_evaluation={"status": "ok", "relative_score": 0.8},
        candidate_evaluations={"variant_0": {"status": status, "relative_score": score}},
        output_dir=tmp_path / "selection",
    )
    assert (outcome.status == "committed") is accepted
    assert state.current_workspace == (candidate if accepted else parent)
    record = load_history(tmp_path)[0]
    assert record["task_id"] == "task-A"
    evidence = json.loads((tmp_path / record["change_evaluation"]).read_text())
    assert evidence["scope"] == "same_task_parent_vs_candidate"
    assert evidence["candidates"][0]["manifest"]["changes"][0]["id"] == "chg-1"
    if accepted:
        assert record["score_delta"] == pytest.approx(score - 0.8)
    else:
        assert record["score_delta"] is None


def test_retains_compaction_and_diagnostic_prompt(tmp_path):
    config_path = _prepare_evolve_runtime(SOURCE, tmp_path / "runtime")
    config = yaml.safe_load(config_path.read_text())
    compaction = next(m for m in config["middlewares"] if "context_compaction" in m["import"])
    assert compaction["params"]["max_context_tokens"] == config["max_context_tokens"]
    prompt = (config_path.parent / "evolve_prompt.md").read_text()
    assert "root causes already identified" in prompt
    assert "may not contain a root-cause diagnosis" not in prompt
    assert "mean benchmark score per attempt" in prompt
    assert any(t["name"] == "read_visual_file" for t in config["tools"])


def test_debugger_scoped_settings_ignore_global_profile(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE / "skills/agent-debugger-cli/_source"))
    resolver = importlib.import_module("agent_debugger_core.cli.llm_resolver")
    monkeypatch.setattr(resolver, "load_config", lambda: {"llm": {
        "model": "unrelated", "base_url": "https://unrelated.invalid", "api_key": "unrelated",
    }})
    monkeypatch.setenv("AHE_DEBUGGER_SCOPED_LLM", "1")
    monkeypatch.setenv("QA_MODEL_NAME", "also-unrelated")
    monkeypatch.setenv("LLM_MODEL", "method-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("LLM_API_KEY", "test")
    assert resolver.resolve_llm_settings()["model"] == "method-model"


def test_checkpoint_restores_only_committed_history(tmp_path):
    workspace = tmp_path / "workspace"
    evolve = tmp_path / "evolve_agent"
    workspace.mkdir()
    evolve.mkdir()
    (workspace / "code_agent.yaml").write_text("name: offline\n")
    (evolve / "evolve_agent.yaml").write_text("name: evolve\n")
    adaptor = SimpleNamespace(
        run_dir=tmp_path, state=AHEState(tmp_path, workspace, evolve),
        _refinement_round=1, _observed_training_tasks=1,
        checkpoint_policy=lambda: {"version": "test"},
        _configure=lambda _: None, _load_config=lambda: {},
    )
    first = {"refinement_id": "iteration_001", "task_id": "A", "parent_score": 0.8}
    save_history(tmp_path, [first])
    snapshot = export_state(adaptor, "ahe")
    save_history(tmp_path, [first, {
        "refinement_id": "iteration_002", "task_id": "B", "parent_score": 1.2,
    }])
    restore_state(adaptor, "ahe", tmp_path, snapshot)
    assert load_history(tmp_path) == [first]
    assert "iteration_002" not in (tmp_path / "evolution_history.md").read_text()
    assert export_state(adaptor, "ahe") == snapshot


def test_real_debugger_and_evolver_construct_without_api_calls(tmp_path):
    pytest.importorskip("nexau")
    config = _prepare_evolve_runtime(SOURCE, tmp_path / "runtime")
    env = os.environ.copy()
    env.update(
        AHE_HOME=str(SOURCE.parents[1]), EVOLVE_WORK_DIR=str(tmp_path),
        LLM_MODEL="offline-test", LLM_BASE_URL="https://example.invalid/v1", LLM_API_KEY="test",
        PYTHONPATH=os.pathsep.join([
            str(SOURCE / "skills/agent-debugger-cli/_source"),
            str(config.parent), str(SOURCE.parents[1]),
        ]),
    )
    code = """
import sys
from pathlib import Path
from agent_debugger_core.runtime.runner import _build_agent
debugger = _build_agent({
    'model': 'offline-test', 'base_url': 'https://example.invalid/v1', 'api_key': 'test',
})
from nexau import Agent, AgentConfig
evolver = Agent(config=AgentConfig.from_yaml(config_path=Path(sys.argv[1])))
print('constructed both agents without running either')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(config)],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    assert "constructed both agents" in completed.stdout


def test_real_ahe_runtime_constructs_inside_filesystem_namespace(tmp_path):
    pytest.importorskip("nexau")
    if sys.platform != "linux" or shutil.which("bwrap") is None:
        pytest.skip("Linux bubblewrap is required")
    probe = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--unshare-user", "/bin/true"],
                           capture_output=True, text=True)
    if probe.returncode:
        pytest.skip(f"User namespaces unavailable: {probe.stderr}")

    from rsi_methods.AHE.build import build
    from rsi_methods.AHE.sandbox import sandbox_command

    state = initialize(tmp_path / "run")
    credentials = {
        "LLM_MODEL": "offline-test", "LLM_BASE_URL": "https://example.invalid/v1",
        "LLM_API_KEY": "test", "LLM_API_TYPE": "openai_chat_completion",
    }
    actor = build(state, model="offline-test", llm_environment=credentials, timeout_seconds=60)
    try:
        snapshot = actor.snapshot_session()
        assert snapshot["history"] == []
    finally:
        actor.close()
    assert not actor._process.is_alive()

    config = _prepare_evolve_runtime(SOURCE, tmp_path / "runtime")
    env = os.environ.copy()
    env.update(credentials, AHE_HOME=str(SOURCE.parents[1]), EVOLVE_WORK_DIR=str(tmp_path))
    code = """
import sys
from pathlib import Path
from agent_debugger_core.runtime.runner import _build_agent, _usage_snapshot
from nexau import Agent, AgentConfig
debugger = _build_agent({
    'model': 'offline-test', 'base_url': 'https://example.invalid/v1', 'api_key': 'test',
})
assert _usage_snapshot(debugger)['model_calls'] == 0
evolver = Agent(config=AgentConfig.from_yaml(config_path=Path(sys.argv[1])))
print('constructed isolated runtimes without inference')
"""
    completed = subprocess.run(sandbox_command(
        [sys.executable, "-c", code, str(config)], readable=[tmp_path], cwd=tmp_path,
        python_paths=[SOURCE / "skills/agent-debugger-cli/_source", config.parent, SOURCE.parents[1]],
    ), env=env, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    assert "constructed isolated runtimes" in completed.stdout
