"""Offline coverage for parent/candidate budgets across AHE refinement."""

import importlib
import json
from unittest.mock import Mock

import pytest

from rsi_methods.AHE.agent import AHEAgent
from rsi_methods.AHE.build import build
from rsi_methods.AHE.init import AHEState


refinement = importlib.import_module("rsi_methods.AHE.refine")


@pytest.mark.parametrize("actor_timeout,step_cap,fallback_timeout,expected", [
    (1800, None, None, 1800),
    (90, None, None, 90),
    (1800, 600, None, 600),
    (90, 600, None, 90),
    (1800, None, 120, 1800),
])
def test_candidates_keep_parent_budget_separate_from_evolver(
    tmp_path, monkeypatch, actor_timeout, step_cap, fallback_timeout, expected,
):
    # Keep real Agent construction and candidate generation, but replace the
    # process/inference boundaries. All evidence is an isolated synthetic fixture.
    monkeypatch.setattr(AHEAgent, "_start_worker", lambda self: None)
    for name, value in (
        ("RSI_AHE_STEP_TIMEOUT", step_cap),
        ("RSI_AHE_ACTOR_TIMEOUT", fallback_timeout),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(value))

    run_dir = tmp_path / "run"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "code_agent.yaml").write_text("name: synthetic-timeout-test\n")
    state = AHEState(run_dir, workspace, run_dir / "evolve_agent")
    parent = build(state, timeout_seconds=actor_timeout)
    trace = parent.artifact_dir / "synthetic_trace.json"
    trace.write_text(json.dumps({"messages": [], "synthetic_fixture": True}))
    parent.trace_paths.append(trace)

    monkeypatch.setattr(refinement, "_write_analysis", lambda *args, **kwargs: None)
    evolve = Mock(return_value={"usage_snapshot": None})
    monkeypatch.setattr(refinement, "_run_evolver", evolve)
    batch = refinement.refine(
        state, parent,
        [{"task_id": "synthetic-timeout-test", "evaluation": {"score": 0.0}}],
        output_dir=tmp_path / "refinement", iteration=1, timeout_seconds=73,
    )

    assert parent.timeout_seconds == expected
    assert len(batch.candidates) == 1
    assert batch.candidates[0].agent.timeout_seconds == expected
    assert batch.candidates[0].agent is not parent
    # The evolver has its own wall-clock budget; it must not become the actor's.
    evolve.assert_called_once()
    assert evolve.call_args.args[7] == 73
