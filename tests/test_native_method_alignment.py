"""Exercise native controllers through adapters with deterministic API/game doubles."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import re

import pytest

from benchmark.contracts import (BackendResponse, TokenUsage, ToolCall, FeedbackResult,
    StepFeedback, TaskEvaluation, ExecutionStatus, EvaluationOutcome)
from benchmark.interfaces import MethodContext
from benchmark.methods import create_adaptor
from benchmark.config import AgentConfig, BenchmarkConfig
from benchmark.runner import AgentController
from benchmark.tasks import _make_task, task_fingerprint


METHODS = ("harness2", "recuris", "rrsi", "skillevolver", "skillopt")
ADAPTERS = {"harness2": "rsi_methods.harness2.SelfImproveAdapter:Harness2Adapter",
            "recuris": "rsi_methods.Recuris.SelfImproveAdapter:RecurisAdapter",
            "rrsi": "rsi_methods.rrsi.SelfImproveAdapter:RRSIAdapter",
            "skillevolver": "rsi_methods.skillevolver.SelfImproveAdapter:SkillEvolverAdapter",
            "skillopt": "rsi_methods.SkillOpt.SelfImproveAdapter:SkillOptAdapter"}


def response(text="action", calls=()):
    return BackendResponse(text=text, usage=TokenUsage(2, 3, 5, 0, 0), tool_calls=tuple(calls))


def feedback(score=.5):
    return FeedbackResult("train-example", {}, (StepFeedback(0, "Public task rules", (),
        "action", {"public": "observation"}, False, score, "training feedback"),),
        TaskEvaluation(ExecutionStatus.COMPLETED, False, score, "training feedback", {"game": "farm"}))


def rollout_result(score=.5, success=False):
    return {"record": {"status": "ok", "relative_score": score, "success": success,
        "llm_metrics": {"tokens": {"total_tokens": 5, "prompt_tokens": 2,
                                   "answer_tokens": 3, "reasoning_tokens": 0}}},
        "feedback": feedback(score), "task_id": "internal", "seconds": .01}


def adapter(tmp_path, method, **options):
    with patch("benchmark.method_support.OpenAIBackend"):
        obj = create_adaptor("rsi", artifact_dir=tmp_path / method, task_count=5,
            agent=AgentConfig(provider="rsi", adapter=ADAPTERS[method], adapter_options=options))
    obj.init()
    obj.dispatcher = SimpleNamespace(distribute=lambda *a, **kw: response())
    return obj


@pytest.mark.parametrize("method", METHODS)
def test_adapter_entry_checkpoint_and_acting_snapshot(tmp_path, method):
    obj = adapter(tmp_path, method)
    if method == "recuris":
        replies = iter([response('{"requests": []}'), response(), response()])
        obj.dispatcher.distribute = lambda *a, **kw: next(replies)
    assert obj.load().act("Public task").text == "action"
    snapshot = obj.export_state()
    obj.restore_state(snapshot)
    assert obj.export_state() == snapshot


@pytest.mark.parametrize("mode,final_prefix", [("sequential", "c"), ("parallel", "agg")])
def test_harness_uses_native_4k_rollouts_3k_improvements_and_finals_only(tmp_path, mode, final_prefix):
    from rsi_methods.harness2.harness2.core import selection
    obj = adapter(tmp_path, "harness2", recursion_mode=mode, proposal_count=2)
    labels, calls = [], []
    obj.bind_evaluator(lambda agent, **kw: (labels.append(kw["label"]) or rollout_result()))
    def ask(prompt, *, stage):
        calls.append((stage, prompt))
        if re.search(r"\.a\d+$", stage):
            return json.dumps({"files": {**obj._state["files"], "systemprompt.md": "Improved transferable policy."},
                               "reasoning": "preserve useful behavior"})
        return "```choice\nA\n```\n<why>Coherent public deliverable</why><experience>Reusable lesson</experience>"
    obj.ask_text = ask
    real_select = selection.run_selection
    seen_pools = []
    def select(*args, **kwargs):
        seen_pools.append(set(args[3]))
        return real_select(*args, **kwargs)
    with patch.object(selection, "run_selection", select):
        obj.refine([feedback()], output_dir=tmp_path / "round")
    assert len(labels) == 8
    assert len([s for s, _ in calls if re.search(r"\.a\d+$", s)]) == 6
    assert seen_pools == [{f"{final_prefix}1", f"{final_prefix}2"}]
    assert obj._version == 1
    assert obj._state["files"]["systemprompt.md"] == "Improved transferable policy."
    assert not json.loads((tmp_path / "round/native_recursion.json").read_text())["fell_back"]
    for _, prompt in calls:
        assert '"relative_score"' not in prompt
        assert "training feedback" not in prompt


def test_file_tools_execute_helpers_and_delegate_in_separate_context(tmp_path):
    from benchmark.artifact_agent import ArtifactAgent
    obj = adapter(tmp_path, "skillevolver")
    replies = iter([
        response("", [ToolCall("a", "read_file", {"path": "references/lesson.md"}),
                       ToolCall("b", "run_script", {"path": "scripts/helper.py"})]),
        response("", [ToolCall("c", "delegate", {"path": "sub_agents/expert.md", "task": "compute"})]),
        response("child answer"), response("final action")])
    messages = []
    def dispatch(*args, **kw):
        messages.append(deepcopy(kw["messages"]))
        return next(replies)
    obj.dispatcher.distribute = dispatch
    agent = ArtifactAgent(obj, {"SKILL.md": "router", "references/lesson.md": "read me",
        "scripts/helper.py": "print(6 * 7)", "sub_agents/expert.md": "private expert"},
        "instructions", always_on=("SKILL.md",), root=tmp_path / "files")
    result = agent.act("public prompt")
    assert result.text == "final action"
    assert result.usage.total_tokens == 20
    assert any('42' in m.get("content", "") for m in messages[1])
    assert "read me" not in str(messages[2])
    assert "private expert" in messages[2][0]["content"]


def test_recuris_native_retrieval_ledger_grounding_and_review(tmp_path):
    obj = adapter(tmp_path, "recuris")
    obj._state["files"]["em/procedure.md"] = """---
id: procedure
type: procedure
trigger: {event: intent_recorded, tool: '*'}
source: training
---
Public task rules: verify prerequisites.
"""
    replies = iter([response('{"requests": [{"description": "Public task rules"}]}'),
                    response("", [ToolCall("action-1", "game_action", {"action": "draft"})]),
                    response("reviewed action")])
    seen = []
    def dispatch(*args, **kw):
        seen.append(str(kw["messages"]))
        return next(replies)
    obj.dispatcher.distribute = dispatch
    agent = obj.load()
    result = agent.act("Public task rules")
    assert result.text == "reviewed action"
    assert result.usage.total_tokens == 15
    assert agent.fingerprint["ledger_update_ok"] == 1
    assert agent.fingerprint["retrieval_delivered"] == 1
    assert agent.fingerprint["write_review_bounce"] == 1
    assert "verify prerequisites" in seen[1]
    assert agent.state.ledger.done()  # genuine memory delivery, not claimed game completion


def test_recuris_empty_memory_directory_is_created_before_runtime_load(tmp_path):
    from rsi_methods.Recuris.SelfImproveAdapter import _write_memory_package
    package = _write_memory_package(tmp_path / "package", {"manifest.yaml": "name: test\n"})
    assert (package / "em").is_dir()


def test_recuris_truth_checker_rewrites_unverified_completion_claim(tmp_path):
    obj = adapter(tmp_path, "recuris")
    replies = iter([response('{"requests": [{"description": "Unanswered knowledge need"}]}'),
                    response("I have completed the request."),
                    response("", [ToolCall("action-1", "game_action", {"action": "draft action"})]),
                    response("valid action")])
    obj.dispatcher.distribute = lambda *a, **kw: next(replies)
    agent = obj.load()
    result = agent.act("Resolve this task")
    assert result.text == "valid action"
    assert agent.fingerprint["truth_bounce"] == 1
    assert agent.state.ledger.pending()  # submitting an action never fabricates an execution receipt
    assert result.usage.total_tokens == 20


@pytest.mark.parametrize("activated,expected", [(False, False), (True, True)])
def test_recuris_native_gate_requires_activation_even_with_score_gain(tmp_path, activated, expected):
    obj = adapter(tmp_path, "recuris", validation_trials=2, trials=1, bootstrap_samples=100)
    state = deepcopy(obj._state)
    state["files"]["em/lesson.md"] = "---\nid: lesson\ntype: knowledge\ntrigger: {event: turn_start}\n---\nDurable lesson."
    obj.ask = lambda *a, **kw: {"state": state, "component": "E", "carrier": "deliver_inject",
        "cluster_id": "missing-guidance", "primitive": "standing-card",
        "action_opportunity": "present", "patchability": "memory"}
    obj.bind_evaluator(lambda agent, **kw: rollout_result())
    def run(state, *, label, **kw):
        result = rollout_result(.8 if label.startswith("candidate") else .5)
        result["mechanism_fingerprint"] = {"deliver_inject": int(activated)}
        return result
    obj.rollout = run
    obj.refine([feedback()], output_dir=tmp_path / "round")
    assert obj.last_selection_report["accepted"] is expected
    assert len(obj._state["prescription_ledger"]) == 1


def test_rrsi_declared_edits_match_actual_changed_components(tmp_path):
    obj = adapter(tmp_path, "rrsi")
    state = deepcopy(obj._state)
    state["harness"]["prompt"] = "new prompt"
    state["harness"]["memory"] = ["undeclared change"]
    with pytest.raises(ValueError, match="actual changed"):
        obj.normalize_candidate({"state": state, "edits": [{"component": "prompt", "hypothesis": "test"}]}, current=obj._state)


def test_json_stringified_structures_are_completed_for_rrsi_and_skillevolver(tmp_path):
    rrsi = adapter(tmp_path, "rrsi")
    state = deepcopy(rrsi._state)
    state["harness"]["prompt"] = "stringified structure accepted"
    candidate = rrsi.normalize_candidate({"state": json.dumps(state),
        "edits": json.dumps([{"component": "prompt", "hypothesis": "robustness"}])},
        current=rrsi._state)
    assert candidate["harness"]["prompt"] == "stringified structure accepted"

    skill = adapter(tmp_path, "skillevolver")
    state = deepcopy(skill._state)
    state["references"] = json.dumps({"lesson.md": "verified"})
    state["scripts"] = json.dumps({})
    state["evidence"] = json.dumps([])
    candidate = skill.normalize_candidate({"state": json.dumps(state)}, current=skill._state)
    assert candidate["references"] == {"lesson.md": "verified"}

    skillopt = adapter(tmp_path, "skillopt")
    candidate = skillopt.normalize_candidate({"patch": json.dumps({
        "reasoning": "repair JSON shape", "edits": json.dumps([])})}, current=skillopt._state)
    assert candidate["best_skill"] == skillopt._state["best_skill"]


def test_rrsi_remeasures_frontier_and_cannot_cascade_below_initial_floor(tmp_path):
    obj = adapter(tmp_path, "rrsi", trials=1, candidate_count=1, delta=.1, repair_rounds=0)
    initial = deepcopy(obj._state["harness"])
    obj._state["harness"]["prompt"] = "incumbent"
    obj._state["frontier"]["harnesses"] = [initial]
    def ask(prompt, *, stage):
        if stage == "analyst": return {}
        if stage.startswith("critic"): return {"accept": True}
        state = deepcopy(obj._state)
        state["harness"]["prompt"] = "candidate"
        return {"state": state, "edits": [{"component": "prompt", "hypothesis": "reduce cost"}]}
    obj.ask = ask
    seen = []
    def evaluate(agent, **kw):
        seen.append(kw["label"])
        r = rollout_result(.66 if kw["label"].startswith("candidate") else
                           (.8 if kw["label"].startswith("frontier") else .75))
        if kw["label"].startswith("candidate"):
            r["record"]["llm_metrics"]["tokens"]["total_tokens"] = 1
        return r
    obj.bind_evaluator(evaluate)
    obj.refine([feedback()], output_dir=tmp_path / "round")
    assert "frontier-0-0" in seen
    assert obj.last_selection_report["accepted"] is False
    assert "floor" in obj.last_selection_report["decisions"][0]["reason"]
    assert obj._state["edit_history"]


def test_skillopt_real_reflect_merge_and_validation_gate(tmp_path):
    obj = adapter(tmp_path, "skillopt", validation_trials=2, minibatch_size=1, workers=2)
    calls = []
    def ask(prompt, *, stage):
        calls.append(stage)
        patch = {"reasoning": "trace-supported", "edits": [
            {"op": "append", "content": "Verify transferable prerequisites."}]}
        return json.dumps({"patch": patch} if stage == "analyst" else patch)
    obj.ask_text = ask
    labels = []
    obj.bind_evaluator(lambda agent, **kw: (labels.append(kw["label"]) or
        rollout_result(.8 if kw["label"].startswith("candidate") else .5,
                       kw["label"].startswith("candidate"))))
    obj.refine([feedback(), feedback(.7)], output_dir=tmp_path / "round")
    assert "analyst" in calls
    assert obj.last_selection_report["accepted"]
    assert len(labels) == 4
    assert "Verify transferable prerequisites" in obj._state["best_skill"]
    assert "rejected_edits" not in obj.load().state


def test_skillopt_scoped_transport_propagates_into_threads(tmp_path):
    from benchmark.native_support import native_package
    native_package("skillopt", Path(__file__).resolve().parents[1] / "rsi_methods/SkillOpt/skillopt")
    from skillopt.model import chat_optimizer
    from skillopt.model.transport import optimizer_transport, ContextExecutor
    with ContextExecutor(max_workers=2) as executor:
        token = optimizer_transport.set(lambda **kw: ("adapter-one", {}))
        a = executor.submit(chat_optimizer, "system", "user")
        optimizer_transport.reset(token)
        token = optimizer_transport.set(lambda **kw: ("adapter-two", {}))
        b = executor.submit(chat_optimizer, "system", "user")
        optimizer_transport.reset(token)
        assert a.result()[0] == "adapter-one"
        assert b.result()[0] == "adapter-two"
    assert optimizer_transport.get() is None


def test_skillevolver_executes_fresh_exploration_then_one_final_validation(tmp_path):
    obj = adapter(tmp_path, "skillevolver", iterations=2, exploration_trials=2, validation_trials=1)
    stages, labels = [], []
    def ask(prompt, *, stage):
        stages.append((stage, prompt))
        if stage == "finalize": return {"version": 2, "reason": "training contrast"}
        if stage.endswith("analysis"): return {"lesson": "transfer"}
        return {"state": {**obj.initial_state(), "SKILL.md": stage}}
    obj.ask = ask
    obj.bind_evaluator(lambda agent, **kw: (labels.append((kw["label"], kw["split"])) or rollout_result()))
    obj.refine([feedback()], output_dir=tmp_path / "round")
    assert len([s for s, _ in labels if "trial" in s]) == 4
    assert len([s for s, split in labels if split == "validation"]) == 1
    assert labels[-1][1] == "validation"
    assert stages[-1][0] == "finalize"
    assert obj._state["SKILL.md"] == "iteration-1-update"


def test_skillevolver_failed_final_validation_is_recorded_without_aborting(tmp_path):
    obj = adapter(tmp_path, "skillevolver", iterations=1, exploration_trials=1,
                  validation_trials=2)
    obj.ask = lambda prompt, *, stage: ({"version": 1} if stage == "finalize" else
        {"lesson": "transfer"} if stage.endswith("analysis") else
        {"state": {**obj.initial_state(), "SKILL.md": stage}})
    def evaluate(agent, *, label, **kwargs):
        if label.startswith("final-validation-0"):
            result = rollout_result()
            result["record"].update(status="error", relative_score=None, error="invalid action")
            return result
        return rollout_result()
    obj.bind_evaluator(evaluate)
    obj.refine([feedback()], output_dir=tmp_path / "round")
    assert obj.last_selection_report["accepted"]
    assert obj.last_selection_report["validation_scores"] == [0.0, 0.5]
    assert obj.last_selection_report["validation_failures"][0]["status"] == "error"


def test_skillopt_failed_paired_validation_rollouts_count_as_zero(tmp_path):
    obj = adapter(tmp_path, "skillopt", validation_trials=2, minibatch_size=1, workers=1)
    obj.ask_text = lambda prompt, *, stage: json.dumps({"patch": {"reasoning": "edit",
        "edits": [{"op": "append", "content": "Check prerequisites."}]}})
    calls = {"base": 0, "candidate": 0}
    def evaluate(agent, *, label, **kwargs):
        key = "candidate" if label.startswith("candidate") else "base"
        calls[key] += 1
        result = rollout_result(.8 if key == "candidate" else .5, key == "candidate")
        if calls[key] == 1:
            result["record"].update(status="error", relative_score=None,
                                    success=False, error="invalid action")
        return result
    obj.bind_evaluator(evaluate)
    obj.refine([feedback()], output_dir=tmp_path / "round")
    report = obj.last_selection_report
    assert report["accepted"]
    assert report["candidate_validation_failures"][0]["status"] == "error"
    assert report["base_validation_failures"][0]["status"] == "error"


def test_internal_validation_is_paired_order_independent_and_excludes_outer_tasks(tmp_path):
    train = _make_task(phase="train", game="farm", difficulty=1, index=0, group_index=0, used=set(), seed_root=42)
    holdout = _make_task(phase="holdout", game="farm", difficulty=1, index=0, group_index=0, used=set(), seed_root=43)
    config = BenchmarkConfig(seed=42, schedule=(), holdout=(), agent=AgentConfig(provider="openai"))
    def controller():
        c = AgentController(config, tmp_path, 1, reserved_tasks=(train, holdout))
        c.request_for_agent = lambda agent: None
        return c
    c, resumed = controller(), controller()
    with patch("benchmark.runner._evaluate", return_value=EvaluationOutcome(rollout_result()["record"], feedback())):
        a = c._method_rollout(train, None, label="a", split="validation", index=2, output_dir=tmp_path / "a")
        for i in range(3):
            b = resumed._method_rollout(train, None, label=f"b{i}", split="validation", index=i, output_dir=tmp_path / f"b{i}")
        assert a["task_id"] == b["task_id"]
        assert c._method_tasks == resumed._method_tasks
        forbidden = {task_fingerprint(train.task), task_fingerprint(holdout.task)}
        assert not forbidden.intersection(task_fingerprint(t.task) for t in c._method_tasks.values())


def test_gameworld_validation_excludes_external_identity_despite_seed_changes(tmp_path):
    options = {"train": [{"game": "example", "task": "outer"},
                         {"game": "example", "task": "validation"}]}
    task = _make_task(phase="train", game="gameworld", difficulty=1, index=0,
                     group_index=0, used=set(), seed_root=42, options=options)
    config = BenchmarkConfig(seed=42, schedule=(), holdout=(), agent=AgentConfig(provider="openai"),
                             game_options={"gameworld": options})
    c = AgentController(config, tmp_path, 1, reserved_tasks=(task,))
    c.request_for_agent = lambda agent: None
    with patch("benchmark.runner._evaluate", return_value=EvaluationOutcome(rollout_result()["record"], feedback())):
        c._method_rollout(task, None, label="a", split="validation", index=0, output_dir=tmp_path / "a")
        assert c._method_tasks[("gameworld", 1, 0)].task["game_task_id"] == "validation"
        with pytest.raises(RuntimeError, match="additional GameWorld train tasks"):
            c._method_rollout(task, None, label="b", split="validation", index=1, output_dir=tmp_path / "b")


def test_failed_native_request_is_recorded_as_unknown_usage(tmp_path):
    obj = adapter(tmp_path, "skillopt")
    obj.bind_evaluator(lambda *a, **kw: rollout_result())
    obj.start_refinement([feedback()], tmp_path / "round")
    obj.dispatcher.distribute = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("provider failed"))
    with pytest.raises(RuntimeError, match="provider failed"):
        obj.ask_text("prompt", stage="analyst")
    usage = obj.last_refinement_report["token_usage_report"]
    assert usage["no_model_calls"] is False
    assert usage["complete"] is False
    assert usage["total_tokens"] is None


def test_skillopt_rejected_patch_preserves_optimizer_history(tmp_path):
    obj = adapter(tmp_path, "skillopt", validation_trials=1)
    obj.ask_text = lambda *a, **kw: json.dumps({"patch": {"edits": [
        {"op": "append", "content": "Verify prerequisites."}]}})
    obj.bind_evaluator(lambda *a, **kw: rollout_result())
    before = obj._state["best_skill"]
    obj.refine([feedback()], output_dir=tmp_path / "round")
    assert obj._state["best_skill"] == before
    assert obj._state["step"] == 1
    assert obj._state["step_buffer"][0]["accepted"] is False
    assert obj._state["rejected_edits"]
    assert not obj.last_selection_report["accepted"]


def test_artifact_workspaces_are_isolated_between_rollouts(tmp_path):
    obj = adapter(tmp_path, "skillevolver")
    first, second = obj.load(), obj.load()
    assert first.root != second.root
    assert first.root.is_dir() and second.root.is_dir()


def test_skillopt_success_failure_merge_and_rank_are_real_stages(tmp_path):
    obj = adapter(tmp_path, "skillopt", max_edits=1, minibatch_size=1, validation_trials=1)
    calls = []
    def ask(prompt, *, stage):
        calls.append(stage)
        if stage == "ranking":
            return json.dumps({"selected_indices": [1]})
        edits = [{"op": "append", "content": "First lesson."},
                 {"op": "append", "content": "Ranked transferable lesson."}]
        patch = {"edits": edits[:1] if stage == "analyst" else edits}
        return json.dumps({"patch": patch} if stage == "analyst" else patch)
    obj.ask_text = ask
    obj.bind_evaluator(lambda *a, **kw: rollout_result(
        .8 if kw["label"].startswith("candidate") else .5,
        kw["label"].startswith("candidate")))
    success = replace(feedback(), evaluation=replace(feedback().evaluation, success=True))
    obj.refine([feedback(), success], output_dir=tmp_path / "round")
    assert calls.count("analyst") == 2
    assert "merge" in calls and "ranking" in calls
    raw = json.loads((tmp_path / "round/native_patch.json").read_text())["raw"]
    assert {p["source_type"] for p in raw} == {"success", "failure"}
    assert "Ranked transferable lesson" in obj._state["best_skill"]
    assert "First lesson" not in obj._state["best_skill"]


def test_rrsi_invalid_candidate_trials_cannot_raise_selection_score(tmp_path):
    obj = adapter(tmp_path, "rrsi", trials=1, candidate_count=1, repair_rounds=0)
    def ask(prompt, *, stage):
        if stage == "analyst": return {}
        if stage.startswith("critic"): return {"accept": True}
        state = deepcopy(obj._state)
        state["harness"]["prompt"] = "candidate"
        return {"state": state, "edits": [{"component": "prompt", "hypothesis": "improve"}]}
    obj.ask = ask
    def evaluate(*a, **kw):
        result = rollout_result(.9)
        if kw["label"].startswith("candidate"):
            result["record"]["status"] = "error"
        return result
    obj.bind_evaluator(evaluate)
    obj.refine([feedback()], output_dir=tmp_path / "round")
    assert not obj.last_selection_report["accepted"]
    assert obj.last_selection_report["decisions"][0]["reason"] == "eval_invalid"
