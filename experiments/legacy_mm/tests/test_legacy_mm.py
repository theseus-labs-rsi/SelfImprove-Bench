import ast
import base64
import hashlib
import json
import shutil
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from benchmark.visual_evidence import frames, sample, bundle_feedback, chat_message, refinement_image_limit
from benchmark.contracts import BackendResponse, TokenUsage
from benchmark.backend import OpenAIBackend
from rsi_methods.ACE.ACEAdaptor import ACEAdaptor, EMPTY_PLAYBOOK
from rsi_methods.ACE.ACEAgent import ACEAgent
from rsi_methods.ACE.ace.ace.core.reflector import Reflector
from rsi_methods.ACE.ace.ace.core.curator import Curator
from rsi_methods.AHE.refine import _prepare_evidence, _prepare_evolve_runtime, _run_evolver, _write_analysis
from rsi_methods.AHE.init import AHEState

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
REF = ROOT / "reference/code"


@pytest.fixture(autouse=True)
def image_limit(monkeypatch):
    monkeypatch.setenv("RSI_MAX_REFINEMENT_IMAGES", "8")


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "frame.png"
    Image.new("RGB", (12, 10), (31, 128, 219)).save(p)
    return p


def feedback(png=None, count=1):
    return {
        "task_id": "test_game_task",
        "task": {"prompt": "Find the exit."},
        "trajectory": [{
            "step_index": i, "prompt": f"Current observation {i}",
            "action": '{"tool_name":"move_forward","arguments":{}}',
            "environment_response": "moved", "score": .4,
            "agent_call": {"response_text": json.dumps({
                "reasoning": f"look {i}", "bullet_ids": ["str-00001"],
                "final_answer": {"tool_name": "move_forward", "arguments": {}},
            })},
            "image_records": [{"path": str(png)}] if png else [],
        } for i in range(count)],
        "evaluation": {"score": .4, "success": False, "execution_status": "completed"},
    }


def function_tree(path, name):
    module = ast.parse(path.read_text())
    return ast.dump(next(n for n in ast.walk(module)
                         if isinstance(n, ast.FunctionDef) and n.name == name),
                    include_attributes=False)


@pytest.mark.parametrize("relative", [
    "benchmark/runner.py", "benchmark/rsi_session.py", "benchmark/resume_state.py",
    "benchmark/metrics.py", "benchmark/tasks.py", "benchmark/config.py",
    "rsi_methods/AHE/select.py", "rsi_methods/AHE/AHEAdaptor.py",
    "rsi_methods/HyperAgents/HyperAgentsAdaptor.py",
    "rsi_methods/HyperAgents/lifecycle.py",
    "rsi_methods/HyperAgents/candidate_registry.py",
])
def test_protected_protocol_bytes_unchanged(relative):
    assert (CODE / relative).read_bytes() == (REF / relative).read_bytes()


@pytest.mark.parametrize("relative,names", [
    ("rsi_methods/ACE/ACEAdaptor.py", ["select", "refine", "_build_agent"]),
    ("rsi_methods/AHE/refine.py", ["refine", "select_parent", "_read_feedback"]),
])
def test_learning_functions_unchanged(relative, names):
    for name in names:
        assert function_tree(CODE / relative, name) == function_tree(REF / relative, name)


def test_snapshot_hashes_are_reproducible():
    expected = json.loads((ROOT / "provenance/baseline_sha256.json").read_text())
    for relative, digest in expected.items():
        assert hashlib.sha256((REF / relative).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("game", ["minecraft", "astray"])
def test_six_configs_old_protocol_with_disjoint_tasks(game):
    from benchmark.config import load_config
    from benchmark.tasks import build_manifest
    train_lists, holdout_lists = [], []
    for method in ("ace", "hyperagents", "ahe"):
        path = ROOT / f"configs/{game}_{method}.json"
        d = json.loads(path.read_text())
        opt = d["game_options"]["gameworld"]
        assert d["agent"]["model"] == "gpt-5.6-sol"
        assert d["agent"]["reasoning_effort"] == opt["reasoning_effort"] == "low"
        assert opt["max_steps"] == 100 and opt["ahe_refinement_every"] == 1
        assert opt["resumable_steps"] is False
        assert "evaluation_panel" not in opt
        train, holdout = opt["train"], opt["holdout"]
        train_lists.append([x["task"] for x in train])
        holdout_lists.append([x["task"] for x in holdout])
        assert len(set(train_lists[-1])) == 20
        assert len(set(holdout_lists[-1])) == 5
        assert not set(train_lists[-1]) & set(holdout_lists[-1])
        task_payloads = []
        for item in train + holdout:
            paths = list((Path(item["gameworld_root"]) / "code/catalog/tasks").glob(f"*/{item['task']}.yaml"))
            assert len(paths) == 1
            payload = yaml.safe_load(paths[0].read_text())
            payload.pop("task_id")
            task_payloads.append(json.dumps(payload, sort_keys=True))
        assert len(set(task_payloads[:20])) == 20, "Duplicate training objective/map"
        manifest = build_manifest(load_config(path))
        assert len(manifest.training) == 20
        assert len(manifest.holdout) == 5
    assert train_lists[0] == train_lists[1] == train_lists[2]
    assert holdout_lists[0] == holdout_lists[1] == holdout_lists[2]


def test_low_configs_all_roles():
    for method, roles in (("ace", ("agent", "reflector", "curator")),
                          ("hyperagents", ("meta_agent", "task_agent"))):
        config = tomllib.loads((ROOT / f"configs/{method}.toml").read_text())
        for role in roles:
            assert config[role]["reasoning_effort"] == "low"
            assert config[role]["model"] == "gpt-5.6-sol"


def test_image_bundle_relocates_and_samples(png, tmp_path):
    original = feedback(png, 100)
    directory = tmp_path / "evidence"
    manifest = bundle_feedback(original, directory)
    assert len(manifest["all_frames"]) == 100
    steps = [x["step_index"] for x in manifest["model_sample"]]
    assert steps == [0, 14, 28, 42, 57, 71, 85, 99]
    assert manifest["model_sample_limit"] == 8
    assert original["trajectory"][0]["image_records"][0]["path"] == str(png)
    relocated = tmp_path / "relocated"
    shutil.copytree(directory, relocated)
    png.unlink()
    for entry in manifest["all_frames"]:
        assert hashlib.sha256((relocated / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    assert all(x["timing"] == "pre_action_observation" for x in manifest["all_frames"])


@pytest.mark.parametrize("value,expected", [("2", 2), ("8", 8), ("16", 16), ("all", 100)])
def test_shared_image_limit_controls_bundle(png, tmp_path, monkeypatch, value, expected):
    monkeypatch.setenv("RSI_MAX_REFINEMENT_IMAGES", value)
    manifest = bundle_feedback(feedback(png, 100), tmp_path / "evidence")
    assert len(manifest["model_sample"]) == expected
    assert manifest["model_sample_limit"] == (value if value == "all" else expected)


def test_missing_image_rejected_before_model(tmp_path):
    with pytest.raises(ValueError, match="Missing"):
        frames(feedback(tmp_path / "missing.png"))
    assert frames(feedback()) == []


class Dispatcher:
    def __init__(self):
        self.requests = []

    def distribute(self, **kwargs):
        # Use the REAL backend encoder to check that transport produces bytes.
        self.requests.append(kwargs)
        payload = OpenAIBackend._build_responses_input(
            kwargs["prompt"], tuple(Path(p) for p in kwargs.get("image_paths", [])))
        kwargs["_wire_payload"] = payload
        if "ACE playbook as reusable guidance" in kwargs["prompt"]:
            text = '{"reasoning":"look","bullet_ids":["str-00001"],"final_answer":{"tool_name":"move_forward"}}'
        else:
            text = '{"reflection":"Inspect the visible goal","bullet_tags":[],"operations":[]}'
        return BackendResponse(text=text, model="offline", usage=TokenUsage(11, 7, 18))


def test_ace_actor_reflector_curator_real_transport(png):
    dispatcher = Dispatcher()
    agent = ACEAgent(dispatcher, "openai", "offline", reasoning_effort="low")
    response = agent.act("Return an action.", [png])
    assert response.metadata["ace"]["bullet_ids"] == ["str-00001"]
    adaptor = ACEAdaptor()
    adaptor._reflector = Reflector(dispatcher, "openai", "offline")
    adaptor._curator = Curator(dispatcher, "openai", "offline")
    adaptor.use_ground_truth = False
    adaptor.use_json_mode = True
    adaptor.reflector_reasoning_effort = adaptor.curator_reasoning_effort = "low"
    adaptor.curator_batch_size = 1
    adaptor.playbook_token_budget = 80000
    record = adaptor._normalize_feedback(feedback(png))
    assert record.bullet_ids == ("str-00001",)
    item, playbook = adaptor._reflect_record(record, EMPTY_PLAYBOOK)
    adaptor._curate_reflections([(record, item.reflection)], playbook, 1, 1)
    assert len(dispatcher.requests) == 3
    for request in dispatcher.requests[:2]:
        assert request["reasoning_effort"] == "low"
        image_part = request["_wire_payload"][0]["content"][1]
        assert base64.b64decode(image_part["image_url"].split(",", 1)[1]) == png.read_bytes()
    assert "pre_action_observation" in dispatcher.requests[1]["prompt"]
    curator = dispatcher.requests[2]
    assert "image_paths" not in curator and isinstance(curator["_wire_payload"], str)
    assert "pre_action_observation" not in curator["prompt"]
    assert item.call_info["visual_inputs"][0]["sha256"] == hashlib.sha256(png.read_bytes()).hexdigest()


def test_text_only_reflection_has_no_new_transport_parameters():
    dispatcher = Dispatcher()
    reflector = Reflector(dispatcher, "openai", "offline")
    reflector.reflect("question", "trace", "answer", None, "feedback", "", use_ground_truth=False)
    assert "image_paths" not in dispatcher.requests[0]
    assert isinstance(dispatcher.requests[0]["_wire_payload"], str)
    assert chat_message("plain", []) == "plain"


@pytest.mark.parametrize("image_limit,expected", [("2", 2), ("8", 8), ("16", 16), ("all", 100)])
def test_ace_image_limit_controls_reflector_only(png, monkeypatch, image_limit, expected):
    """The shared launcher value reaches the Reflector; the Curator stays text-only."""
    monkeypatch.setenv("RSI_MAX_REFINEMENT_IMAGES", image_limit)
    adaptor = ACEAdaptor()
    adaptor.config_path = ROOT / "configs/ace.toml"
    adaptor._configure(adaptor._load_config())
    dispatcher = Dispatcher()
    adaptor._reflector = Reflector(dispatcher, "openai", "offline")
    adaptor._curator = Curator(dispatcher, "openai", "offline")
    record = adaptor._normalize_feedback(feedback(png, 100))
    assert len(record.visual_frames) == expected
    assert record.visual_frames[0]["step_index"] == 0
    assert record.visual_frames[-1]["step_index"] == 99
    # Image sampling must not truncate the textual action/feedback trajectory.
    assert "Step 50\n" in record.reasoning_trace
    assert "Step 99\n" in record.reasoning_trace
    item, playbook = adaptor._reflect_record(record, EMPTY_PLAYBOOK)
    adaptor._curate_reflections([(record, item.reflection)], playbook, 1, 1)
    assert len(dispatcher.requests) == 2
    reflector, curator = dispatcher.requests
    assert len(reflector["image_paths"]) == expected
    assert len(reflector["_wire_payload"][0]["content"]) == expected + 1
    assert "image_paths" not in curator and isinstance(curator["_wire_payload"], str)


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "true", "", "eight"])
def test_invalid_image_limit_rejected(monkeypatch, value):
    monkeypatch.setenv("RSI_MAX_REFINEMENT_IMAGES", value)
    with pytest.raises(ValueError, match="RSI_MAX_REFINEMENT_IMAGES"):
        refinement_image_limit()
    with pytest.raises(ValueError, match="RSI_MAX_REFINEMENT_IMAGES"):
        ACEAdaptor()._configure_refinement({})


def test_ace_config_has_no_separate_image_limit():
    for path in (ROOT / "configs/ace.toml", CODE / "rsi_methods/ACE/config.toml"):
        assert "max_refinement_images" not in tomllib.loads(path.read_text())["refine"]


def test_ace_short_episode_keeps_every_frame(png):
    adaptor = ACEAdaptor()
    adaptor._configure_refinement({})
    assert len(adaptor._normalize_feedback(feedback(png, 5)).visual_frames) == 5


def test_nexau_chat_image_survives_actual_legacy_conversion(png):
    from nexau.core.adapters.legacy import messages_to_legacy_openai_chat
    messages = chat_message("Observe this.", [png])
    payload = messages_to_legacy_openai_chat(messages, tool_image_policy="embed_in_tool_message")
    part = payload[0]["content"][1]
    assert part["type"] == "image_url"
    assert base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) == png.read_bytes()


def test_ahe_evidence_keeps_all_trace_requests(png, tmp_path):
    parent = tmp_path / "workspace"
    parent.mkdir()
    (parent / "policy.txt").write_text("unchanged")
    trace_paths = []
    for i in range(3):
        d = tmp_path / f"call_{i}"
        d.mkdir()
        t = d / "trace.json"
        t.write_text(json.dumps({"turn": i}))
        (d / "request.json").write_text(json.dumps({"prompt": f"turn {i}", "image_paths": [str(png)]}))
        trace_paths.append(t)
    trial, _ = _prepare_evidence(parent, feedback(png, 3), .4, trace_paths[-1],
                                 tmp_path / "runs/iteration_001", trace_paths)
    assert float((trial / "verifier/reward.txt").read_text()) == .4
    paths = json.loads((trial / "agent/all_step_traces.json").read_text())["paths"]
    assert len(paths) == 3
    for i, value in enumerate(paths):
        assert json.loads((trial / value).read_text()) == {"turn": i}
        request = json.loads((trial / value).with_name("request.json").read_text())
        assert (trial / request["image_paths"][0]).read_bytes() == png.read_bytes()


def test_ahe_task_trace_contains_every_step(png, tmp_path):
    """Like the text version: the learner's trace covers the whole task, not step 0."""
    parent = tmp_path / "workspace"
    parent.mkdir()
    trace = tmp_path / "trace.json"
    # Shape produced by the NexAU converter: step 0 input, blank assistant text.
    trace.write_text(json.dumps({
        "name": "Agent: nexau_code_agent", "system_prompt": "sys",
        "input": {"message": "Current observation 0"},
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "Current observation 0"},
                     {"role": "assistant", "content": " "}],
        "messages_count": 3, "total_tokens": 5,
    }))
    iteration = tmp_path / "runs/iteration_001"
    original = feedback(png, 100)
    trial, name = _prepare_evidence(parent, original, .4, trace, iteration)
    rebuilt = json.loads((trial / "agent/nexau_in_memory_tracer.cleaned.json").read_text())
    messages = rebuilt["messages"]
    assert rebuilt["messages_count"] == len(messages) == 201
    assert messages[0] == {"role": "system", "content": "sys"}
    assert rebuilt["total_tokens"] == 5 and rebuilt["name"] == "Agent: nexau_code_agent"
    manifest = json.loads((trial / "visual_manifest.json").read_text())
    for i in range(100):
        user, assistant = messages[1 + 2 * i], messages[2 + 2 * i]
        assert user["role"] == "user" and assistant["role"] == "assistant"
        assert user["step_index"] == assistant["step_index"] == i
        assert user["content"].startswith(f"Current observation {i}")
        image = manifest["all_frames"][i]["path"]
        assert f"[screenshot: {image} " in user["content"]
        assert (trial / image).read_bytes() == png.read_bytes()
        assert assistant["content"] == original["trajectory"][i]["agent_call"]["response_text"]
    assert rebuilt["output"]["response"] == messages[-1]["content"]
    overview = _write_analysis(iteration, trial, name, original, .4)
    assert "complete task trajectory" in overview
    assert "feedback.json" in overview and "visual_manifest.json" in overview


def test_ahe_trace_kept_when_trajectory_empty(tmp_path):
    parent = tmp_path / "workspace"
    parent.mkdir()
    trace = tmp_path / "trace.json"
    trace.write_text('{"messages": []}')
    trial, _ = _prepare_evidence(parent, feedback(None, 0), .4, trace, tmp_path / "runs/iteration_001")
    assert (trial / "agent/nexau_in_memory_tracer.cleaned.json").read_text() == '{"messages": []}'


def test_ahe_runtime_visual_tool_bound_and_not_truncated(tmp_path):
    source = CODE / "rsi_methods/AHE/agentic-harness-engineering/agents/evolve_agent"
    config_path = _prepare_evolve_runtime(source, tmp_path / "runtime")
    config = yaml.safe_load(config_path.read_text())
    assert len([t for t in config["tools"] if t["name"] == "read_visual_file"]) == 1
    middleware = next(m for m in config["middlewares"] if "long_tool_output" in m["import"])
    assert "read_visual_file" in middleware["params"]["bypass_tool_names"]


def test_evolver_subprocess_receives_actual_sample_paths(png, tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("rsi_methods.AHE.refine")
    parent = tmp_path / "workspace"
    parent.mkdir()
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    iteration = tmp_path / "runs/iteration_001"
    _prepare_evidence(parent, feedback(png, 10), .4, trace, iteration)
    assert len(json.loads((next(iteration.glob("input/benchmark/local/*")) / "visual_manifest.json").read_text())["model_sample"]) == 8
    candidate = tmp_path / "candidates/variant_0"
    workspace = candidate / "workspace"
    workspace.mkdir(parents=True)
    def run(argv, **kwargs):
        request = json.loads(Path(argv[-2]).read_text())
        assert request["reasoning_effort"] == "low"
        assert len(request["image_paths"]) == 8
        assert all(Path(p).read_bytes() == png.read_bytes() for p in request["image_paths"])
        assert "pre_action_observation" in request["query"]
        Path(argv[-1]).write_text('{"result":"offline","usage_snapshot":null}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    _run_evolver(AHEState(tmp_path, parent, tmp_path / "evolver"), candidate,
                  iteration, workspace, "Improve", "offline", "low", 60)


def test_hyperagents_meta_agent_uses_shared_image_limit(png, tmp_path, monkeypatch):
    import sys
    from unittest.mock import patch
    from rsi_methods.HyperAgents.archive_utils import _install_visual_evidence_hook
    source = CODE / "rsi_methods/HyperAgents/hyperagents"
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy2(source / "meta_agent.py", repo / "meta_agent.py")
    monkeypatch.setenv("RSI_MAX_REFINEMENT_IMAGES", "3")
    _install_visual_evidence_hook(repo)
    patched = (repo / "meta_agent.py").read_text()
    _install_visual_evidence_hook(repo)
    assert (repo / "meta_agent.py").read_text() == patched
    assert "range(8)" not in patched and "before that step's action" in patched
    assert json.loads((repo / "visual_evidence_limit.json").read_text()) == {"max_refinement_images": 3}
    assert (source / "meta_agent.py").read_text() != patched  # vendored source untouched
    evidence = tmp_path / "evidence"
    (evidence / "shots").mkdir(parents=True)
    for i in range(10):
        shutil.copy2(png, evidence / f"shots/{i}.png")
    (evidence / "result.json").write_text(json.dumps({"cases": [{"feedback": {"trajectory": [
        {"image_records": [{"path": f"shots/{i}.png"}]} for i in range(10)]}}]}))
    monkeypatch.syspath_prepend(str(source))
    import importlib.util
    spec = importlib.util.spec_from_file_location("patched_meta_agent", repo / "meta_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = object.__new__(module.MetaAgent)
    for key, value in dict(model="m", reasoning_effort="low", backend="openai", api_mode="responses",
                           api_key_env="K", base_url_env="U").items():
        setattr(instance, key, value)
    instance.log = lambda *_: None
    instance.logger_manager = SimpleNamespace(log_file=str(tmp_path / "chat.log"))
    with patch.object(module, "chat_with_agent", return_value=[{"text": "seen"}]) as chat:
        assert instance._inspect_game_screenshots(str(evidence)) == "seen"
    sent = [Path(p).name for p in chat.call_args.kwargs["image_paths"]]
    assert sent == ["0.png", "4.png", "9.png"]
    (repo / "visual_evidence_limit.json").write_text('{"max_refinement_images": null}')
    with patch.object(module, "chat_with_agent", return_value=[{"text": "seen"}]) as chat:
        instance._inspect_game_screenshots(str(evidence))
    assert len(chat.call_args.kwargs["image_paths"]) == 10


def test_all_methods_receive_identical_actor_input(png, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "gameworld_train_astray/code"))
    from benches.gameworld.ace_game_client import ACEGameClient
    from benches.gameworld.ahe_game_client import AHEGameClient
    from benches.gameworld.hyperagents_game_client import HyperAgentsGameClient
    memory_image = tmp_path / "memory.png"
    shutil.copy(png, memory_image)
    memory = [SimpleNamespace(type="text", text="step 1: move_forward"),
              SimpleNamespace(type="image", image_file=lambda: memory_image)]
    sent = {}

    class Agent:
        def __init__(self, name):
            self.name = name

        def act(self, prompt, image_paths):
            sent[self.name] = (prompt, list(image_paths))
            return BackendResponse(text='{"tool_name":"move_forward","arguments":{}}')

    for name, cls in (("ace", ACEGameClient), ("ahe", AHEGameClient), ("hyper", HyperAgentsGameClient)):
        client = object.__new__(cls)
        client.config = SimpleNamespace(system_prompt="Play the game.")
        client._action_tool_names = {"move_forward"}
        client._collect_memory_context = lambda: memory
        client._format_memory_text_entry = lambda entry, as_action_history: entry.text
        client._complete_action = lambda **kwargs: kwargs
        setattr(client, {"ace": "_ace_agent", "ahe": "_ahe_agent", "hyper": "_runtime_agent"}[name], Agent(name))
        result = client.get_action(png)
        assert result["tool_call"] == {"tool_name": "move_forward", "arguments": {}}
        assert result["raw_message_sent"] == sent[name][0]
    assert sent["ace"] == sent["ahe"] == sent["hyper"]
    prompt, images = sent["ace"]
    assert images == [png]
    assert "step 1: move_forward" in prompt and "# Action Transport" in prompt
    assert all(m not in prompt for m in ("ACE", "AHE", "HyperAgents"))


def test_ahe_request_keeps_screenshot_after_browser_temp_removed(png, tmp_path):
    from rsi_methods.AHE.agent import AHEAgent
    temp = tmp_path / ".screenshots_temp" / "frame_000000_agent_0.png"
    temp.parent.mkdir()
    shutil.copy(png, temp)
    agent = object.__new__(AHEAgent)
    agent.artifact_dir = tmp_path / "calls"
    agent.artifact_dir.mkdir()
    agent.timeout_seconds = 1
    agent._unreconciled_failed_attempts = False
    agent._connection = SimpleNamespace(send=lambda message: None)
    agent._record_attempt = lambda *args: None

    def stop(timeout):
        raise ConnectionError("stop after request is recorded")

    agent._receive = stop
    with pytest.raises(ConnectionError):
        agent.act("Game screen:", image_paths=[temp])
    shutil.rmtree(temp.parent)  # GameWorld removes it when the browser closes
    request = json.loads(next(agent.artifact_dir.glob("*/request.json")).read_text())
    assert request["source_image_paths"] == [str(temp.resolve())]
    archived = Path(request["image_paths"][0])
    assert archived.is_file() and archived.read_bytes() == png.read_bytes()


def test_minecraft_training_tasks_mirror_holdout_quota_only():
    # Each training task is an official holdout task with only the quota changed by 1-2;
    # same official world (no URL suffix, same game files) and step budget.
    import random
    d = json.loads((ROOT / "configs/minecraft_ace.json").read_text())
    tasks = d["game_options"]["gameworld"]["train"]
    official = ROOT / "gameworld/code/catalog/tasks/18_minecraft-clone-glm"
    game = "code/games/benchmark/18_minecraft-clone-glm"
    for name in ("game_api.js", "index.html", "main.js"):
        assert (ROOT / "gameworld_train_minecraft" / game / name).read_bytes() == (ROOT / "gameworld" / game / name).read_bytes()
    names = [item["task"] for item in tasks]
    assert len(set(names)) == 20 and names != sorted(names)
    sources = []
    for item in tasks:
        payload = yaml.safe_load(next((Path(item["gameworld_root"]) / "code/catalog/tasks").glob(
            f"*/{item['task']}.yaml")).read_text())
        assert "game_url_suffix" not in payload
        matches = [yaml.safe_load(s.read_text()) for s in (official / f"18_0{i}.yaml" for i in range(1, 6))
                   if yaml.safe_load(s.read_text())["evaluator_config"] == payload["evaluator_config"]]
        assert len(matches) == 1
        ref = matches[0]
        sources.append(ref["task_id"])
        assert 1 <= abs(payload["task_target_score_field"] - ref["task_target_score_field"]) <= 2
        assert payload["max_steps"] == ref["max_steps"] == 100
        assert payload["task_prompt"] == ref["task_prompt"].replace(
            str(ref["task_target_score_field"]), str(payload["task_target_score_field"]), 1)
        rest = {k: v for k, v in payload.items() if k not in ("task_id", "task_target_score_field", "task_prompt")}
        assert rest == {k: v for k, v in ref.items() if k not in ("task_id", "task_target_score_field", "task_prompt")}
    assert sorted(set(sources)) == ["18_01", "18_02", "18_03", "18_04", "18_05"]
    assert all(sources.count(x) == 4 for x in set(sources))
    for method in ("hyperagents", "ahe"):
        other = json.loads((ROOT / f"configs/minecraft_{method}.json").read_text())
        assert [x["task"] for x in other["game_options"]["gameworld"]["train"]] == names
