import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.contracts import BackendResponse, ExecutionStatus, TokenUsage
from benchmark.rsi_session import refinement_token_metrics
from benches.gameworld.benchmark import GameWorldBenchmark
from rsi_methods.ACE.ace.llm import timed_llm_call
from rsi_methods.AHE.agent import AHEAgent


@pytest.mark.parametrize("status,progress,valid", [
    ("error", .6, False), ("unknown", .4, False),
    ("success", None, False), ("success", float("nan"), False),
    ("fail", "malformed", False), ("fail", True, False),
    ("fail", 0, True), ("success", 1, True),
])
def test_evaluator_validity(tmp_path, status, progress, valid):
    p = tmp_path / "agent_0/evaluation/summary.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"finalized": True, "task_status": status, "progress": progress}))
    result = GameWorldBenchmark._feedback({"task_id": "probe"}, tmp_path)
    assert (result.evaluation.execution_status == ExecutionStatus.COMPLETED) is valid
    if not valid:
        assert result.evaluation.score is None


def test_retry_cost_remains_unknown(tmp_path):
    class Fake:
        calls = 0

        def distribute(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("timed out")
            return BackendResponse(
                text="ok", usage=TokenUsage(input_tokens=3, output_tokens=4, total_tokens=7),
                model="fixture", provider="fixture",
            )

    fake = Fake()
    _, info = timed_llm_call(fake, "fake", "fixture", "p", "reflector", "probe",
                            retries_on_timeout=2, sleep_seconds=0, log_dir=tmp_path)
    assert fake.calls == 2
    assert info["total_num_tokens"] is None
    assert info["known_successful_response_tokens"] == 7
    assert len(info["attempt_ledger"]) == 2
    assert info["attempt_ledger"][0]["unknown_may_have_billed"]
    assert refinement_token_metrics({"call_info": info})["complete"] is False
    assert (tmp_path / "request_attempts.jsonl").is_file()


def test_ahe_request_images_survive_observation_cleanup(tmp_path):
    from PIL import Image
    image = tmp_path / "temporary.png"
    Image.new("RGB", (8, 8), "green").save(image)
    original = image.read_bytes()
    agent = AHEAgent.__new__(AHEAgent)
    agent.artifact_dir = tmp_path / "calls"
    agent.artifact_dir.mkdir()
    requests = []
    agent._connection = SimpleNamespace(send=requests.append)
    agent._record_attempt = lambda *a, **kw: None

    def stop_after_send(*args):
        raise TimeoutError("offline fixture, no model launched")

    agent._receive = stop_after_send
    agent.timeout_seconds = 1
    with pytest.raises(TimeoutError):
        agent.act("look", image_paths=[image])
    image.unlink()
    request = requests[0]
    saved = json.loads((Path(request["call_dir"]) / "request.json").read_text())
    assert saved["image_paths"] == request["image_paths"]
    assert Path(saved["image_paths"][0]).read_bytes() == original


def test_ahe_candidate_actor_uses_configured_timeout(tmp_path, monkeypatch):
    # refine.generate_candidates calls build() without timeout_seconds.
    from rsi_methods.AHE import build as build_module
    seen = {}
    monkeypatch.setattr(build_module, "AHEAgent", lambda *a, **k: seen.update(k))
    workspace = tmp_path / "run" / "ws"
    workspace.mkdir(parents=True)
    (workspace / "code_agent.yaml").write_text("x: 1\n")
    state = type("S", (), {"run_dir": (tmp_path / "run").resolve(), "current_workspace": workspace})()
    monkeypatch.setenv("RSI_AHE_ACTOR_TIMEOUT", "600.0")
    monkeypatch.delenv("RSI_AHE_STEP_TIMEOUT", raising=False)
    build_module.build(state)
    assert seen["timeout_seconds"] == 600.0
    build_module.build(state, timeout_seconds=42)
    assert seen["timeout_seconds"] == 42
    # Launcher cap applies to parent (explicit) and candidate (default) actors.
    monkeypatch.setenv("RSI_AHE_STEP_TIMEOUT", "300")
    build_module.build(state)
    assert seen["timeout_seconds"] == 300.0
    build_module.build(state, timeout_seconds=600)
    assert seen["timeout_seconds"] == 300.0
    build_module.build(state, timeout_seconds=42)
    assert seen["timeout_seconds"] == 42


def test_openai_backend_transport_bounds(monkeypatch):
    from benchmark.backend.OpenAIBackend import OpenAIBackend
    monkeypatch.delenv("RSI_OPENAI_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("RSI_OPENAI_MAX_RETRIES", raising=False)
    default = OpenAIBackend(api_key="x")._client
    assert default.max_retries == 2 and default.timeout.read == 600
    monkeypatch.setenv("RSI_OPENAI_REQUEST_TIMEOUT", "150")
    monkeypatch.setenv("RSI_OPENAI_MAX_RETRIES", "0")
    bounded = OpenAIBackend(api_key="x")._client
    assert bounded.max_retries == 0 and bounded.timeout == 150.0


def test_ahe_worker_bounds_each_model_request():
    # A stalled gateway stream must fail inside NexAU's retry loop, well before
    # the actor's per-step timeout ends the whole episode.
    source = (Path(__file__).resolve().parents[1] / "code/rsi_methods/AHE/agent.py").read_text()
    assert 'config.llm_config.timeout = float(os.environ.get("RSI_AHE_REQUEST_TIMEOUT", 55))' in source
    # The OpenAI client's own retries multiply the per-request bound (default 2 extra tries).
    assert "config.llm_config.max_retries = 1" in source
    # The timeout bounds each read only; streaming would let a trickling response outlive it.
    assert "config.llm_config.stream = False" in source


def test_hyperagents_cleanup_removes_one_off_run_image():
    # Leftover "<repo>:run-<id>" images filled Docker's disk mid-run.
    from types import SimpleNamespace
    from rsi_methods.HyperAgents.hyperagents.utils.docker_utils import cleanup_container
    removed = []
    images = SimpleNamespace(remove=lambda image_id, force: removed.append(image_id))
    def container(tags):
        return SimpleNamespace(name="c", stop=lambda timeout: None, remove=lambda force: None,
                               image=SimpleNamespace(id=f"sha-{len(tags)}", tags=tags),
                               client=SimpleNamespace(images=images))
    cleanup_container(container(["hyperagents:run-abc"]), verbose=False)
    cleanup_container(container(["hyperagents-runtime:py312-v2", "x"]), verbose=False)
    assert removed == ["sha-1"]


def test_gateway_524_is_retried_also_when_wrapped_by_hyperagents_container():
    from benches.gameworld.network_retry import is_transient_connection_error
    class APIStatus(Exception):
        status_code = 524
    assert is_transient_connection_error(APIStatus())
    wrapped = RuntimeError("TaskAgent failed with exit code 1: Traceback ...\n"
                           "openai.InternalServerError: Error code: 524 - {'title': 'Error 524: A timeout occurred'}")
    assert is_transient_connection_error(wrapped)
    assert not is_transient_connection_error(RuntimeError("openai.BadRequestError: Error code: 400"))
