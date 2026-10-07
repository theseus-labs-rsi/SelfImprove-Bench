import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.resume_state import digest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("legacy_mm_launch", ROOT / "tools/launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def committed(out, *, score=.5, complete=True, steps=1):
    image = out / "frame.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"offline-image-existence-fixture")
    record = {
        "run_id": "mock_task", "status": "ok", "relative_score": score,
        "selection": {"status": "unchanged"},
        "interactions": [{"step_index": i, "image_records": [{"path": str(image)}]}
                         for i in range(steps)],
        "token_coverage": {"complete": complete},
        "total_tokens": 18 if complete else None,
        "total_seconds": 1.5,
    }
    initial = {"config_hash": "mock_config", "state": {}}
    cp = {"config_hash": "mock_config", "phase": "train", "ordinal": 1,
          "run_id": "mock_task", "parent_digest": digest(initial),
          "record_digest": digest(record)}
    launch.write_json(out / "checkpoints/initial.json", initial)
    launch.write_json(out / "checkpoints/train_0001.json", cp)
    launch.write_json(out / "runs/train/0001_mock_task.json", record)
    return record, cp


def test_empty_status_does_not_create_run(tmp_path):
    out = tmp_path / "nonexistent"
    state = launch.audit(out)
    assert state["training_checkpoints"] == 0
    assert state["errors"] == [] and not state["inflight"]
    assert not out.exists()


def test_unknown_usage_warns_without_fabricated_zero_or_stopping(tmp_path):
    committed(tmp_path, complete=False)
    state = launch.audit(tmp_path)
    assert not state["errors"]
    assert len(state["warnings"]) == 1
    assert state["checks"][0]["total_tokens"] is None


@pytest.mark.parametrize("score", [True, float("nan"), float("inf"), None])
def test_invalid_score_is_not_accepted(tmp_path, score):
    committed(tmp_path, score=score)
    assert any("score/status" in error for error in launch.audit(tmp_path)["errors"])


def test_audit_detects_checkpoint_tampering_and_missing_images(tmp_path):
    _, cp = committed(tmp_path)
    cp["parent_digest"] = "wrong"
    launch.write_json(tmp_path / "checkpoints/train_0001.json", cp)
    (tmp_path / "frame.png").unlink()
    errors = launch.audit(tmp_path)["errors"]
    assert any("chain" in x for x in errors)
    assert any("Missing step image" in x for x in errors)


def test_audit_detects_record_tampering(tmp_path):
    record, _ = committed(tmp_path)
    record["relative_score"] = .9
    launch.write_json(tmp_path / "runs/train/0001_mock_task.json", record)
    assert any("digest" in x for x in launch.audit(tmp_path)["errors"])


def test_audit_rejects_more_than_100_interactions(tmp_path):
    committed(tmp_path, steps=101)
    assert any("interaction count" in x for x in launch.audit(tmp_path)["errors"])


def test_environment_is_own_config_with_low_roles(monkeypatch, tmp_path):
    monkeypatch.setenv("RSI_KEY_FILE", str(tmp_path / "no-key-file"))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")
    launch.configure_environment()
    import os
    assert os.environ["LLM_API_KEY"] == "test"
    assert os.environ["LLM_BASE_URL"] == "https://offline.invalid/v1"
    assert os.environ["RSI_ACE_CONFIG_PATH"] == str(ROOT / "configs/ace.toml")
    assert os.environ["RSI_HYPERAGENTS_CONFIG_PATH"] == str(ROOT / "configs/hyperagents.toml")


@pytest.mark.parametrize("command,extra,error", [
    ("train", [], "require --allow-paid"),
    ("holdout", ["--allow-paid"], "20-round agent"),
    ("train", ["--allow-paid", "--rounds", "0"], "must be positive"),
])
def test_paid_stage_guards_no_worker_called(tmp_path, monkeypatch, capsys, command, extra, error):
    monkeypatch.setattr(launch, "paths", lambda *_: (SimpleNamespace(), tmp_path))
    monkeypatch.setattr(launch, "verify_release", lambda: None)
    monkeypatch.setattr(launch, "worker", lambda *a: pytest.fail("Worker must not run"))
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: pytest.fail("No subprocess"))
    monkeypatch.setattr(launch.sys, "argv", ["launch", command, "minecraft", "ahe", *extra])
    with pytest.raises(SystemExit) as exc:
        launch.main()
    assert exc.value.code == 2
    assert error in capsys.readouterr().err


def test_inflight_requires_explicit_round_retry(tmp_path, monkeypatch, capsys):
    (tmp_path / "inflight.json").write_text("{}")
    monkeypatch.setattr(launch, "paths", lambda *_: (SimpleNamespace(), tmp_path))
    monkeypatch.setattr(launch, "verify_release", lambda: None)
    monkeypatch.setattr(launch.sys, "argv",
                        ["launch", "train", "astray", "ace", "--allow-paid"])
    with pytest.raises(SystemExit):
        launch.main()
    assert "--retry-inflight" in capsys.readouterr().err


def test_worker_uses_legacy_single_round_checkpoint_boundary(tmp_path, monkeypatch):
    from benchmark import runner
    calls = []
    def fake(cfg, **kwargs):
        calls.append(kwargs)
        return {"summary": {"training_errors": 0, "holdout_errors": 0}}
    monkeypatch.setattr(runner, "run_benchmark", fake)
    launch.worker(None, tmp_path, "train")
    launch.write_json(tmp_path / "checkpoints/initial.json", {})
    launch.worker(None, tmp_path, "holdout")
    assert calls == [
        {"resume": False, "max_new_train_runs": 1},
        {"resume": True, "max_new_train_runs": None},
    ]


def test_seal_detects_modified_file(tmp_path, monkeypatch):
    import hashlib
    (tmp_path / "code.py").write_text("original")
    launch.write_json(tmp_path / "provenance/release_sha256.json",
                      {"code.py": hashlib.sha256(b"original").hexdigest()})
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    launch.verify_release()
    (tmp_path / "code.py").write_text("modified")
    with pytest.raises(RuntimeError, match="Release file changed"):
        launch.verify_release()


def test_hyperagents_workers_use_project_docker_daemon(tmp_path, monkeypatch):
    # Every worker re-runs main(), so later rounds of a running launcher pick this up.
    import socket, sys
    sock = tmp_path / "d.sock"
    server = socket.socket(socket.AF_UNIX); server.bind(str(sock))
    monkeypatch.setenv("RSI_HYPERAGENTS_DOCKER_SOCK", str(sock))
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(sys, "argv", ["launch.py", "status", "astray", "hyperagents"])
    monkeypatch.setattr(launch, "paths", lambda *a: (None, tmp_path))
    monkeypatch.setattr(launch, "audit", lambda out: {})
    launch.main()
    assert launch.os.environ["DOCKER_HOST"] == f"unix://{sock}"
    assert launch.os.environ["HYPERAGENTS_DOCKER_NETWORK"] == "host"
    server.close()


def test_worker_rereads_rotated_key_file(tmp_path, monkeypatch):
    # Each round runs in a fresh worker that calls configure_environment().
    key_file = tmp_path / "keys.env"
    key_file.write_text("export OPENAI_API_KEY='sk-new-test'\nexport OPENAI_BASE_URL='https://example.invalid/v1'\n")
    monkeypatch.setenv("RSI_KEY_FILE", str(key_file))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-old-test")
    monkeypatch.setenv("RSI_LOCAL_TMPDIR", str(tmp_path / "tmp"))
    launch.configure_environment()
    assert launch.os.environ["OPENAI_API_KEY"] == launch.os.environ["LLM_API_KEY"] == "sk-new-test"
    assert launch.os.environ["LLM_BASE_URL"] == "https://example.invalid/v1"
