"""Fast checks of the step-resume wiring (browser-level checks: tools/check_step_resume.py)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_enabled_by_launcher_env_not_by_config(monkeypatch):
    from benches.gameworld.benchmark import step_resume_enabled
    monkeypatch.delenv("RSI_STEP_RESUME", raising=False)
    assert not step_resume_enabled({"game_id": "03_astray"})
    monkeypatch.setenv("RSI_STEP_RESUME", "1")
    assert step_resume_enabled({"game_id": "03_astray"})
    for game in ("minecraft", "astray"):
        for method in ("ace", "hyperagents", "ahe"):
            config = json.loads((ROOT / f"configs/{game}_{method}.json").read_text())
            # Not in the hashed config: running lineages keep their identity.
            assert config["game_options"]["gameworld"]["resumable_steps"] is False


def test_parent_cell_is_stable_and_candidates_never_share_it():
    from benches.gameworld.benchmark import episode_cell_key
    parent, candidate, other = object(), object(), object()
    key = "train-gameworld-d1-0007-test"
    assert episode_cell_key(key, parent) == key
    assert episode_cell_key(key, parent) == key
    first, second = episode_cell_key(key, candidate), episode_cell_key(key, other)
    assert key != first != second and first.startswith(key + "__candidate_")


def test_both_games_instrument_their_exact_files(tmp_path):
    from benches.gameworld.astray_step_resume import AstrayTransaction
    for game, root, task_id in (("03_astray", "gameworld_train_astray", "03_t01"),
                                ("03_astray", "gameworld", "03_01"),
                                ("18_minecraft-clone-glm", "gameworld_train_minecraft", "18_m02"),
                                ("18_minecraft-clone-glm", "gameworld", "18_01")):
        task = {"task_id": f"t_{task_id}", "game_id": game, "game_task_id": task_id,
                "framework": "ace", "max_steps": 100, "reasoning_effort": "low"}
        a = AstrayTransaction(run_root=tmp_path / task_id, task=task,
                              gameworld_root=ROOT / root, framework="ace")
        b = AstrayTransaction(run_root=tmp_path / task_id, task={**task, "resumable_steps": True},
                              gameworld_root=ROOT / root, framework="ace")
        assert a.store.source_digest == b.store.source_digest
        assert all(b"__gw" in body for body in a._routes.values())


def test_unreceipted_call_is_recorded_as_unknown_cost_before_retry(tmp_path):
    from benches.gameworld.astray_step_resume import AstrayTransaction
    task = {"task_id": "t", "game_id": "03_astray", "game_task_id": "03_t01",
            "framework": "ace", "max_steps": 100, "reasoning_effort": "low"}
    t = AstrayTransaction(run_root=tmp_path, task=task, gameworld_root=ROOT / "gameworld_train_astray",
                          framework="ace")
    t.store.save_genesis(world={"w": 1}, observed_state={"o": 1}, agent_state={"a": 1},
                         ledger={"schema": 1, "total_tokens": 0, "wall_seconds": 0.0})
    t.store.mark_model_call(1, "req1")
    record = t.prepare_retry()
    assert record["usage_status"] == "unknown_may_have_billed"
    assert not (t.root / "inflight.json").exists()
    assert t.prepare_retry() is None
