"""Archive acceptance and next-parent sampling are independent decisions."""

import json

import pytest

from analyze.reader import _build_hyper_node


def hyper_node(tmp_path, *, selected="older", accepted=True, archived=True, evaluation=False):
    task = tmp_path / "rsi/hyperagents/run/task_0001"
    candidate_dir = task / "candidates/new"
    candidate_dir.mkdir(parents=True)
    selection = {"selected_candidate_id": selected, "accepted": accepted,
                 "parent_score": 0.8, "candidate_score": 0.2}
    if archived:
        selection.update(archived_candidate_id="new", next_parent_id=selected)
    if evaluation:
        (candidate_dir / "evaluation").mkdir()
        (candidate_dir / "evaluation/result.json").write_text(json.dumps({
            "status": "ok", "relative_score": 0.3, "parent_run_id": "original",
        }))
        (tmp_path / "runs/train").mkdir(parents=True)
        (tmp_path / "runs/train/0001_original.json").write_text(json.dumps({
            "status": "ok", "relative_score": 0.9,
        }))
    return _build_hyper_node(task, "run", {
        "candidate_id": "new", "parent_id": "initial", "changed": True,
    }, selection)


def test_accepted_child_not_sampled_as_next_parent_keeps_its_score(tmp_path):
    node = hyper_node(tmp_path)
    assert node["accepted"] is True
    assert node["selected"] is False
    assert node["next_parent_id"] == "older"
    assert node["current_score"] == 0.2
    assert node["score_delta"] == pytest.approx(-0.6)


def test_current_runner_evaluation_location_is_authoritative(tmp_path):
    node = hyper_node(tmp_path, evaluation=True)
    assert node["parent_score"] == 0.9
    assert node["current_score"] == 0.3
    assert node["score_delta"] == pytest.approx(-0.6)


def test_invalid_archive_child_is_not_accepted(tmp_path):
    assert hyper_node(tmp_path, accepted=False)["accepted"] is False


def test_legacy_single_lineage_selection_remains_supported(tmp_path):
    node = hyper_node(tmp_path, selected="new", archived=False)
    assert node["accepted"] is True
    assert node["selected"] is True
    assert node["current_score"] == 0.2


def hyper_source(node_id, *, artifact):
    return {
        "node_id": node_id, "parent_id": None, "task_id": "task_0001", "run_id": "run",
        "selected": False, "accepted": True, "changed": True,
        "artifact": artifact, "artifact_path": None,
        "parent_score": None, "current_score": None, "score_delta": None,
    }


def test_missing_patch_marks_hyper_analysis_incomplete(tmp_path):
    from unittest.mock import patch

    from analyze.hyper import run_hyper_analysis

    with patch("analyze.hyper.collect_hyperagent_artifacts",
               return_value=[hyper_source("new", artifact=None)]), \
            patch("analyze.hyper.analyze_artifact", side_effect=AssertionError("No model allowed")):
        state = run_hyper_analysis(tmp_path, tmp_path / "analysis.json")
    assert state["status"] == "incomplete"
    assert state["missing_artifact_count"] == 1
    assert json.loads((tmp_path / "analysis.json").read_text())["status"] == "incomplete"


def test_incomplete_hyper_cli_returns_failure(tmp_path, monkeypatch):
    import sys
    from unittest.mock import patch

    from analyze.__main__ import main as analyze_main

    monkeypatch.setattr(sys, "argv", ["analyze", "hyper", str(tmp_path), "--model", "offline"])
    with patch("analyze.__main__.run_hyper_analysis", return_value={
        "status": "incomplete", "missing_artifact_count": 1, "nodes": [],
    }), pytest.raises(SystemExit, match="1 missing artifacts"):
        analyze_main()
