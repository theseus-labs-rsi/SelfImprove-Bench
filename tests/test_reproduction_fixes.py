"""Offline regressions for replay baselines and trajectory artifact inheritance."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from analyze.ace import run_ace_analysis
from analyze.__main__ import main as analyze_main
from analyze.ahe import run_ahe_analysis
from analyze.llm import AnalysisOutput, ImprovementOpinion
from analyze.reader import collect_ace_artifacts, collect_ahe_artifacts
from benchmark.contracts import EvaluationOutcome
from benchmark.metrics import calculate_metrics, post_refinement_score
from benchmark.runner import _run_one
from benchmark.tasks import PlannedTask
from benches.farm import game as farm_game
from benches.combat import game as combat_game


class GameActionContractTest(unittest.TestCase):
    def test_farm_generated_prompt_matches_json_operation_order(self):
        task = farm_game.generate(1, seed=42, task_id="farm-order")
        self.assertNotIn("daily_operation_order", task["rules"])
        prompt = farm_game.build_prompt(task)
        self.assertNotIn("daily_operation_order", prompt)
        self.assertIn("operations execute in the order their keys appear", prompt)
        self.assertIn("There is no fixed operation order", prompt)

    def test_prompts_exclude_reference_answers(self):
        for game, hidden in (
            (farm_game, ("reference_daily_profits", "reference_profit", "reference_method")),
            (combat_game, ("reference_score", "reference_method", "solvability")),
        ):
            with self.subTest(game=game.GAME_NAME):
                task = {"task_id": "visible-task", **dict.fromkeys(hidden, "hidden-reference-answer")}
                prompt = game.build_prompt(task)
                self.assertIn("visible-task", prompt)
                self.assertNotIn("hidden-reference-answer", prompt)

    def test_farm_validates_task_identity_and_plan_shape(self):
        task = {"task_id": "farm-test"}
        action = {"task_id": "farm-test", "days": []}
        self.assertIs(farm_game.validate_action(action, task), action)
        for invalid in ({"task_id": "other", "days": []}, {"days": {}}):
            with self.subTest(action=invalid), self.assertRaises(ValueError):
                farm_game.validate_action(invalid, task)

    def test_combat_validates_stage_identity_and_decision_shape(self):
        for decision_type, action, invalid in (
            ("combat", {"turns": []}, {"turns": {}}),
            ("shop", {"buy_cards": [], "heal": False}, {"buy_cards": [], "heal": "false"}),
        ):
            stage = {"task_id": "combat-test", "stage_id": "current", "decision_type": decision_type}
            valid = {"task_id": "combat-test", "stage_id": "current", **action}
            with self.subTest(decision_type=decision_type):
                self.assertIs(combat_game.validate_action(valid, stage), valid)
                for bad in (
                    {**valid, "task_id": "other"},
                    {**valid, "stage_id": "other"},
                    {**valid, **invalid},
                ):
                    with self.subTest(action=bad), self.assertRaises(ValueError):
                        combat_game.validate_action(bad, stage)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def selected_record(score=0.8):
    return {
        "run_id": "original", "relative_score": 0.2, "status": "ok",
        "selection": {
            "status": "committed", "committed_candidate_id": "winner",
            "stages": [{
                "status": "committed", "committed_candidate_id": "winner",
                "evaluations": [
                    {"candidate_id": "rejected", "status": "ok", "relative_score": 0.95},
                    {"candidate_id": "winner", "status": "ok", "relative_score": score},
                ],
            }],
        },
    }


class ReplayBaselineTest(unittest.TestCase):
    def test_pre_score_not_committed_score_or_maximum(self):
        original = selected_record()
        replay = {"run_id": "replay", "replay": True, "replay_of": "original",
                  "replay_gap": 3, "relative_score": 0.7, "status": "ok"}
        retention = calculate_metrics([original, replay], [], evolution_window=1)["capability_retention"]
        self.assertAlmostEqual(retention["average_score_change"], 0.5)
        self.assertEqual(retention["baseline"], "pre_refinement_execution_score")
        self.assertEqual(retention["pairs"][0]["baseline_score"], 0.2)
        self.assertEqual(retention["pairs"][0]["replay_score"], 0.7)

    def test_rejected_candidate_retains_parent(self):
        record = selected_record()
        record["selection"].update(status="unchanged", committed_candidate_id=None)
        self.assertEqual(post_refinement_score(record), 0.2)

    def test_zero_is_a_valid_committed_score(self):
        self.assertEqual(post_refinement_score(selected_record(0.0)), 0.0)

    def test_no_refinement_retains_execution_score(self):
        self.assertEqual(post_refinement_score({"relative_score": 0.3, "refinement": None,
                                               "selection": None}), 0.3)

    def test_explicit_post_score_and_explicit_missing_score(self):
        record = selected_record()
        record["post_refinement_score"] = 0.6
        self.assertEqual(post_refinement_score(record), 0.6)
        record["post_refinement_score"] = None
        self.assertIsNone(post_refinement_score(record))

    def test_invalid_or_missing_evidence_is_not_pre_score(self):
        for score in (None, True, float("nan"), float("inf")):
            with self.subTest(score=score):
                self.assertIsNone(post_refinement_score(selected_record(score)))
        self.assertIsNone(post_refinement_score({"relative_score": 0.2}))
        self.assertIsNone(post_refinement_score({"relative_score": 0.2, "refinement": {}}))
        record = selected_record()
        record["selection"]["stages"][-1]["evaluations"] = []
        self.assertIsNone(post_refinement_score(record))
        record["selection"]["status"] = "error"
        self.assertIsNone(post_refinement_score(record))

    def test_only_final_committed_stage_is_used(self):
        record = selected_record()
        earlier = {"status": "needs_validation", "committed_candidate_id": "winner",
                   "evaluations": [{"candidate_id": "winner", "status": "ok", "relative_score": 0.99}]}
        record["selection"]["stages"].insert(0, earlier)
        self.assertEqual(post_refinement_score(record), 0.8)

    def test_missing_post_score_does_not_prevent_retention(self):
        record = selected_record()
        record["selection"]["status"] = "error"
        replay = {"run_id": "replay", "replay": True, "replay_of": "original",
                  "replay_gap": 3, "relative_score": 0.7, "status": "ok"}
        retention = calculate_metrics([record, replay], [], evolution_window=1)["capability_retention"]
        self.assertEqual(retention["replay_pair_count"], 1)
        self.assertEqual(retention["unavailable_pair_count"], 0)
        self.assertAlmostEqual(retention["average_score_change"], 0.5)

    def test_missing_execution_score_is_reported(self):
        record = selected_record()
        record["relative_score"] = None
        replay = {"run_id": "replay", "replay": True, "replay_of": "original",
                  "replay_gap": 3, "relative_score": 0.7, "status": "ok"}
        retention = calculate_metrics([record, replay], [], evolution_window=1)["capability_retention"]
        self.assertEqual(retention["replay_pair_count"], 0)
        self.assertEqual(retention["unavailable_pair_count"], 1)
        self.assertEqual(retention["unavailable_pairs"][0]["reason"], "execution_score_unavailable")
        self.assertIsNone(retention["average_score_change"])

    def test_failed_original_or_replay_counts_as_zero(self):
        for original_status, replay_status, expected in (
            ("error", "ok", 0.7), ("skipped", "ok", 0.7),
            ("ok", "error", -0.2), ("ok", "skipped", -0.2),
        ):
            with self.subTest(original=original_status, replay=replay_status):
                record = selected_record()
                record["status"] = original_status
                replay = {"run_id": "replay", "replay": True, "replay_of": "original",
                          "replay_gap": 3, "relative_score": 0.7, "status": replay_status}
                retention = calculate_metrics([record, replay], [], evolution_window=1)["capability_retention"]
                self.assertAlmostEqual(retention["average_score_change"], expected)

    def test_runner_persists_post_score_without_changing_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = MagicMock()
            controller.provider = "ace"
            controller.session.refinement_telemetry = {}
            controller.session.pending_batch = SimpleNamespace(
                candidates=[SimpleNamespace(candidate_id="winner")])
            outcome = EvaluationOutcome({"status": "ok", "game": "farm", "difficulty": 3,
                                         "relative_score": 0.2, "llm_metrics": {}})
            task = PlannedTask("ordinary", "train", "farm", 3, {"task_id": "original"}, 1, 0)
            selection = SimpleNamespace(status="committed", committed_candidate_id="winner")
            evaluations = [{"candidate_id": "winner", "status": "ok", "relative_score": 0.8}]
            with patch("benchmark.runner._evaluate", return_value=outcome), \
                    patch("benchmark.runner._evaluate_candidates",
                          return_value=(selection, evaluations, {}, 0, 0)), \
                    patch("benchmark.runner._is_fatal_method_error", return_value=False):
                record = _run_one(task, controller=controller, ordinal=1, output_dir=root)
            self.assertEqual(record["relative_score"], 0.2)
            self.assertEqual(record["post_refinement_score"], 0.8)
            saved = json.loads((root / "runs/train/0001_ordinary.json").read_text())
            self.assertEqual(saved["post_refinement_score"], 0.8)


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def ace_task(self):
        task = self.root / "rsi/ace/run_test/task_0001"
        task.mkdir(parents=True)
        return task

    def test_ace_nested_candidate_operations_reach_analysis(self):
        task = self.ace_task()
        operation = {"operation_type": "ADD", "content": "Check action validity."}
        write_json(task / "candidates/refine_1_candidate_000/curator_operations_diff.jsonl", operation)
        self.assertEqual(collect_ace_artifacts(self.root), [("task_0001", [operation])])
        output = AnalysisOutput(method="ACE", round="task_0001", opinions=[])
        with patch("analyze.ace.analyze_artifact", return_value=output) as analyze:
            state = run_ace_analysis(self.root, self.root / "analysis.json")
        analyze.assert_called_once()
        self.assertEqual(state["status"], "completed")

    def test_ace_legacy_flat_operations(self):
        task = self.ace_task()
        operation = {"operation_type": "ADD", "content": "Check action validity."}
        write_json(task / "candidates/curator_operations_diff.jsonl", operation)
        self.assertEqual(collect_ace_artifacts(self.root)[0][1], [operation])

    def test_ace_missing_artifact_is_not_completed(self):
        self.ace_task()
        with patch("analyze.ace.analyze_artifact", side_effect=AssertionError("No model allowed")):
            state = run_ace_analysis(self.root, self.root / "analysis.json")
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["missing_artifact_count"], 1)

    def test_ace_incomplete_cli_returns_failure(self):
        with patch.object(sys, "argv", ["analyze", "ace", str(self.root), "--model", "offline"]), \
                patch("analyze.__main__.run_ace_analysis", return_value={
                    "status": "incomplete", "missing_artifact_count": 1, "nodes": [],
                }), self.assertRaisesRegex(SystemExit, "1 missing artifacts"):
            analyze_main()

    def test_ace_empty_operations_are_valid(self):
        task = self.ace_task()
        path = task / "candidates/candidate/curator_operations_diff.jsonl"
        path.parent.mkdir(parents=True)
        path.touch()
        with patch("analyze.ace.analyze_artifact", side_effect=AssertionError("No model allowed")):
            state = run_ace_analysis(self.root, self.root / "analysis.json")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["nodes"][0]["status"], "unchanged")

    def test_ace_ambiguous_operations_fail(self):
        task = self.ace_task()
        for candidate in ("first", "second"):
            write_json(task / "candidates" / candidate / "curator_operations_diff.jsonl", {})
        with self.assertRaisesRegex(ValueError, "not unique"):
            collect_ace_artifacts(self.root)

    def ahe_iteration(self, index, accepted):
        run = self.root / "rsi/ahe/run_test"
        iteration = f"iteration_{index:03d}"
        candidate = run / "state/candidates" / iteration / "variant_0"
        write_json(candidate / "evolve_result.json", {"summary": f"proposal {index}"})
        write_json(candidate / "workspace/agent.json", {"version": index})
        write_json(run / "state/runs" / iteration / "input/workspace/agent.json", {"version": 0})
        report_path = run / f"task_{index:04d}/selection_stage_000/ahe_selection.json"
        write_json(report_path, {
            "refinement_id": iteration, "status": "committed" if accepted else "unchanged",
            "parent_score": 0.5, "candidate_scores": {"variant_0": 0.8 if accepted else 0.2},
            "committed_candidate_id": "variant_0" if accepted else None,
        })
        return report_path

    def test_ahe_rejected_candidates_do_not_become_parents(self):
        for index, accepted in enumerate((False, True, False, True, False), 1):
            self.ahe_iteration(index, accepted)
        nodes = collect_ahe_artifacts(self.root)
        self.assertEqual([node["parent_id"] for node in nodes], [
            None, None, "iteration_002:variant_0", "iteration_002:variant_0", "iteration_004:variant_0",
        ])
        self.assertEqual([node["accepted"] for node in nodes], [False, True, False, True, False])

    def test_ahe_analysis_uses_parent_snapshots_and_retained_final(self):
        for index, accepted in enumerate((False, True, False, True, False), 1):
            self.ahe_iteration(index, accepted)
        priors = []

        def analyze(method, node_id, artifact, prior, **kwargs):
            priors.append([opinion["created_at_node"] for opinion in prior])
            return AnalysisOutput(method=method, round=node_id, opinions=[ImprovementOpinion(
                normalized_opinion=f"proposal from {node_id}", source_evidence=["saved diff"],
                scope=None, function=[], surface=None, semantically_new=True,
                contradiction_or_revision=False, duplicate=False,
                replaces_prior_opinion_ids=[], replaces_current_opinion_indices=[],
            )])

        with patch("analyze.ahe.analyze_artifact", side_effect=analyze):
            state = run_ahe_analysis(self.root, self.root / "analysis.json")
        self.assertEqual(priors, [
            [], [], ["iteration_002:variant_0"], ["iteration_002:variant_0"],
            ["iteration_002:variant_0", "iteration_004:variant_0"],
        ])
        retained = [state["opinions"][key]["created_at_node"] for key in state["final_snapshot_opinion_ids"]]
        self.assertEqual(retained, ["iteration_002:variant_0", "iteration_004:variant_0"])
        self.assertEqual(len(state["nodes"]), 5)
        self.assertEqual(len(state["opinions"]), 5)

    def test_ahe_all_rejected_has_empty_final_snapshot(self):
        self.ahe_iteration(1, False)
        with patch("analyze.ahe.analyze_artifact", return_value=AnalysisOutput(
                method="AHE", round="iteration_001:variant_0", opinions=[])):
            state = run_ahe_analysis(self.root, self.root / "analysis.json")
        self.assertEqual(state["final_snapshot_opinion_ids"], [])

    def test_ahe_missing_selection_fails_instead_of_inventing_chain(self):
        report = self.ahe_iteration(1, True)
        report.unlink()
        with self.assertRaisesRegex(ValueError, "selection result is missing"):
            collect_ahe_artifacts(self.root)


if __name__ == "__main__":
    unittest.main()
