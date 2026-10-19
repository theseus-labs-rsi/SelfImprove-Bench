"""Offline checks for HCI formulas, real config grouping, and identity safeguards."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchmark.config import ReplayConfig, load_config
from benchmark.hci import calculate_hci, headroom_closed_index, main
from benchmark.tasks import GAMES, build_manifest, task_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def fixture(game="farm", method="ace"):
    config = load_config(ROOT / "configs/main" / f"{game}_{method}.json")
    initial_config = replace(config, schedule=(), replay=ReplayConfig())

    def generate(difficulty, *, seed, task_id):
        return {"difficulty": difficulty, "seed": seed, "task_id": task_id}

    # Exercise actual orchestration/grouping without running reference solvers or agents.
    with ExitStack() as stack:
        for name, module in GAMES.items():
            if name != "gameworld":
                stack.enter_context(patch.object(module, "generate", side_effect=generate))
        manifest = build_manifest(config).as_dict()
        initial_manifest = build_manifest(initial_config).as_dict()

    def result(settings, tasks, baseline):
        def record(task):
            return {
                **{key: value for key, value in task.items()
                   if key not in {"task", "task_fingerprint"}},
                "status": "ok", "relative_score": 0.2 if baseline else 0.6,
                "post_refinement_score": 0.4,
            }
        return {"config": settings.as_dict(),
                "training_runs": [record(task) for task in tasks["training"]],
                "holdout_runs": [record(task) for task in tasks["holdout"]]}

    return result(config, manifest, False), result(initial_config, initial_manifest, True), manifest, initial_manifest


class HCITest(unittest.TestCase):
    def test_all_eighteen_main_configs_group_correctly(self):
        for game in ("farm", "combat", "sweepmine", "hero", "minecraft", "astray"):
            for method in ("ace", "ahe", "hyperagents"):
                with self.subTest(game=game, method=method):
                    report = calculate_hci(*fixture(game, method))
                    self.assertEqual(report["status"], "completed")
                    self.assertEqual(report["source_game"], game)
                    self.assertEqual(report["method"], method)
                    self.assertAlmostEqual(report["groups"]["G0"]["hci"], 0)
                    self.assertEqual(report["g0_baseline"], "pre_refinement_execution_score")
                    self.assertEqual(report["groups"]["G0"]["task_count"], 4)
                    self.assertEqual(
                        [(p["original_round"], p["replay_after_round"])
                         for p in report["groups"]["G0"]["pairs"]],
                        [(2, 5), (7, 10), (12, 15), (17, 20)],
                    )
                    expected = {
                        "minecraft": (2, 2, 1), "astray": (1, 1, 2),
                    }.get(game, (20, 20, 20))
                    for name, count in zip(("G1", "G2", "G3"), expected):
                        group = report["groups"][name]
                        self.assertEqual(group["task_count"], count)
                        self.assertAlmostEqual(group["hci"], 0.5)

    def test_g3_uses_destination_medium_and_matching_initial(self):
        data = fixture("minecraft")
        # First five holdouts are source Minecraft; last five are destination Astray.
        for record in data[1]["holdout_runs"][:5]:
            record["relative_score"] = 0.9
        for record in data[1]["holdout_runs"][5:]:
            record["relative_score"] = 0.4
        group = calculate_hci(*data)["groups"]["G3"]
        self.assertEqual(group["target_game"], "astray")
        self.assertAlmostEqual(group["initial_score"], 0.4)
        self.assertAlmostEqual(group["hci"], 1 / 3)
        self.assertEqual([pair["game_task_id"] for pair in group["pairs"]], ["03_03"])

    def test_aggregate_scores_before_dividing_not_mean_of_task_hci(self):
        data = fixture("minecraft")
        for records, values in ((data[1]["holdout_runs"], (0.2, 0.8)),
                                (data[0]["holdout_runs"], (0.4, 0.9))):
            records[1]["relative_score"], records[2]["relative_score"] = values
        group = calculate_hci(*data)["groups"]["G1"]
        self.assertAlmostEqual(group["initial_score"], 0.5)
        self.assertAlmostEqual(group["final_score"], 0.65)
        self.assertAlmostEqual(group["hci"], 0.3)

    def test_g0_uses_pre_score_not_committed_score_or_best_candidate(self):
        data = fixture()
        for record in data[0]["training_runs"]:
            if not record["replay"]:
                del record["post_refinement_score"]
                record["relative_score"] = 0.1
                record["selection"] = {
                    "status": "committed", "committed_candidate_id": "winner",
                    "stages": [{"committed_candidate_id": "winner", "evaluations": [
                        {"candidate_id": "rejected", "status": "ok", "relative_score": 0.95},
                        {"candidate_id": "winner", "status": "ok", "relative_score": 0.5},
                    ]}],
                }
        group = calculate_hci(*data)["groups"]["G0"]
        self.assertEqual(group["initial_score"], 0.1)
        self.assertAlmostEqual(group["hci"], 5 / 9)

    def test_failed_and_skipped_executions_are_zero_and_statuses_survive(self):
        data = fixture("minecraft")
        data[0]["holdout_runs"][1].update(status="error", relative_score=0.9)
        data[0]["holdout_runs"][2].update(status="skipped", relative_score=None)
        group = calculate_hci(*data)["groups"]["G1"]
        self.assertEqual(group["final_score"], 0)
        self.assertAlmostEqual(group["hci"], -0.25)
        self.assertEqual([p["final_status"] for p in group["pairs"]], ["error", "skipped"])
        replay = next(record for record in data[0]["training_runs"] if record["replay"])
        replay.update(status="error", relative_score=None)
        self.assertEqual(calculate_hci(*data)["groups"]["G0"]["pairs"][0]["final_score"], 0)

    def test_success_with_missing_or_nonfinite_score_is_unavailable(self):
        for value in (None, True, float("nan"), float("inf")):
            with self.subTest(value=value):
                data = fixture("minecraft")
                data[0]["holdout_runs"][1]["relative_score"] = value
                report = calculate_hci(*data)
                self.assertEqual(report["status"], "incomplete")
                self.assertIsNone(report["groups"]["G1"]["hci"])
                self.assertIsNone(report["groups"]["G1"]["final_score"])

    def test_zero_headroom_is_undefined_not_zero(self):
        data = fixture("minecraft")
        for record in data[1]["holdout_runs"]:
            record["relative_score"] = 1.0
        report = calculate_hci(*data)
        for name in ("G1", "G2", "G3"):
            self.assertEqual(report["groups"][name]["status"], "undefined")
            self.assertEqual(report["groups"][name]["reason"], "zero_headroom")
            self.assertIsNone(report["groups"][name]["hci"])

    def test_formula_does_not_clip_and_ignores_raw_reference_scale(self):
        self.assertAlmostEqual(headroom_closed_index(0.5, 1.5), 2)
        self.assertAlmostEqual(headroom_closed_index(1.5, 1.8), -0.6)
        data = fixture()
        for result in data[:2]:
            for record in result["holdout_runs"]:
                record["reference_score"] = 1000
        self.assertAlmostEqual(calculate_hci(*data)["groups"]["G1"]["hci"], 0.5)

    def test_g0_does_not_require_post_refinement_evidence(self):
        data = fixture()
        for record in data[0]["training_runs"]:
            if not record["replay"]:
                record["post_refinement_score"] = None
        report = calculate_hci(*data)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["groups"]["G0"]["hci"], 0)
        self.assertEqual(report["groups"]["G0"]["valid_pair_count"], 4)

    def test_g0_missing_or_invalid_pre_score_is_not_replaced_by_post_score(self):
        for value in (None, True, float("nan"), float("inf")):
            with self.subTest(value=value):
                data = fixture()
                for record in data[0]["training_runs"]:
                    if not record["replay"]:
                        record["relative_score"] = value
                group = calculate_hci(*data)["groups"]["G0"]
                self.assertEqual(group["status"], "unavailable")
                self.assertIsNone(group["hci"])
                self.assertEqual(group["valid_pair_count"], 0)

    def test_missing_replays_is_not_zero_retention(self):
        data = fixture()
        data[2]["training"] = [t for t in data[2]["training"] if not t["replay"]]
        data[0]["training_runs"] = [r for r in data[0]["training_runs"] if not r["replay"]]
        self.assertEqual(calculate_hci(*data)["groups"]["G0"]["status"], "unavailable")

    def test_one_missing_replay_baseline_prevents_partial_g0_average(self):
        data = fixture()
        original_id = next(r["replay_of"] for r in data[0]["training_runs"] if r["replay"])
        original = next(r for r in data[0]["training_runs"] if r["run_id"] == original_id)
        original["relative_score"] = None
        group = calculate_hci(*data)["groups"]["G0"]
        self.assertEqual(group["valid_pair_count"], 3)
        self.assertIsNone(group["hci"])

    def test_failed_execution_uses_zero_g0_baseline_regardless_of_selection(self):
        for status in ("error", "skipped"):
            with self.subTest(status=status):
                data = fixture()
                for record in data[0]["training_runs"]:
                    if not record["replay"]:
                        record.update(status=status, selection={"status": "committed"})
                group = calculate_hci(*data)["groups"]["G0"]
                self.assertEqual(group["initial_score"], 0)
                self.assertAlmostEqual(group["hci"], 0.6)

    def test_mismatch_and_incomplete_records_are_rejected(self):
        mutations = (
            lambda d: d[0]["holdout_runs"].pop(),
            lambda d: d[0]["training_runs"].pop(),
            lambda d: d[0]["holdout_runs"].append(deepcopy(d[0]["holdout_runs"][0])),
            lambda d: d[1]["config"]["agent"].update(provider="ahe"),
            lambda d: d[1]["config"].update(schedule=[{"game": "farm"}]),
            lambda d: d[3]["holdout"][0]["task"].update(seed=-1),
            lambda d: d[0]["holdout_runs"][0].update(task_seed=-1),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                data = fixture()
                mutation(data)
                with self.assertRaises(ValueError):
                    calculate_hci(*data)

    def test_same_display_id_with_different_task_content_is_rejected(self):
        data = fixture()
        task = data[3]["holdout"][0]
        task["task"]["seed"] = -1
        task["task_fingerprint"] = task_fingerprint(task["task"])
        with self.assertRaisesRegex(ValueError, "identities differ"):
            calculate_hci(*data)

    def test_replay_must_match_original_task(self):
        data = fixture()
        task = next(t for t in data[2]["training"] if t["replay"])
        task["task"] = {**task["task"], "seed": -1}
        task["task_fingerprint"] = task_fingerprint(task["task"])
        with self.assertRaisesRegex(ValueError, "Replay does not match"):
            calculate_hci(*data)

    def test_missing_medium_task_is_not_a_partial_mean(self):
        data = fixture("minecraft")
        run_id = data[2]["holdout"][1]["run_id"]
        for manifest in data[2:]:
            manifest["holdout"] = [t for t in manifest["holdout"] if t["run_id"] != run_id]
        for result in data[:2]:
            result["holdout_runs"] = [r for r in result["holdout_runs"] if r["run_id"] != run_id]
        group = calculate_hci(*data)["groups"]["G1"]
        self.assertEqual(group["status"], "unavailable")
        self.assertIsNone(group["hci"])

    def test_cli_success_incomplete_and_source_overwrite_guard(self):
        data = fixture("minecraft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, result, manifest in (("trained", data[0], data[2]), ("initial", data[1], data[3])):
                (root / name).mkdir()
                (root / name / "result.json").write_text(json.dumps(result))
                (root / name / "task_manifest.json").write_text(json.dumps(manifest))
            output = root / "hci.json"
            command = ["hci", "--trained", str(root / "trained"), "--initial", str(root / "initial"),
                       "--output", str(output)]
            with patch.object(sys, "argv", command), redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(json.loads(output.read_text())["status"], "completed")
            before = (root / "trained/result.json").read_bytes()
            with patch.object(sys, "argv", command[:-1] + [str(root / "trained/result.json")]), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main()
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual((root / "trained/result.json").read_bytes(), before)
            data[0]["holdout_runs"][1]["relative_score"] = None
            (root / "trained/result.json").write_text(json.dumps(data[0]))
            with patch.object(sys, "argv", command), redirect_stdout(io.StringIO()), \
                    self.assertRaises(SystemExit) as exc:
                main()
            self.assertEqual(exc.exception.code, 1)
            self.assertEqual(json.loads(output.read_text())["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
