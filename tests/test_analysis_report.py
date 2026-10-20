"""Offline regression tests for evidence-linked per-game trajectory reports."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from analyze.__main__ import _build_parser, main
from analyze.llm import AnalysisLLMError
from analyze.report import (ReportInsights, add_insights, attach_score_statistics, build_report,
                            build_statistics, render_markdown, run_report, write_report)


def opinion(text="Inspect the current target before mining.", **overrides):
    return {
        "normalized_opinion": text, "source_evidence": [text],
        "scope": "task_specific", "function": ["exploration", "execution"],
        "surface": "prompt_memory", "semantically_new": True,
        "contradiction_or_revision": False, "duplicate": False,
        "replaces_prior_opinion_ids": [], "replaces_current_opinion_indices": [],
        **overrides,
    }


def node(node_id="round_1", *, method="AHE", parent=0.5, current=0.3, opinions=None, **overrides):
    return {
        "node_id": node_id, "parent_id": None, "status": "completed", "accepted": True,
        "parent_score": parent, "current_score": current,
        "score_delta": current - parent if current is not None and parent is not None else None,
        "analysis": {"method": method, "round": node_id,
                     "opinions": [opinion(), opinion("Follow the requested action format.")]
                     if opinions is None else opinions},
        # A corrupt cached summary must never affect recomputed counts.
        "summary": {"opinion_count": 9999}, **overrides,
    }


def state(nodes=None, method="AHE", **overrides):
    return {"schema_version": 7 if method == "AHE" else 6, "method": method,
            "status": "completed", "nodes": [node(method=method)] if nodes is None else nodes, **overrides}


class GameReportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def source(self, payload=None, name="analysis.json"):
        path = self.root / name
        path.write_text(json.dumps(state() if payload is None else payload), encoding="utf-8")
        return path

    def test_recomputes_counts_and_deduplicates_node_score_associations(self):
        report = build_report([self.source()], "minecraft")
        run = report["runs"][0]
        self.assertEqual(run["proposal_count"], 2)
        self.assertEqual(run["classification_counts"]["scope"], {"task_specific": 2})
        self.assertEqual(run["accepted_negative_node_count"], 1)
        association = next(row for row in run["category_score_associations"] if row["label"] == "execution")
        self.assertEqual(association["node_count"], 1)
        self.assertEqual(association["negative"], 1)
        self.assertAlmostEqual(association["mean_score_delta"], -0.2)
        self.assertEqual(len(report["evidence"]), 3)

    def test_missing_scores_unknown_acceptance_and_incomplete_analysis_are_visible(self):
        unknown = node("round_2", parent=None, current=0.6, accepted=None, score_delta=0.4)
        missing = node("round_3", parent=0, current=0, analysis=None, status="missing_artifact", accepted=False)
        report = build_report([self.source(state([node(), unknown, missing], status="incomplete"))], "farm")
        run = report["runs"][0]
        self.assertEqual(run["analysis_status"], "incomplete")
        self.assertEqual(run["classified_node_count"], 2)
        self.assertEqual(run["scored_node_count"], 2)
        self.assertEqual(run["missing_score_count"], 1)
        self.assertEqual(run["unknown_acceptance_count"], 1)
        self.assertIsNone(report["nodes"][1]["score_delta"])
        self.assertIsNone(report["nodes"][2]["proposal_count"])

    def test_unknown_classification_and_overlapping_relations(self):
        proposals = [opinion(scope=None, function=[], surface=None, semantically_new=None),
                     opinion(contradiction_or_revision=True)]
        report = build_report([self.source(state([node(opinions=proposals)]))], "hero")
        run = report["runs"][0]
        self.assertEqual(run["unknown_relation_counts"], {"semantically_new": 1})
        self.assertEqual(run["classification_counts"]["scope"]["undetermined"], 1)
        self.assertEqual(run["classification_counts"]["function"]["undetermined"], 1)
        self.assertEqual(run["classification_counts"]["relations"]["contradiction_or_revision"], 1)

    def test_three_frameworks_have_separate_run_and_evidence_ids(self):
        sources = [self.source(state(method=method), f"{method}.json") for method in ("ACE", "AHE", "HyperAgent")]
        for game in ("farm", "combat", "sweepmine", "hero", "minecraft", "astray"):
            with self.subTest(game=game):
                report = build_report(sources, game)
                self.assertEqual([run["method"] for run in report["runs"]], ["ACE", "AHE", "HyperAgents"])
                self.assertEqual(len(report["evidence"]), 9)

    def test_duplicate_inputs_nodes_bad_labels_and_identity_are_rejected(self):
        source = self.source()
        with self.assertRaisesRegex(ValueError, "Duplicate trajectory"):
            build_report([source, source], "farm")
        cases = (
            state([node(), node()]),
            state([node(score_delta=99)]),
            state([node(opinions=[opinion(surface="invented")])]),
            state([node(method="ACE")]),
            {"schema_version": 4, "method": "AHE", "nodes": []},
        )
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                build_report([self.source(payload)], "farm")

    def test_archived_schema_five_and_embedded_game_validation(self):
        source = self.source(state(schema_version=5, game="farm"))
        report = build_report([source], "farm")
        self.assertEqual(report["sources"][0]["game_identity"], "embedded")
        with self.assertRaisesRegex(ValueError, "game mismatch"):
            build_report([source], "combat")

    def test_json_csv_and_markdown_preserve_evidence_and_missing_values(self):
        source = self.source(state([node(parent=None, opinions=[opinion("Check A | B\nthen continue.")])]))
        original = source.read_bytes()
        report = build_report([source], "farm")
        write_report(report, self.root / "reports")
        self.assertEqual(source.read_bytes(), original)
        saved = json.loads((self.root / "reports/report.json").read_text())
        self.assertEqual(saved["proposals"][0]["source_evidence"], ["Check A | B\nthen continue."])
        self.assertIsNone(saved["nodes"][0]["score_delta"])
        with (self.root / "reports/proposals.csv").open(newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)
        markdown = (self.root / "reports/report.md").read_text()
        self.assertIn("Classification and Score Report", markdown)
        self.assertTrue(markdown.isascii())
        self.assertEqual(saved["language"], "en")
        self.assertNotIn("Check A", markdown)
        statistics = (self.root / "reports/statistics.json").read_text()
        self.assertNotIn("Check A", statistics)
        self.assertNotIn(str(source), statistics)
        self.assertIn("N/A", markdown)
        self.assertIn("N/A: no scores available", render_markdown(report))

    def test_empty_trajectory_and_zero_delta_do_not_invent_gains(self):
        report = build_report([self.source(state([]))], "astray")
        self.assertEqual(report["runs"][0]["proposal_count"], 0)
        self.assertIsNone(report["runs"][0]["mean_score_delta"])
        write_report(report, self.root / "empty")
        report = build_report([self.source(state([node(parent=0, current=0, opinions=[])]))], "astray")
        self.assertEqual(report["runs"][0]["zero"], 1)
        self.assertEqual(report["runs"][0]["mean_score_delta"], 0)

    def test_input_overwrite_guard(self):
        source = self.source(name="report.json")
        report = build_report([source], "farm")
        original = source.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            write_report(report, self.root)
        self.assertEqual(source.read_bytes(), original)

    def insights(self, ref="run_001/summary"):
        statement = dict(title="Candidate selection", observation="One accepted node has a negative delta.",
                         interpretation="Selection may use another criterion.", implication="Audit selection metrics.",
                         evidence_ids=[ref], next_experiment="Replay parent and candidate on identical tasks.")
        return ReportInsights(findings=[statement], hypotheses=[statement])

    def client(self, insights):
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(output_parsed=insights, id="offline-response", usage=None)
        return client

    def test_model_insights_require_existing_evidence_and_retain_telemetry(self):
        report = build_report([self.source()], "minecraft")
        client = self.client(self.insights())
        add_insights(report, model="offline-test", client=client)
        self.assertEqual(report["status"], "interpreted")
        self.assertNotIn("Recommendations", render_markdown(report))
        self.assertTrue(render_markdown(report).isascii())
        self.assertEqual(report["language"], "en")
        self.assertIn("Write all report text in English", client.responses.parse.call_args.kwargs["input"])
        self.assertEqual(report["telemetry"]["responses"][0]["response_id"], "offline-response")
        self.assertIn("not causal", client.responses.parse.call_args.kwargs["input"])
        for ref in ("invented/node", "", "run_001/node_0001", "run_001/node_0001/proposal_0000"):
            with self.subTest(ref=ref), self.assertRaises(AnalysisLLMError):
                add_insights(report, model="offline-test", client=self.client(self.insights(ref)))

    def test_model_size_limit_is_checked_before_api_call(self):
        report = build_report([self.source()], "farm")
        client = Mock()
        with self.assertRaisesRegex(ValueError, "character limit"):
            add_insights(report, model="offline-test", client=client, max_input_chars=10)
        client.responses.parse.assert_not_called()

    def test_only_statistics_are_sent_even_for_large_raw_proposals(self):
        sentinel = "PRIVATE_RAW_PROPOSAL_DO_NOT_SEND"
        nodes = [node(f"round_{i}", opinions=[opinion(sentinel * 9000)]) for i in range(4)]
        report = build_report([self.source(state(nodes))], "farm")
        client = Mock()

        def reply(**kwargs):
            payload = json.loads(kwargs["input"].split("Requested language: en\n", 1)[1])
            self.assertEqual(payload, build_statistics(report))
            self.assertNotIn(sentinel, kwargs["input"])
            self.assertNotIn("source_evidence", kwargs["input"])
            self.assertNotIn("normalized_opinion", kwargs["input"])
            return SimpleNamespace(output_parsed=self.insights(), id="statistics-only", usage=None)

        client.responses.parse.side_effect = reply
        add_insights(report, model="offline-test", client=client, max_input_chars=19000)
        self.assertEqual(client.responses.parse.call_count, 1)
        self.assertEqual(report["interpretation_scope"], "classification_and_score_statistics_only")
        self.assertEqual(report["status"], "interpreted")

    def score_csv(self, rows, name="scores.csv"):
        path = self.root / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def evaluation(self, **overrides):
        return dict(game="minecraft", method="ahe", metric="G1", target_game="minecraft",
                    score_group="medium", initial_score=0.1234567890123, final_score=0.7,
                    reference_score=1, hci_ratio=0.123456789012345, hci_status="ok", **overrides)

    def test_score_import_preserves_precision_and_discards_arbitrary_text(self):
        report = build_report([self.source()], "minecraft")
        row = self.evaluation()
        row.update(evidence="PRIVATE_TEXT", initial_source="SECRET_PATH", paper_hci_ratio=99)
        path = self.score_csv([row, {**row, "game": "farm"}])
        attach_score_statistics(report, [path])
        saved = report["score_statistics"][0]
        self.assertEqual(saved["initial_score"], row["initial_score"])
        self.assertEqual(saved["hci_ratio"], row["hci_ratio"])
        serialized = json.dumps(build_statistics(report))
        for text in ("PRIVATE_TEXT", "SECRET_PATH", "paper_hci", str(path)):
            self.assertNotIn(text, serialized)
        with self.assertRaisesRegex(ValueError, "Duplicate score"):
            attach_score_statistics(report, [path])
        with self.assertRaisesRegex(ValueError, "duplicate score summary"):
            attach_score_statistics(report, [self.score_csv([self.evaluation()], "other.csv")])

    def test_invalid_numeric_score_metadata_and_counts_are_rejected(self):
        base = self.evaluation()
        for change in ({"method": "ace"}, {"metric": "G4"}, {"target_game": "unknown"},
                       {"score_group": "easy"}, {"hci_status": "made_up"}, {"final_score": "nan"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                attach_score_statistics(build_report([self.source()], "minecraft"),
                                        [self.score_csv([{**base, **change}])])
        summary = dict(game="minecraft", method="ahe", valid_pairs=19, positive=7,
                       zero=3, negative=9, mean_delta=-0.03)
        report = build_report([self.source()], "minecraft")
        attach_score_statistics(report, [self.score_csv([summary])])
        self.assertEqual(report["score_statistics"][0]["kind"], "adjacent_training_rounds")
        for change in ({"positive": 8}, {"valid_pairs": -1}, {"zero": 3.5}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                attach_score_statistics(build_report([self.source()], "minecraft"),
                                        [self.score_csv([{**summary, **change}])])

    def test_category_score_import_and_unknown_category(self):
        row = dict(game="minecraft", method="ahe", category="planning", pairs=4, positive=2,
                   zero=1, negative=1, mean_delta=0.03)
        report = build_report([self.source()], "minecraft")
        attach_score_statistics(report, [self.score_csv([row])])
        self.assertEqual(report["score_statistics"][0]["category"], "planning")
        with self.assertRaisesRegex(ValueError, "Unknown score category"):
            attach_score_statistics(build_report([self.source()], "minecraft"),
                                    [self.score_csv([{**row, "category": "raw proposal text"}])])

    def test_hci_json_missing_scores_remain_missing_and_raw_pairs_are_excluded(self):
        data = dict(schema_version=1, source_game="minecraft", method="ahe", groups={
            "G1": dict(target_game="minecraft", difficulty_group="medium", status="undefined",
                       initial_score=1, final_score=1, reference_score=1, hci=None,
                       pairs=[{"secret": "RAW_PAIR_TEXT"}]),
            "G0": dict(target_game="minecraft", status="unavailable", initial_score=None,
                       final_score=None, reference_score=1, hci=None),
        })
        report = build_report([self.source()], "minecraft")
        path = self.source(data, "hci.json")
        attach_score_statistics(report, [path])
        self.assertIsNone(report["score_statistics"][0]["hci_ratio"])
        self.assertIsNone(report["score_statistics"][1]["initial_score"])
        self.assertNotIn("RAW_PAIR_TEXT", json.dumps(build_statistics(report)))
        self.assertIn("N/A", render_markdown(report))
        with self.assertRaisesRegex(ValueError, "Invalid HCI group"):
            attach_score_statistics(build_report([self.source()], "minecraft"),
                                    [self.source({**data, "groups": {"G1": None}}, "bad.json")])

    def test_score_input_overwrite_guard_and_cli_flag(self):
        path = self.score_csv([self.evaluation()], "nodes.csv")
        report = build_report([self.source()], "minecraft")
        attach_score_statistics(report, [path])
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            write_report(report, self.root)
        self.assertEqual(path.read_bytes(), original)
        args = _build_parser().parse_args(["report", "--game", "minecraft", "--input", "a.json",
                                          "--score-stats", "s.csv", "--output-dir", "out"])
        self.assertEqual(args.score_stats, ["s.csv"])
        self.assertFalse(hasattr(args, "language"))

    def test_report_cli_has_no_language_selector(self):
        command = ["report", "--game", "minecraft", "--input", "a.json", "--output-dir", "out"]
        for language in ("zh", "en"):
            with self.subTest(language=language), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                _build_parser().parse_args(command + ["--language", language])
            self.assertEqual(error.exception.code, 2)

    def test_cli_statistics_mode_does_not_call_model_even_with_model_env(self):
        source = self.source()
        args = ["analyze", "report", "--game", "minecraft", "--input", str(source),
                "--output-dir", str(self.root / "cli")]
        with patch("sys.argv", args), patch.dict("os.environ", {"OPENAI_MODEL": "do-not-call"}), \
                patch("analyze.report._create_openai_client") as create, redirect_stdout(io.StringIO()):
            main()
        create.assert_not_called()
        self.assertTrue((self.root / "cli/report.md").is_file())
        self.assertEqual(_build_parser().parse_args(args[1:]).command, "report")

    def test_failed_model_keeps_statistical_report(self):
        source = self.source()
        args = SimpleNamespace(inputs=[source], game="farm", output_dir=self.root / "failed",
                               model="offline-test", timeout=None)
        with patch("analyze.report.add_insights", side_effect=AnalysisLLMError("offline failure")), \
                self.assertRaises(AnalysisLLMError):
            run_report(args)
        report = json.loads((args.output_dir / "report.json").read_text())
        self.assertEqual(report["status"], "interpretation_failed")
        self.assertEqual(report["runs"][0]["proposal_count"], 2)
        self.assertTrue((args.output_dir / "statistics.json").is_file())
        self.assertIn("offline failure", (args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
