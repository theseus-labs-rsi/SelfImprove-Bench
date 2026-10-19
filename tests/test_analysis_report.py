"""Offline regression tests for evidence-linked per-game trajectory reports."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from analyze.__main__ import _build_parser, main
from analyze.llm import AnalysisLLMError
from analyze.report import ReportInsights, add_insights, build_report, render_markdown, run_report, write_report


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
        self.assertIn("轨迹分析报告", markdown)
        self.assertIn("Check A \\| B then continue.", markdown)
        self.assertIn("N/A", markdown)
        self.assertIn("not adjacent", render_markdown(report, "en"))

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

    def insights(self, ref="run_001/node_0001"):
        statement = dict(title="Candidate selection", observation="One accepted node has a negative delta.",
                         interpretation="Selection may use another criterion.", implication="Audit selection metrics.",
                         evidence_ids=[ref], next_experiment="Replay parent and candidate on identical tasks.")
        recommendation = dict(title="Audit selection", problem="Negative recorded delta.",
                              proposed_change="Log the exact selection metric alongside game score.",
                              expected_benefit="Make acceptance explainable.", evidence_ids=[ref],
                              validation_experiment="Compare paired candidate evaluations.")
        return ReportInsights(findings=[statement], recommendations=[recommendation], hypotheses=[statement])

    def client(self, insights):
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(output_parsed=insights, id="offline-response", usage=None)
        return client

    def test_model_insights_require_existing_evidence_and_retain_telemetry(self):
        report = build_report([self.source()], "minecraft")
        client = self.client(self.insights())
        add_insights(report, model="offline-test", client=client)
        self.assertEqual(report["status"], "interpreted")
        self.assertIn("改进建议", render_markdown(report))
        self.assertEqual(report["telemetry"]["responses"][0]["response_id"], "offline-response")
        self.assertIn("not causal", client.responses.parse.call_args.kwargs["input"])
        for ref in ("invented/node", ""):
            with self.subTest(ref=ref), self.assertRaises(AnalysisLLMError):
                add_insights(report, model="offline-test", client=self.client(self.insights(ref)))

    def test_model_size_limit_is_checked_before_api_call(self):
        report = build_report([self.source()], "farm")
        client = Mock()
        with self.assertRaisesRegex(ValueError, "character limit"):
            add_insights(report, model="offline-test", client=client, max_input_chars=10)
        client.responses.parse.assert_not_called()

    def test_large_reports_batch_whole_nodes_then_synthesize(self):
        nodes = [node(f"round_{i}", opinions=[opinion("x" * 9000)]) for i in range(4)]
        report = build_report([self.source(state(nodes))], "farm")
        client = Mock()

        def reply(**kwargs):
            payload = json.loads(kwargs["input"].split("Requested language: zh\n", 1)[1])
            catalogue = payload.get("evidence_catalogue")
            if catalogue:
                refs = {entry["evidence_id"] for entry in catalogue}
                for entry in catalogue:
                    if "node_evidence_id" in entry:
                        self.assertIn(entry["node_evidence_id"], refs)
                ref = next(entry["evidence_id"] for entry in catalogue if "proposal_count" in entry)
            else:
                ref = payload["batch_interpretations"][0]["findings"][0]["evidence_ids"][0]
            return SimpleNamespace(output_parsed=self.insights(ref), id="batched", usage=None)

        client.responses.parse.side_effect = reply
        add_insights(report, model="offline-test", client=client, max_input_chars=19000)
        self.assertGreater(report["interpretation_batch_count"], 1)
        self.assertEqual(client.responses.parse.call_count, report["interpretation_batch_count"] + 1)
        self.assertEqual(len(report["telemetry"]["responses"]), client.responses.parse.call_count)
        self.assertEqual(report["status"], "interpreted")

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
                               model="offline-test", language="zh", timeout=None)
        with patch("analyze.report.add_insights", side_effect=AnalysisLLMError("offline failure")), \
                self.assertRaises(AnalysisLLMError):
            run_report(args)
        report = json.loads((args.output_dir / "report.json").read_text())
        self.assertEqual(report["status"], "interpretation_failed")
        self.assertEqual(report["runs"][0]["proposal_count"], 2)
        self.assertIn("offline failure", (args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
