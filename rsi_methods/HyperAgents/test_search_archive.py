"""Offline regressions for archive search and unrestricted tool evolution."""

import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .candidate import CandidateSnapshot, with_lineage
from .HyperAgentsAdaptor import (
    HyperAgentsAdaptor, HyperAgentsCandidateBatch, HyperAgentsCandidateSpec,
)
from .search_archive import SearchArchive
from benchmark.interfaces import MethodContext


class SearchArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "state" / "run_test"
        self.workspace.mkdir(parents=True)
        self.initial = CandidateSnapshot("initial", self.workspace, self.workspace, "root")

    def candidate(self, key="child", parent=None, *, changed=True):
        parent = parent or self.initial
        directory = self.root / key
        directory.mkdir(exist_ok=True)
        patch_file = directory / "patch.diff"
        patch_file.write_text("patch contents\n")
        snapshot = with_lineage(
            parent, candidate_id=key, parent_id=parent.candidate_id,
            lineage_patches=[*parent.lineage_patches, patch_file] if changed else parent.lineage_patches,
        )
        return HyperAgentsCandidateSpec(
            key, str(parent.candidate_id), directory, Mock(), snapshot,
            Mock(), changed, patch_file if changed else None, {},
        )

    def adaptor(self):
        adaptor = HyperAgentsAdaptor(max_refinement_rounds=20)
        adaptor.workspace_dir = self.workspace
        adaptor._snapshot = self.initial
        adaptor._archive.add(self.initial, score=1.0, valid_parent=True)
        self.addCleanup(adaptor.close)
        return adaptor

    def select(self, adaptor, candidate, *, score=0.2, status="ok", policy=None):
        batch_dir = self.root / f"task_{adaptor._refinement_round:04d}_test"
        batch_dir.mkdir()
        batch = HyperAgentsCandidateBatch("refine", "ready", batch_dir, (candidate,))
        adaptor._pending_batch = batch
        evaluation = {"status": status, "relative_score": score, "success": False}
        kwargs = {} if policy is None else {"return_value": policy}
        choose = patch.object(adaptor._archive, "choose", **kwargs) if policy is not None else None
        if choose is not None:
            choose.start()
        try:
            output = adaptor.select(
                batch, parent_evaluation={"status": "ok", "relative_score": 1.0, "success": True},
                candidate_evaluations={candidate.candidate_id: evaluation},
                output_dir=batch_dir / "selection_stage_0000",
            )
        finally:
            if choose is not None:
                choose.stop()
        return output

    def test_worse_candidate_is_retained_and_can_be_selected_as_parent(self):
        adaptor = self.adaptor()
        candidate = self.candidate()
        adaptor.parent_selection = "latest"
        result = self.select(adaptor, candidate)
        self.assertTrue(result.report.accepted)
        self.assertEqual(result.report.reason, "valid_candidate_added_to_archive")
        self.assertTrue(adaptor._archive.nodes["child"]["valid_parent"])
        self.assertEqual(adaptor.snapshot, candidate.snapshot)
        self.assertEqual(adaptor._archive.choose("best"), self.initial)

    def test_invalid_candidate_is_recorded_but_not_selected(self):
        adaptor = self.adaptor()
        candidate = self.candidate()
        result = self.select(adaptor, candidate, status="error", score=None)
        self.assertFalse(result.report.accepted)
        self.assertIn("child", adaptor._archive.nodes)
        self.assertFalse(adaptor._archive.nodes["child"]["valid_parent"])
        self.assertEqual(adaptor.snapshot, self.initial)
        candidate.instantiated.close.assert_called_once()

    def test_next_parent_can_come_from_an_older_branch(self):
        adaptor = self.adaptor()
        ancestor = self.candidate("ancestor")
        adaptor._archive.add(ancestor.snapshot, score=0.3, valid_parent=True)
        current = self.candidate("current")
        adaptor._archive.add(current.snapshot, score=0.5, valid_parent=True)
        adaptor._snapshot = current.snapshot
        previous = Mock()
        adaptor._instantiated = previous
        candidate = self.candidate(parent=current.snapshot)
        result = self.select(adaptor, candidate, policy=ancestor.snapshot)
        self.assertEqual(result.committed_candidate_id, "ancestor")
        self.assertEqual(result.report.archived_candidate_id, "child")
        self.assertEqual(adaptor.snapshot, ancestor.snapshot)
        self.assertIsNone(adaptor._instantiated)
        previous.close.assert_called_once()
        candidate.instantiated.close.assert_called_once()

    def test_final_agent_is_last_round_output_even_when_score_regresses(self):
        adaptor = self.adaptor()
        adaptor._refinement_round = 20
        candidate = self.candidate()
        result = self.select(adaptor, candidate)
        self.assertEqual(adaptor.snapshot, candidate.snapshot)
        self.assertTrue(result.report.accepted)
        self.assertEqual(result.report.parent_selection, "final_round")
        self.assertEqual(result.status, "committed")

    def test_invalid_final_candidate_retains_current_parent_not_historical_best(self):
        adaptor = self.adaptor()
        current = self.candidate("current")
        adaptor._archive.add(current.snapshot, score=0.1, valid_parent=True)
        adaptor._snapshot = current.snapshot
        adaptor._refinement_round = 20
        candidate = self.candidate(parent=current.snapshot)
        with patch.object(adaptor._archive, "choose", side_effect=AssertionError("No final archive search")):
            result = self.select(adaptor, candidate, status="error", score=None)
        self.assertEqual(adaptor.snapshot, current.snapshot)
        self.assertFalse(result.report.accepted)

    def test_child_counts_and_probabilities_match_official_policy(self):
        archive = SearchArchive(seed=42)
        archive.add(self.initial, score=0.5, valid_parent=True)
        candidate = self.candidate()
        archive.add(candidate.snapshot, score=0.5, valid_parent=True)
        with patch.object(archive.rng, "choices", return_value=["initial"]) as choices:
            archive.choose("score_child_prop")
        weights = choices.call_args.kwargs["weights"]
        self.assertAlmostEqual(weights[0], 0.5 * math.exp(-((1 / 8) ** 3)))
        self.assertEqual(weights[1], 0.5)

    def test_new_archive_uses_benchmark_seed_including_init_reset(self):
        import random

        context = MethodContext(self.root, 20, {}, seed=73)
        adaptor = HyperAgentsAdaptor(context)
        self.addCleanup(adaptor.close)
        expected = random.Random(73).getstate()
        self.assertEqual(adaptor._archive.rng.getstate(), expected)
        adaptor._archive.rng.random()
        with patch.object(adaptor, "_configure"), patch.object(adaptor, "_load_config"), \
                patch.object(adaptor, "_create_workspace", return_value=self.workspace), \
                patch("rsi_methods.HyperAgents.HyperAgentsAdaptor.initialize_agent",
                      return_value=self.initial):
            adaptor.init()
        self.assertEqual(adaptor._archive.rng.getstate(), expected)

    def test_archive_roundtrip_preserves_branch_patches_and_random_sequence(self):
        archive = SearchArchive(seed=5)
        archive.add(self.initial, score=1.0, valid_parent=True)
        candidate = self.candidate()
        archive.add(candidate.snapshot, score=0.2, valid_parent=True, artifact_dir=candidate.artifact_dir)
        payload = json.loads(json.dumps(archive.to_dict()))
        restored = SearchArchive.from_dict(payload, root=self.root)
        self.assertEqual(archive.nodes, restored.nodes)
        self.assertEqual(
            [archive.choose("score_child_prop") for _ in range(15)],
            [restored.choose("score_child_prop") for _ in range(15)],
        )
        candidate.patch_path.write_text("tampered")
        with self.assertRaisesRegex(RuntimeError, "archived lineage patches changed"):
            SearchArchive.from_dict(payload, root=self.root)

    def test_history_allowlist_keeps_training_and_excludes_holdout(self):
        archive = SearchArchive()
        candidate = self.candidate()
        for relative in ("evidence/result.json", "evaluation/result.json", "test/answers.json",
                         "holdout/result.json", "credentials.env", "archive_history/old.json"):
            path = candidate.artifact_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        archive.add(candidate.snapshot, score=0.2, valid_parent=True, artifact_dir=candidate.artifact_dir)
        destination = self.root / "history"
        archive.stage_history(destination)
        self.assertTrue((destination / "child/evidence/result.json").is_file())
        self.assertTrue((destination / "child/evaluation/result.json").is_file())
        self.assertFalse((destination / "child/test").exists())
        self.assertFalse((destination / "child/holdout").exists())
        self.assertFalse((destination / "child/credentials.env").exists())
        self.assertFalse((destination / "child/archive_history").exists())

    def test_method_checkpoint_restores_all_branches_and_rejects_archived_tampering(self):
        from rsi_methods.checkpoint import export_state, restore_state

        adaptor = self.adaptor()
        adaptor._configure(adaptor._load_config())
        (self.root / "method_config.toml").write_bytes(adaptor._config_snapshot)
        candidate = self.candidate()
        adaptor._archive.add(candidate.snapshot, score=0.2, valid_parent=True,
                             artifact_dir=candidate.artifact_dir)
        state = json.loads(json.dumps(export_state(adaptor, "hyperagents")))
        restored = HyperAgentsAdaptor()
        self.addCleanup(restored.close)
        restore_state(restored, "hyperagents", self.root, state)
        self.assertEqual(export_state(restored, "hyperagents"), state)
        self.assertEqual(restored._archive.nodes["child"]["snapshot"], candidate.snapshot)
        candidate.patch_path.write_text("tampered archived branch\n")
        with self.assertRaisesRegex(RuntimeError, "archived lineage patches changed"):
            restore_state(restored, "hyperagents", self.root, state)

    def test_legacy_checkpoint_is_not_silently_migrated(self):
        from rsi_methods.checkpoint import restore_state

        with self.assertRaisesRegex(RuntimeError, "predates archive restoration"):
            restore_state(self.adaptor(), "hyperagents", self.root, {})

    def test_copied_batch_resume_relocates_all_archived_branches(self):
        source = self.root / "original"
        workspace = source / "state/run_test"
        workspace.mkdir(parents=True)
        initial = CandidateSnapshot("initial", workspace, workspace, "root")
        task = source / "task_0001_test"
        artifact = task / "candidates/child"
        artifact.mkdir(parents=True)
        patch_file = artifact / "patch.diff"
        patch_file.write_text("patch fixture\n")
        child = with_lineage(initial, candidate_id="child", parent_id="initial",
                             lineage_patches=[patch_file])
        archive = SearchArchive(seed=3)
        archive.add(initial, score=1.0, valid_parent=True)
        archive.add(child, score=0.2, valid_parent=True, artifact_dir=artifact)
        (task / "search_archive.json").write_text(json.dumps({
            **archive.to_dict(), "active_candidate_id": "child", "refinement_round": 1,
        }))
        (workspace / "search_archive.json").write_text(json.dumps(archive.to_dict()))
        adaptor = HyperAgentsAdaptor(max_refinement_rounds=20)
        self.addCleanup(adaptor.close)
        (source / "method_config.toml").write_bytes(adaptor.config_path.read_bytes())
        target = self.root / "copied"
        shutil.copytree(source, target)
        relocated_initial = CandidateSnapshot(
            "initial", target / "state/run_test", target / "state/run_test", "root",
        )
        with patch("rsi_methods.HyperAgents.HyperAgentsAdaptor.initialize_agent",
                   return_value=relocated_initial):
            adaptor.resume(target, source_batch=source, before_task_index=2,
                           target_dir=target / "state")
        self.assertEqual(adaptor.snapshot.candidate_id, "child")
        self.assertEqual(adaptor._refinement_round, 1)
        self.assertEqual(adaptor.snapshot.lineage_patches,
                         (target / patch_file.relative_to(source),))
        self.assertEqual(adaptor._archive.nodes["initial"]["snapshot"], relocated_initial)
        self.assertEqual(adaptor._archive.nodes["child"]["artifact_dir"],
                         target / artifact.relative_to(source))
        self.assertEqual(archive.choose("score_child_prop").candidate_id,
                         adaptor._archive.choose("score_child_prop").candidate_id)

    def test_history_rejects_symlinked_artifact_directories(self):
        archive = SearchArchive()
        candidate = self.candidate()
        external = self.root / "holdout"
        external.mkdir()
        (external / "result.json").write_text('{"answer":"held out"}')
        (candidate.artifact_dir / "evaluation").symlink_to(external, target_is_directory=True)
        archive.add(candidate.snapshot, score=0.2, valid_parent=True,
                    artifact_dir=candidate.artifact_dir)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            archive.stage_history(self.root / "history")

    def test_zero_score_remains_eligible_and_sigmoid_does_not_overflow(self):
        from .archive_utils import select_parent_from_candidates

        archive = SearchArchive()
        archive.add(self.initial, score=0.0, valid_parent=True)
        self.assertEqual(archive.choose("score_child_prop"), self.initial)
        self.assertIn(select_parent_from_candidates({"bad": -1000.0, "good": 1000.0},
                                                   "score_prop"), {"bad", "good"})

    def test_unmeasured_archived_parent_does_not_inherit_current_candidate_score(self):
        from benchmark.metrics import post_refinement_score

        record = {
            "relative_score": 0.3,
            "selection": {
                "status": "committed", "committed_candidate_id": "older_branch",
                "stages": [{
                    "status": "committed", "committed_candidate_id": "older_branch",
                    "evaluations": [{"candidate_id": "new_child", "status": "ok", "relative_score": 0.8}],
                }],
            },
        }
        self.assertIsNone(post_refinement_score(record))

    def test_direct_evidence_uploads_staged_screenshots(self):
        from . import lifecycle

        image = self.root / "step.png"
        image.write_bytes(b"image fixture")
        evidence = {"cases": [{"feedback": {"trajectory": [
            {"image_records": [{"path": str(image)}]},
        ]}}]}
        agent = SimpleNamespace(container=Mock(), snapshot=self.initial)
        with patch.object(lifecycle, "copy_to_container") as copy, \
                patch.object(lifecycle, "log_container_output"):
            lifecycle._stage_direct_evidence(agent, evidence, self.root / "evidence_task", metadata={})
        staged = self.root / "evidence_task/evidence/artifacts"
        self.assertTrue(any(staged == call.args[1] and call.args[2].endswith("/artifacts")
                            for call in copy.call_args_list))

    def test_tool_and_meta_edits_survive_real_git_reconstruction(self):
        from . import archive_utils, lifecycle, runtime_safety

        repo = self.workspace
        for relative in ("agent/llm_withtools.py", "agent/tools/bash.py"):
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runtime_safety._OVERLAY / relative, target)
        (repo / "meta_agent.py").write_text("original = True\n")

        def git(*args):
            return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout

        git("init")
        git("add", ".")
        git("-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "seed")
        root_commit = git("rev-parse", "HEAD").decode().strip()
        for relative in ("meta_agent.py", "agent/llm_withtools.py", "agent/tools/bash.py"):
            path = repo / relative
            path.write_text(path.read_text() + "\nevolved = True\n")
        patch_file = self.root / "tools.diff"
        patch_file.write_bytes(git("diff"))
        git("reset", "--hard", root_commit)
        snapshot = CandidateSnapshot("evolved", repo, repo, root_commit, "initial", (patch_file,))

        class LocalContainer:
            def exec_run(inner, command, *, workdir, **kwargs):
                translated = [arg.replace(f"/{lifecycle.REPO_NAME}/", f"{repo}/") for arg in command]
                output = subprocess.run(translated, cwd=repo, capture_output=True)
                return SimpleNamespace(exit_code=output.returncode, output=output.stdout + output.stderr)

        def copy_file(container, source, destination):
            shutil.copy2(source, repo / Path(destination).relative_to(f"/{lifecycle.REPO_NAME}"))

        agent = SimpleNamespace(container=LocalContainer(), snapshot=snapshot, commit_hash=root_commit)
        with patch.object(archive_utils, "copy_to_container", side_effect=copy_file), \
                patch.object(lifecycle, "log_container_output"):
            lifecycle.reconstruct_agent(agent, snapshot)
            # A second rebuild exercises reset-to-seed and reapplication as well.
            lifecycle.reconstruct_agent(agent, snapshot)
        for relative in ("meta_agent.py", "agent/llm_withtools.py", "agent/tools/bash.py"):
            self.assertIn("evolved = True", (repo / relative).read_text())
        self.assertEqual(git("status", "--porcelain"), b"")


if __name__ == "__main__":
    unittest.main()
