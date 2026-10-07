"""Offline interruption/restart tests; never send model or Docker requests."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark import resume_state, runner
from benchmark.config import config_from_mapping
from benchmark.rsi_session import RSIBatchSession
from benchmark.tasks import build_manifest


class MethodStateTests(unittest.TestCase):
    def test_ace_state_restore_after_uncommitted_mutation(self):
        from rsi_methods.ACE.ACEAdaptor import ACEAdaptor, CONFIG_PATH
        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-not-used"}), tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = base / 'ace.toml'
            config.write_text(CONFIG_PATH.read_text().replace('workspace_root = "../../../outputs/ace"',
                                      f'workspace_root = "{base / "ace_workspaces"}"'))
            first = ACEAdaptor()
            first.config_path = config
            session = RSIBatchSession('ace', base / 'rsi', task_count=2, adaptor=first)
            session.adaptor._playbook = 'selected'
            session.adaptor._version = 1
            session.adaptor._refinement_round = 1
            session.adaptor._next_global_id = 42
            saved = resume_state.export(session)
            session.adaptor._playbook = 'uncommitted'
            session.adaptor._persist_playbook(session.adaptor.workspace_dir, 'uncommitted')
            fresh = ACEAdaptor()
            fresh.config_path = config
            recovered = RSIBatchSession('ace', base / 'rsi', task_count=2,
                                        batch_id=session.batch_id, adaptor=fresh, resume=True)
            resume_state.restore(recovered, saved)
            self.assertEqual((fresh.playbook, fresh.version, fresh._refinement_round,
                              fresh._next_global_id), ('selected', 1, 1, 42))
            self.assertEqual((fresh.workspace_dir / 'playbook.txt').read_text(), 'selected')
            session.close()
            recovered.close()

    def test_ahe_selected_workspace_and_iteration_recovered(self):
        from rsi_methods.AHE.AHEAdaptor import AHEAdaptor
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = AHEAdaptor(run_dir=base / 'rsi' / 'ahe' / 'batch' / 'state')
            session = RSIBatchSession('ahe', base / 'rsi', task_count=2, batch_id='batch', adaptor=first)
            first._refinement_round = 1
            saved = resume_state.export(session)
            first.state.current_workspace = first.state.evolve_agent_dir
            recovered_adaptor = AHEAdaptor(run_dir=first.run_dir)
            recovered = RSIBatchSession('ahe', base / 'rsi', task_count=2,
                                        batch_id='batch', adaptor=recovered_adaptor, resume=True)
            resume_state.restore(recovered, saved)
            self.assertEqual(recovered_adaptor.state.current_workspace, session.artifact_dir / 'state' / 'workspace')
            self.assertEqual(recovered_adaptor._refinement_round, 1)
            recovered.close()
            session.close()

    def test_hyperagents_real_initial_snapshot_without_docker(self):
        from rsi_methods.HyperAgents.HyperAgentsAdaptor import HyperAgentsAdaptor
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = RSIBatchSession('hyperagents', base / 'rsi', task_count=2)
            initial = resume_state.export(first)
            first.close()
            recovered = RSIBatchSession('hyperagents', base / 'rsi', task_count=2,
                                        batch_id=first.batch_id, adaptor=HyperAgentsAdaptor(
                                            run_dir=first.artifact_dir / 'state',
                                            max_refinement_rounds=2), resume=True)
            resume_state.restore(recovered, initial)
            self.assertEqual(recovered.adaptor.snapshot.root_commit, first.adaptor.snapshot.root_commit)
            self.assertEqual(recovered.adaptor.snapshot.workspace, first.adaptor.snapshot.workspace)
            recovered.close()

    def test_hyperagents_snapshot_restored_without_docker(self):
        from rsi_methods.HyperAgents.HyperAgentsAdaptor import HyperAgentsAdaptor, CONFIG_PATH
        from rsi_methods.HyperAgents.candidate import CandidateSnapshot
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / 'rsi' / 'hyperagents' / 'batch'
            root.mkdir(parents=True)
            state = root / 'state' / 'run_x'
            workspace = state / 'workspace'
            workspace.mkdir(parents=True)
            patchfile = root / 'patch.diff'
            patchfile.write_text('a patch')
            (root / 'method_config.toml').write_bytes(CONFIG_PATH.read_bytes())
            (root / 'batch_manifest.json').write_text(json.dumps({'method': 'hyperagents', 'batch_id': 'batch', 'task_count': 2}))
            first = HyperAgentsAdaptor(run_dir=root / 'state', max_refinement_rounds=2)
            first.workspace_dir = state
            first._snapshot = CandidateSnapshot('winner', state, workspace, 'rootsha', 'initial', (patchfile,))
            first._refinement_round = 1
            fake_session = SimpleNamespace(method='hyperagents', batch_id='batch', adaptor=first, artifact_dir=root,
                                           _selection_stage=1)
            saved = resume_state.export(fake_session)
            second = HyperAgentsAdaptor(run_dir=root / 'state', max_refinement_rounds=2)
            recovered = RSIBatchSession('hyperagents', base / 'rsi', task_count=2,
                                        batch_id='batch', adaptor=second, resume=True)
            resume_state.restore(recovered, saved)
            self.assertEqual((second.snapshot.candidate_id, second._refinement_round,
                              second.snapshot.lineage_patches), ('winner', 1, (patchfile,)))
            recovered.close()
            with self.assertRaisesRegex(RuntimeError, 'patches changed'):
                patchfile.write_text('changed')
                resume_state.restore(recovered, saved)


class CandidateFailureTests(unittest.TestCase):
    def test_candidate_api_error_never_commits_selection(self):
        from benchmark.game_adapters import EvaluationOutcome
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = build_manifest(config_from_mapping({
                'seed': 4, 'schedule': [{'game': 'farm', 'difficulty': 1, 'count': 1}],
                'holdout': [{'game': 'farm', 'difficulty': 1, 'count': 1}],
                'holdout_tasks_per_difficulty': 1, 'output_dir': str(root),
                'agent': {'provider': 'ace'}})).training[0]
            candidate = SimpleNamespace(candidate_id='c1', artifact_dir=root / 'candidate', agent=object())
            batch = SimpleNamespace(candidates=[candidate])
            class Controller:
                def request_for_agent(self, agent): return lambda *a, **kw: None
                class Session:
                    def select(self, *args): raise AssertionError('must not select on API error')
                session = Session()
            with patch.object(runner, '_evaluate', return_value=EvaluationOutcome({
                 'status': 'error', 'error_type': 'RateLimitError', 'error': 'no credit', 'llm_metrics': {}})):
                with self.assertRaisesRegex(RuntimeError, 'Candidate validation failed'):
                    runner._evaluate_candidates(task, batch, SimpleNamespace(),
                        controller=Controller(), output_dir=root, ordinal=1, stage_index=0)
            self.assertTrue((candidate.artifact_dir / 'evaluation' / 'result.json').is_file())


class RunnerResumeTests(unittest.TestCase):
    def test_legacy_output_is_not_modified_or_silently_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'legacy'
            output.mkdir()
            (output / 'result.json').write_text('{"old": true}')
            config = config_from_mapping({
                'seed': 8, 'schedule': [{'game': 'farm', 'difficulty': 1, 'count': 1}],
                'holdout': [{'game': 'farm', 'difficulty': 1, 'count': 1}],
                'holdout_tasks_per_difficulty': 1,
                'agent': {'provider': 'ace'}, 'output_dir': str(output)})
            with self.assertRaisesRegex(RuntimeError, 'No initial checkpoint'):
                runner.run_benchmark(config, resume=True)
            self.assertEqual(sorted(p.name for p in output.iterdir()), ['result.json'])

    def test_resume_skips_committed_and_retries_only_failed_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = config_from_mapping({
                'seed': 11, 'schedule': [{'game': 'farm', 'difficulty': 1, 'count': 3}],
                'holdout': [{'game': 'farm', 'difficulty': 1, 'count': 1}],
                'holdout_tasks_per_difficulty': 1, 'agent': {'provider': 'ace'},
                'output_dir': str(Path(tmp) / 'out')})
            calls = []
            export = lambda session: {'method': 'ace', 'batch_id': session.batch_id, 'round': session.round}
            def restore(session, state):
                session.round = state['round']
            class FakeSession:
                method = 'ace'
                artifact_dir = Path(tmp) / 'rsi'
                def __init__(self, batch_id=None):
                    self.batch_id = batch_id or 'batch'
                    self.round = 0
                def close(self): pass
            class FakeController:
                def __init__(self, cfg, output_dir, count, *, batch_id=None, resume=False):
                    self.session = FakeSession(batch_id)
                def close(self): pass
            failed = [False]
            def run(task, *, controller, ordinal, output_dir):
                calls.append((task.phase, ordinal, controller.session.round))
                bad = task.phase == 'train' and ordinal == 2 and not failed[0]
                if bad: failed[0] = True
                else: controller.session.round += 1
                record = {'status': 'error' if bad else 'ok', 'run_id': task.run_id,
                          'phase': task.phase, 'game': task.game, 'difficulty': task.difficulty,
                          'group_index': task.group_index, 'replay': task.replay, 'replay_of': task.replay_of, 'replay_gap': task.replay_gap,
                          'relative_score': None if bad else 0.8, 'success': not bad,
                          'selection': {'status': 'committed'},
                          'token_coverage': {'complete': not bad}, 'total_tokens': None if bad else 10}
                runner._write_json(runner._run_path(output_dir, task.phase, ordinal, task), record)
                return record
            with (patch.object(runner, 'AgentController', FakeController),
                  patch.object(runner, '_run_one', run),
                  patch.object(resume_state, 'export', export),
                  patch.object(resume_state, 'restore', restore)):
                first = runner.run_benchmark(config)
                self.assertEqual(first['summary']['attempted_training_runs'], 2)
                self.assertEqual(first['summary']['completed_training_runs'], 1)
                self.assertEqual(first['summary']['committed_training_runs'], 1)
                second = runner.run_benchmark(config, resume=True)
                self.assertEqual(second['summary']['completed_training_runs'], 3)
                self.assertEqual(second['summary']['completed_holdout_runs'], 1)
                self.assertEqual(calls, [('train', 1, 0), ('train', 2, 1),
                                         ('train', 2, 1), ('train', 3, 2), ('holdout', 1, 3)])
                self.assertFalse(second['summary']['full_token_coverage'])
                self.assertEqual(second['summary']['abandoned_attempts'], 1)
                # SIGKILL immediately after a durable checkpoint but before
                # cleanup must not cause the committed holdout to run twice.
                last = json.loads((Path(config.output_dir) / 'checkpoints' / 'holdout_0001.json').read_text())
                resume_state.atomic_json(Path(config.output_dir) / 'inflight.json', {
                    'phase': 'holdout', 'ordinal': 1, 'run_id': last['run_id'],
                    'parent_digest': last['parent_digest']})
                runner.run_benchmark(config, resume=True)
                self.assertEqual(calls, [('train', 1, 0), ('train', 2, 1),
                                         ('train', 2, 1), ('train', 3, 2), ('holdout', 1, 3)])
                self.assertFalse((Path(config.output_dir) / 'inflight.json').exists())
                with self.assertRaisesRegex(RuntimeError, 'not empty'):
                    runner.run_benchmark(config)


    def test_pause_after_one_committed_training_run_can_resume_without_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = config_from_mapping({
                'seed': 42, 'schedule': [{'game': 'farm', 'difficulty': 1, 'count': 3}],
                'holdout': [{'game': 'farm', 'difficulty': 1, 'count': 1}],
                'holdout_tasks_per_difficulty': 1, 'agent': {'provider': 'ace'},
                'output_dir': str(Path(tmp) / 'out')})
            calls = []
            class FakeSession:
                method = 'ace'
                batch_id = 'batch'
                artifact_dir = Path(tmp) / 'rsi'
                def __init__(self):
                    self.round = 0
                def close(self): pass
            class FakeController:
                def __init__(self, cfg, output_dir, count, *, batch_id=None, resume=False):
                    self.session = FakeSession()
                def close(self): pass
            def run(task, *, controller, ordinal, output_dir):
                calls.append((task.phase, ordinal, controller.session.round))
                if task.phase == 'train':
                    controller.session.round += 1
                record = {'status': 'ok', 'run_id': task.run_id, 'phase': task.phase,
                          'game': task.game, 'difficulty': task.difficulty,
                          'group_index': task.group_index, 'replay': task.replay,
                          'replay_of': task.replay_of, 'replay_gap': task.replay_gap,
                          'relative_score': 0.5, 'success': True,
                          'selection': {'status': 'committed'},
                          'token_coverage': {'complete': True}, 'total_tokens': 10}
                runner._write_json(runner._run_path(output_dir, task.phase, ordinal, task), record)
                return record
            def restore(session, state):
                session.round = state['round']
            with (patch.object(runner, 'AgentController', FakeController),
                  patch.object(runner, '_run_one', run),
                  patch.object(resume_state, 'export', lambda session: {
                      'method': 'ace', 'batch_id': session.batch_id, 'round': session.round}),
                  patch.object(resume_state, 'restore', restore)):
                first = runner.run_benchmark(config, max_new_train_runs=1)
                self.assertTrue(first['summary']['paused_after_checkpoint'])
                self.assertEqual(first['summary']['committed_training_runs'], 1)
                self.assertEqual(first['summary']['scheduled_training_runs'], 3)
                self.assertEqual(first['summary']['attempted_holdout_runs'], 0)
                second = runner.run_benchmark(config, resume=True, max_new_train_runs=1)
                self.assertEqual(second['summary']['committed_training_runs'], 2)
                final = runner.run_benchmark(config, resume=True)
                self.assertFalse(final['summary']['paused_after_checkpoint'])
                self.assertEqual(final['summary']['committed_training_runs'], 3)
                self.assertEqual(final['summary']['committed_holdout_runs'], 1)
                self.assertEqual(calls, [('train', 1, 0), ('train', 2, 1),
                                         ('train', 3, 2), ('holdout', 1, 3)])


if __name__ == '__main__':
    unittest.main()
