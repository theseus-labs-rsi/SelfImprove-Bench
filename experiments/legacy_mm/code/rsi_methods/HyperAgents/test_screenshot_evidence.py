"""Regression tests for GameWorld visual evidence passed to HyperAgents MetaAgent."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsi_methods.HyperAgents.lifecycle import _stage_feedback_screenshots


class ScreenshotEvidenceTests(unittest.TestCase):
    def test_pinned_source_receives_versioned_hook_once(self):
        from rsi_methods.HyperAgents.archive_utils import _install_visual_evidence_hook
        with tempfile.TemporaryDirectory() as directory:
            meta = Path(directory) / 'meta_agent.py'
            meta.write_text('class MetaAgent:\n'
                            '    def forward(self, eval_path):\n'
                            '        instruction = "improve"\n'
                            '        new_msg_history = chat_with_agent(instruction)\n'
                            '        return new_msg_history\n')
            _install_visual_evidence_hook(directory)
            self.assertTrue((Path(directory) / 'visual_evidence.py').is_file())
            first = meta.read_text()
            self.assertIn('visual_observations = inspect_game_screenshots(self, eval_path)', first)
            compile(first, str(meta), 'exec')
            _install_visual_evidence_hook(directory)
            self.assertEqual(meta.read_text(), first)

    def test_unexpected_upstream_meta_agent_fails_before_experiments(self):
        from rsi_methods.HyperAgents.archive_utils import _install_visual_evidence_hook
        with tempfile.TemporaryDirectory() as directory:
            meta = Path(directory) / 'meta_agent.py'
            meta.write_text('class MetaAgent:\n    pass\n')
            with self.assertRaisesRegex(RuntimeError, 'Cannot install visual evidence'):
                _install_visual_evidence_hook(directory)

    def test_feedback_image_is_bundled_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'original.png'
            source.write_bytes(b'\x89PNG\r\n\x1a\nsource')
            case = {'feedback': {'trajectory': [
                {'step_index': 1, 'image_records': [{'path': str(source)}]},
                {'step_index': 2, 'image_records': [{'path': str(source)}]},
            ]}}
            original = {'cases': [case]}
            staged = copy.deepcopy(original)
            paths = _stage_feedback_screenshots(staged, root / 'stage')
            self.assertEqual(len(paths), 2)
            self.assertEqual(len(set(paths)), 2)
            for step, path in zip(staged['cases'][0]['feedback']['trajectory'], paths):
                self.assertEqual(step['image_records'][0]['path'], path)
                self.assertEqual((root / 'stage' / path).read_bytes(), source.read_bytes())
            self.assertEqual([p['path'] for p in staged['cases'][0]['artifacts']['game_screenshots']], paths)
            self.assertEqual(original['cases'][0], case)
            self.assertEqual(case['feedback']['trajectory'][0]['image_records'][0]['path'], str(source))

    def test_missing_feedback_image_fails_instead_of_silent_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / 'missing.png')
            with self.assertRaises(FileNotFoundError):
                _stage_feedback_screenshots(
                    {'cases': [{'feedback': {'trajectory': [{'image_records': [{'path': missing}]}]}}]},
                    Path(directory) / 'stage',
                )

    def test_parent_owned_hook_sends_multimodal_request(self):
        source = Path(__file__).resolve().parent / 'hyperagents'
        sys.path.insert(0, str(source))
        from rsi_methods.HyperAgents import visual_evidence
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'artifacts' / 'bench_screenshots' / 'step.png'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'\x89PNG\r\n\x1a\nsource')
            (root / 'result.json').write_text(json.dumps({'cases': [{'feedback': {
                'trajectory': [{'image_records': [{'path': 'artifacts/bench_screenshots/step.png'}]}],
            }}]}))
            instance = type('Agent', (), {})()
            instance.model = 'test'
            instance.log = lambda *_: None
            instance.reasoning_effort = 'high'
            instance.backend = 'openai'
            instance.api_mode = 'responses'
            instance.api_key_env = 'OPENAI_API_KEY'
            instance.base_url_env = 'OPENAI_BASE_URL'
            instance.logger_manager = type('Logger', (), {'log_file': str(root / 'chat.log')})()
            with patch.object(visual_evidence, 'chat_with_agent', return_value=[{
                'role': 'assistant', 'text': 'Visible tree.', 'usage': {'total_tokens': 11},
            }]) as chat:
                self.assertEqual(visual_evidence.inspect_game_screenshots(instance, root), 'Visible tree.')
            self.assertEqual(chat.call_args.kwargs['image_paths'], [str(image)])
            self.assertEqual(json.loads((root / 'screenshot_analysis.json').read_text())['usage']['total_tokens'], 11)

    def test_meta_inspection_sends_image_paths_and_persists_usage(self):
        source = Path(__file__).resolve().parent / 'hyperagents'
        sys.path.insert(0, str(source))
        from meta_agent import MetaAgent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = 'artifacts/bench_screenshots/step.png'
            image = root / path
            image.parent.mkdir(parents=True)
            image.write_bytes(b'\x89PNG\r\n\x1a\nsource')
            (root / 'result.json').write_text(json.dumps({'cases': [{'feedback': {
                'trajectory': [{'image_records': [{'path': path}]}],
            }}]}), encoding='utf-8')
            instance = object.__new__(MetaAgent)
            instance.model = 'test'
            instance.log = lambda *_: None
            instance.reasoning_effort = 'high'
            instance.backend = 'openai'
            instance.api_mode = 'responses'
            instance.api_key_env = 'OPENAI_API_KEY'
            instance.base_url_env = 'OPENAI_BASE_URL'
            instance.logger_manager = type('Logger', (), {'log_file': str(root / 'chat.log')})()
            usage = {'total_tokens': 15, 'input_tokens': 10, 'output_tokens': 5}
            with patch('meta_agent.chat_with_agent', return_value=[{
                'role': 'assistant', 'text': 'The player moves to the left.', 'usage': usage,
            }]) as chat:
                observation = instance._inspect_game_screenshots(str(root))
            self.assertIn('moves to the left', observation)
            self.assertEqual(chat.call_args.kwargs['image_paths'], [str(image.resolve())])
            self.assertEqual(chat.call_args.kwargs['tools_available'], [])
            analysis = json.loads((root / 'screenshot_analysis.json').read_text())
            self.assertEqual(analysis['usage'], usage)
            self.assertEqual(analysis['image_paths'], [path])
            (root / 'result.json').write_text(json.dumps({'cases': []}), encoding='utf-8')
            with patch('meta_agent.chat_with_agent') as chat:
                self.assertEqual(instance._inspect_game_screenshots(str(root)), '')
                chat.assert_not_called()

    def test_visual_observation_is_forwarded_to_evolution_prompt(self):
        source = Path(__file__).resolve().parent / 'hyperagents'
        sys.path.insert(0, str(source))
        from meta_agent import MetaAgent
        instance = object.__new__(MetaAgent)
        instance.model = 'test'
        instance.log = lambda *_: None
        instance.reasoning_effort = 'high'
        instance.backend = 'openai'
        instance.api_mode = 'responses'
        instance.api_key_env = 'OPENAI_API_KEY'
        instance.base_url_env = 'OPENAI_BASE_URL'
        with patch.object(instance, '_inspect_game_screenshots', return_value='Visible player beside a tree.') as inspect, \
                patch('meta_agent.chat_with_agent', return_value=[{'role': 'assistant', 'text': 'ok'}]) as chat:
            instance.forward('/sample/repo', '/sample/evidence', iterations_left=1)
        inspect.assert_called_once_with('/sample/evidence')
        self.assertIn('Visible player beside a tree.', chat.call_args.args[0])
        self.assertEqual(chat.call_args.kwargs['tools_available'], 'all')

    def test_visual_and_evolution_attempts_are_both_counted(self):
        from benchmark.token_ledger import hyperagents_usage_report
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / 'meta_agent_history.json'
            history.write_text(json.dumps([{'role': 'assistant', 'text': 'done'}]))
            ledger = history.with_name('model_attempts.jsonl')
            ledger.write_text('\n'.join(json.dumps({'status': 'completed', 'usage': {
                'input_tokens': tokens - 1, 'output_tokens': 1, 'total_tokens': tokens,
                'cached_input_tokens': 0, 'reasoning_tokens': 0,
            }}) for tokens in (17, 29)) + '\n')
            report = hyperagents_usage_report(history)
            self.assertTrue(report['complete'])
            self.assertEqual(report['total_tokens'], 46)
            self.assertEqual(len(report['calls']), 2)

    def test_meta_inspection_refuses_missing_or_escaped_image(self):
        source = Path(__file__).resolve().parent / 'hyperagents'
        sys.path.insert(0, str(source))
        from meta_agent import MetaAgent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'result.json').write_text(json.dumps({'cases': [{'feedback': {
                'trajectory': [{'image_records': [{'path': '../../outside.png'}]}],
            }}]}), encoding='utf-8')
            instance = object.__new__(MetaAgent)
            with patch('meta_agent.chat_with_agent') as chat:
                with self.assertRaises(FileNotFoundError):
                    instance._inspect_game_screenshots(str(root))
                chat.assert_not_called()


if __name__ == '__main__':
    unittest.main()
