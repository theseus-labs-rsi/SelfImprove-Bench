"""Offline checks for the published main configurations and initial CLI."""

import json
import sys
import tomllib
import unittest
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from benchmark.__main__ import main
from benchmark.config import ReplayConfig, load_config
from benchmark.methods import BUILTIN_ADAPTERS, method_identity
from benchmark.tasks import GAMES as GAME_MODULES, build_manifest, task_fingerprint
from benches.gameworld.asset_paths import ASSETS_ROOT


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs' / 'main'
METHODS = ('ace', 'ahe', 'hyperagents')
GAMES = ('farm', 'combat', 'sweepmine', 'hero', 'minecraft', 'astray')


class MainConfigurationsTest(unittest.TestCase):
    def setUp(self):
        self.generators = ExitStack()
        self.addCleanup(self.generators.close)
        # Repeated methods/initial checks use identical generated tasks.
        for module in GAME_MODULES.values():
            original = module.generate
            cache = {}

            def cached(*args, _original=original, _cache=cache, **kwargs):
                key = json.dumps([args, kwargs], sort_keys=True, default=asdict)
                if key not in _cache:
                    _cache[key] = _original(*args, **kwargs)
                return deepcopy(_cache[key])

            self.generators.enter_context(patch.object(module, 'generate', cached))

    def test_text_manifests_and_initial_holdout_identity(self):
        self._check_manifests(('farm', 'combat', 'sweepmine', 'hero'))

    def test_gameworld_manifests_and_initial_holdout_identity(self):
        self._check_manifests(('minecraft', 'astray'))

    def _check_manifests(self, games):
        self.assertEqual(len(list(CONFIGS.glob('*.json'))), 18)
        signatures = {}
        for game in games:
            for method in METHODS:
                with self.subTest(game=game, method=method):
                    config = load_config(CONFIGS / f'{game}_{method}.json')
                    print(f'\nChecking {game}/{method}', flush=True)
                    self.assertEqual(config.agent.provider, 'rsi')
                    self.assertEqual(config.agent.adapter, BUILTIN_ADAPTERS[method])
                    self.assertEqual(method_identity(config.agent), method)
                    self.assertEqual(config.output_dir, f'outputs/main/{game}/{method}')
                    manifest = build_manifest(config)
                    legacy = replace(config, agent=replace(config.agent, provider=method, adapter=None))
                    legacy_manifest = build_manifest(legacy)
                    self.assertEqual(manifest.training, legacy_manifest.training)
                    self.assertEqual(manifest.holdout, legacy_manifest.holdout)
                    normal = [task for task in manifest.training if not task.replay]
                    replay = [task for task in manifest.training if task.replay]
                    self.assertEqual(len(normal), 20)
                    self.assertEqual(len(replay), 4)
                    self.assertEqual([task.replay_of for task in replay],
                                     [normal[i].run_id for i in (1, 6, 11, 16)])
                    self.assertEqual([task.task for task in replay],
                                     [normal[i].task for i in (1, 6, 11, 16)])
                    initial = build_manifest(replace(config, schedule=(), replay=ReplayConfig()))
                    self.assertFalse(initial.training)
                    self.assertEqual(manifest.holdout, initial.holdout)
                    signature = tuple(task_fingerprint(task.task) for task in manifest.holdout)
                    if game in signatures and game not in ('minecraft', 'astray'):
                        self.assertEqual(signature, signatures[game])
                    signatures[game] = signature
                    train_keys = {(task.game, task_fingerprint(task.task)) for task in normal}
                    eval_keys = {(task.game, task_fingerprint(task.task)) for task in manifest.holdout}
                    self.assertFalse(train_keys & eval_keys)
                    if game in ('minecraft', 'astray'):
                        self.assertEqual(len(manifest.holdout), 10)
                        own = '18_minecraft-clone-glm' if game == 'minecraft' else '03_astray'
                        self.assertTrue(all(t.task['game_id'] == own for t in manifest.holdout[:5]))
                        self.assertTrue(all(t.task['game_id'] != own for t in manifest.holdout[5:]))
                        for task in (*normal, *manifest.holdout):
                            assets = Path(task.task['task_assets_root'])
                            self.assertTrue(assets.is_dir())
                            directory = 'minecraft' if task.task['game_id'].startswith('18_') else 'astray'
                            self.assertTrue((assets / directory / (task.task['game_task_id'] + '.yaml')).is_file(),
                                            task.task['game_task_id'])
                    else:
                        self.assertEqual(len(manifest.holdout), 60)
                        hard = 4 if game in ('hero', 'sweepmine') else 5
                        self.assertEqual(config.holdout[1].difficulty, hard)

    def test_farm_catalog_supports_twenty_unique_hard_tasks(self):
        catalog = json.loads((ROOT / 'benches/farm/crop_data.json').read_text())
        self.assertEqual(len(catalog), 18)
        self.assertEqual(catalog['carrot']['seed_price'], 40)
        self.assertEqual(catalog['carrot']['maturity_days'], 6)
        self.assertEqual(catalog['carrot']['fruit_price'], 90)

    def test_method_backends(self):
        for method in ('ACE', 'AHE', 'HyperAgents'):
            with self.subTest(method=method):
                config = tomllib.loads((ROOT / 'rsi_methods' / method / 'config.toml').read_text())
                backend = config['backend']['openai']
                self.assertEqual(backend['api_key_env'], 'OPENAI_API_KEY')
                self.assertEqual(backend['base_url_env'], 'OPENAI_BASE_URL')
                for section in ('agent', 'reflector', 'curator', 'meta_agent', 'task_agent'):
                    if section in config:
                        self.assertEqual(config[section].get('backend', 'openai'), 'openai')
                        self.assertEqual(config[section]['reasoning_effort'], 'low')
        self.assertTrue((ASSETS_ROOT / 'core/code/catalog/models/gpt-5.6-sol.yaml').is_file())

    def test_initial_cli_dispatch(self):
        for method in METHODS:
            config_path = str(CONFIGS / f'farm_{method}.json')
            with self.subTest(method=method), patch.object(sys, 'argv',
                    ['benchmark', config_path, '--initial']), \
                    patch('benchmark.__main__.run_benchmark') as run:
                main()
                config = run.call_args.args[0]
                self.assertEqual(config.agent.provider, 'rsi')
                self.assertEqual(method_identity(config.agent), method)
                self.assertFalse(config.schedule)
                self.assertFalse(config.replay.enabled)
                self.assertTrue(config.output_dir.endswith('_initial'))

    def test_normal_cli_keeps_training(self):
        path = str(CONFIGS / 'minecraft_ace.json')
        with patch.object(sys, 'argv', ['benchmark', path]), \
                patch('benchmark.__main__.run_benchmark') as run:
            main()
            self.assertEqual(sum(item.count for item in run.call_args.args[0].schedule), 20)

    def test_initial_rejects_evolved_workspace(self):
        path = str(CONFIGS / 'farm_ace.json')
        with patch.object(sys, 'argv', ['benchmark', path, '--initial', '--target-dir', 'other']), \
                patch('benchmark.__main__.run_benchmark') as run, \
                self.assertRaises(SystemExit) as exc:
            main()
        self.assertEqual(exc.exception.code, 2)
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
