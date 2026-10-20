"""Offline checks for the three published training and evaluation frameworks."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from analyze.__main__ import _build_parser
from benchmark.client import RSI_PROVIDERS as CLIENT_METHODS
from benchmark.config import AgentConfig, SUPPORTED_PROVIDERS, config_from_mapping, load_config
from benchmark.rsi_session import RSI_PROVIDERS, RSIBatchSession
from benchmark.runner import RSI_METHODS
from benchmark.methods import BUILTIN_ADAPTERS, method_identity


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"ace", "ahe", "hyperagents"}
REMOVED_METHODS = ("hyperdsh", "baseline", "self_harness", "codex", "claude", "deepseek", "openai")


class FrameworkScopeTest(unittest.TestCase):
    def test_framework_registries_agree(self):
        for registry in (SUPPORTED_PROVIDERS, RSI_PROVIDERS, RSI_METHODS, CLIENT_METHODS):
            self.assertEqual(set(registry), METHODS | {"rsi"})
        self.assertEqual(AgentConfig().provider, "ace")

    def test_all_main_configs_load(self):
        paths = sorted((ROOT / "configs/main").glob("*.json"))
        self.assertEqual(len(paths), 18)
        for path in paths:
            with self.subTest(config=path.name):
                config = load_config(path)
                self.assertEqual(config.agent.provider, "rsi")
                self.assertIn(method_identity(config.agent), METHODS)
                self.assertEqual(config.agent.adapter, BUILTIN_ADAPTERS[method_identity(config.agent)])
                self.assertEqual(sum(item.count for item in config.schedule), 20)

    def test_removed_providers_are_rejected(self):
        for method in REMOVED_METHODS:
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "agent.provider"):
                config_from_mapping({
                    "schedule": [{"game": "farm", "difficulty": 3, "count": 1}],
                    "holdout": [{"game": "farm", "difficulty": 3}],
                    "agent": {"provider": method},
                })

    def test_evaluation_only_configs_use_same_frameworks(self):
        for method in METHODS:
            with self.subTest(method=method):
                config = config_from_mapping({
                    "schedule": [],
                    "holdout": [{"game": "farm", "difficulty": 3}],
                    "agent": {"provider": method},
                })
                self.assertEqual(config.agent.provider, method)
                self.assertFalse(config.schedule)

    def test_sessions_initialize_only_supported_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            for method in METHODS:
                with self.subTest(method=method):
                    adaptor = SimpleNamespace(init=Mock(), close=Mock())
                    session = RSIBatchSession(method, directory, task_count=1, adaptor=adaptor)
                    adaptor.init.assert_called_once_with()
                    self.assertEqual(session.method, method)
                    session.close()
            for method in REMOVED_METHODS:
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "Unsupported RSI method"):
                    RSIBatchSession(method, directory, task_count=1)

    def test_analysis_cli_has_only_three_methods(self):
        parser = _build_parser()
        for method in ("ace", "ahe", "hyper"):
            with self.subTest(method=method):
                self.assertEqual(parser.parse_args([method, "run"]).command, method)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parser.parse_args(["hyperdsh", "run"])
        self.assertEqual(error.exception.code, 2)

    def test_all_frameworks_keep_paired_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            for method in METHODS:
                with self.subTest(method=method):
                    adaptor = SimpleNamespace(
                        init=Mock(), close=Mock(),
                        select=Mock(return_value=SimpleNamespace(status="unchanged")),
                    )
                    session = RSIBatchSession(method, directory, task_count=1, adaptor=adaptor)
                    session._pending_batch = SimpleNamespace(
                        candidates=[SimpleNamespace(
                            candidate_id="candidate", agent=SimpleNamespace(act=Mock()),
                        )],
                        work_dir=session.artifact_dir / "candidate",
                    )
                    selection = session.select(
                        {"status": "ok", "relative_score": 0.8},
                        {"candidate": {"status": "ok", "relative_score": 0.2}},
                    )
                    self.assertEqual(selection.status, "unchanged")
                    self.assertEqual(adaptor.select.call_args.kwargs["selection_mode"], "paired")
                    self.assertIsNone(session._pending_batch)
                    session.close()

    def test_removed_framework_and_configs_are_not_published(self):
        self.assertFalse(any((ROOT / "rsi_methods/HyperDSH").rglob("*.py")))
        self.assertFalse(list((ROOT / "configs").glob("smoke_dsh_*.json")))
        self.assertFalse(list((ROOT / "configs").glob("baseline_*.json")))


if __name__ == "__main__":
    unittest.main()
