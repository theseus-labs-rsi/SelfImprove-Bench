"""Offline checks for per-task AHE refinement without changing old lineages."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.config import config_from_mapping
from benchmark.runner import AgentController
from benchmark.rsi_session import RSIBatchSession


class RefinementFrequencyTest(unittest.TestCase):
    def _config(self, frequency=None):
        options = {"framework": "ahe", "train": [{"game": "03_astray", "task": "03_t01"}],
                   "holdout": [{"game": "03_astray", "task": "03_01"}]}
        if frequency is not None:
            options["ahe_refinement_every"] = frequency
        return config_from_mapping({
            "seed": 42,
            "schedule": [{"game": "gameworld", "difficulty": 1, "count": 1}],
            "holdout": [{"game": "gameworld", "difficulty": 1, "count": 1}],
            "holdout_tasks_per_difficulty": 1,
            "agent": {"provider": "ahe", "reasoning_effort": "low"},
            "game_options": {"gameworld": options},
        })

    def test_explicit_frequency_passed_to_session(self):
        with TemporaryDirectory() as tmp:
            with patch("benchmark.runner.load_rsi_integration") as integration:
                AgentController(self._config(1), Path(tmp), 20)
                arguments = integration.return_value.create_rsi_batch.call_args.kwargs
                self.assertEqual(arguments["agent"].ahe_refinement_every, 1)
                self.assertEqual(arguments["agent"].reasoning_effort, "low")

    def test_legacy_lineage_preserves_frequency_five(self):
        with TemporaryDirectory() as tmp:
            with patch("benchmark.runner.load_rsi_integration") as integration:
                AgentController(self._config(), Path(tmp), 20)
                arguments = integration.return_value.create_rsi_batch.call_args.kwargs
                self.assertEqual(arguments["agent"].ahe_refinement_every, 5)

    def test_frequency_must_be_positive_integer(self):
        with TemporaryDirectory() as tmp:
            for invalid in (0, -1, True, "1"):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "ahe_refinement_every"):
                    AgentController(self._config(invalid), Path(tmp), 20)

    def test_refinement_every_task_is_persisted_in_checkpoint_policy(self):
        from rsi_methods.AHE.AHEAdaptor import AHEAdaptor
        with TemporaryDirectory() as tmp:
            adaptor = AHEAdaptor(run_dir=Path(tmp) / "ahe", refinement_every=1)
            self.assertEqual(adaptor.checkpoint_policy()["refinement_every"], 1)

    def test_session_passes_frequency_to_adaptor(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "rsi"
            with patch("rsi_methods.AHE.AHEAdaptor.AHEAdaptor") as adaptor:
                adaptor.return_value.config_path = None
                adaptor.return_value.model = None
                adaptor.return_value.reasoning_effort = "low"
                adaptor.return_value.run_dir = str(root)
                adaptor.return_value.reuse_existing = None
                adaptor.return_value.max_refinement_rounds = None
                RSIBatchSession("ahe", root, task_count=20,
                                agent=SimpleNamespace(ahe_refinement_every=1))
                self.assertEqual(adaptor.call_args.kwargs["refinement_every"], 1)


if __name__ == "__main__":
    unittest.main()
