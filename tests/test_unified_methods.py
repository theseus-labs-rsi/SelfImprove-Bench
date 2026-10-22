"""Bundled methods use the same public adapter lifecycle as external methods."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchmark import resume_state
from benchmark.config import AgentConfig, load_config, config_from_mapping
from benchmark.methods import BUILTIN_ADAPTERS, REQUIRED_HOOKS, create_adaptor, method_identity
from benchmark.rsi_session import RSIBatchSession
from benchmark.runner import AgentController, _is_fatal_method_error


ROOT = Path(__file__).resolve().parents[1]


class UnifiedMethodsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def settings(self, method):
        return AgentConfig(provider="rsi", adapter=BUILTIN_ADAPTERS[method], timeout=123)

    def test_all_real_constructors_accept_the_public_context(self):
        for method in BUILTIN_ADAPTERS:
            with self.subTest(method=method):
                explicit = self.settings(method)
                alias = replace(explicit, provider=method, adapter=None)
                first = create_adaptor("rsi", artifact_dir=self.root, task_count=5, agent=explicit)
                second = create_adaptor(method, artifact_dir=self.root, task_count=5, agent=alias)
                self.addCleanup(first.close)
                self.addCleanup(second.close)
                self.assertIs(type(first), type(second))
                self.assertTrue(all(callable(getattr(first, hook)) for hook in REQUIRED_HOOKS))
                self.assertEqual(first._method_context, second._method_context)
                self.assertEqual(method_identity(explicit), method)
                self.assertEqual(method_identity(alias), method)
                if method in ("ahe", "hyperagents"):
                    self.assertEqual(first.run_dir, self.root / "state")
                if method == "ahe":
                    self.assertEqual(first.timeout_seconds, 123)
                    self.assertEqual(first.checkpoint_policy(), second.checkpoint_policy())
                if method == "hyperagents":
                    self.assertEqual(first.max_refinement_rounds, 5)
                with self.assertRaisesRegex(ValueError, "adapter_options"):
                    create_adaptor("rsi", artifact_dir=self.root, task_count=5,
                                   agent=replace(explicit, adapter_options={"unknown": True}))

    def test_real_initial_states_roundtrip_through_public_hooks(self):
        from rsi_methods.ACE.ACEAdaptor import ACEAdaptor
        configure = ACEAdaptor._configure

        def configure_ace(adaptor, config):
            configure(adaptor, config)
            adaptor.workspace_root = self.root / "ace_workspaces"

        with patch("rsi_methods.ACE.ACEAdaptor.OpenAIBackend"), \
                patch.object(ACEAdaptor, "_configure", configure_ace):
            for method in BUILTIN_ADAPTERS:
                with self.subTest(method=method):
                    settings = self.settings(method)
                    original = RSIBatchSession("rsi", self.root / "rsi", task_count=5,
                                               batch_id=method, agent=settings)
                    self.addCleanup(original.close)
                    original._selection_stage = 3
                    state = resume_state.export(original)
                    self.assertEqual(state["adapter"], BUILTIN_ADAPTERS[method])
                    self.assertEqual(state["method"], method)
                    self.assertEqual(original.artifact_dir, self.root / "rsi" / method / method)
                    resumed = RSIBatchSession("rsi", self.root / "rsi", task_count=5,
                                              batch_id=method, agent=settings, resume=True)
                    self.addCleanup(resumed.close)
                    resume_state.restore(resumed, state)
                    self.assertEqual(resume_state.export(resumed), state)
                    self.assertEqual(resumed._selection_stage, 3)
                    # Old alias configurations and pre-interface envelopes still restore.
                    legacy = {key: state[key] for key in ("method", "batch_id", "selection_stage")}
                    legacy.update(state["adapter_state"])
                    resume_state.restore(resumed, legacy)
                    self.assertEqual(resume_state.export(resumed), state)
                    alias = replace(settings, provider=method, adapter=None)
                    old_entry = RSIBatchSession(method, self.root / "rsi", task_count=5,
                                                batch_id=method, agent=alias, resume=True)
                    self.addCleanup(old_entry.close)
                    resume_state.restore(old_entry, state)
                    self.assertEqual(resume_state.export(old_entry), state)
                    if method == "ahe":
                        changed = json.loads(json.dumps(state))
                        changed["adapter_state"]["ahe_policy"]["refinement_every"] += 1
                        with self.assertRaisesRegex(RuntimeError, "policy"):
                            resume_state.restore(resumed, changed)
                    if method == "ace":
                        policy_path = original.adaptor.workspace_dir / "checkpoint_policy.json"
                        self.assertEqual(json.loads(policy_path.read_text()),
                                         state["adapter_state"]["ace_policy"])
                        for field in (None, "evolution_policy", "prompt_digest"):
                            changed = json.loads(json.dumps(state))
                            if field is None:
                                changed["adapter_state"].pop("ace_policy")
                            else:
                                changed["adapter_state"]["ace_policy"][field] = "old"
                            with self.subTest(policy_field=field):
                                with self.assertRaisesRegex(RuntimeError, "ACE checkpoint policy"):
                                    resume_state.restore(resumed, changed)
                        self.assertEqual(resume_state.export(resumed), state)
                        with patch("rsi_methods.ACE.ACEAdaptor.ACE_AGENT_PROMPT", "changed prompt"):
                            with self.assertRaisesRegex(RuntimeError, "ACE checkpoint policy"):
                                resume_state.restore(resumed, state)
                        copied_batch = self.root / "copied_ace_batch"
                        copied_batch.mkdir()
                        (copied_batch / "batch_manifest.json").write_text("{}")
                        for policy in (None, {"evolution_policy": "old"}):
                            if policy is not None:
                                (copied_batch / "checkpoint_policy.json").write_text(json.dumps(policy))
                            with self.assertRaisesRegex(RuntimeError, "ACE resume policy"):
                                resumed.adaptor.resume(
                                    copied_batch, source_batch=original.artifact_dir,
                                    before_task_index=1, target_dir=self.root / "resumed_ace",
                                )
                        self.assertFalse((self.root / "resumed_ace").exists())
                        (copied_batch / "checkpoint_policy.json").write_text(
                            json.dumps(original.adaptor.checkpoint_policy()))
                        (copied_batch / "method_config.toml").write_bytes(
                            original.adaptor._config_snapshot)
                        resumed.adaptor.resume(
                            copied_batch, source_batch=original.artifact_dir,
                            before_task_index=1, target_dir=self.root / "resumed_ace",
                        )
                        self.assertEqual(
                            json.loads((original.artifact_dir / "checkpoint_policy.json").read_text()),
                            original.adaptor.checkpoint_policy(),
                        )
                        self.assertEqual(resumed.adaptor.version, 0)
                    if method == "hyperagents":
                        self.assertEqual(resumed.adaptor.max_refinement_rounds, 5)

    def test_gameworld_transport_and_manifest_options_keep_method_identity(self):
        for method in BUILTIN_ADAPTERS:
            with self.subTest(method=method):
                config = load_config(ROOT / "configs/main" / f"minecraft_{method}.json")
                self.assertEqual(config.game_options["gameworld"]["framework"], method)
                raw = json.loads((ROOT / "configs/main" / f"minecraft_{method}.json").read_text())
                raw["game_options"]["gameworld"].pop("framework")
                self.assertEqual(config_from_mapping(raw).game_options, config.game_options)
                raw["game_options"]["gameworld"]["framework"] = "rsi"
                with self.assertRaisesRegex(ValueError, "must match"):
                    config_from_mapping(raw)
                with patch("benchmark.runner.load_rsi_integration") as integration:
                    controller = AgentController(config, self.root, 5)
                self.assertEqual(controller.provider, method)
                self.assertEqual(integration.return_value.create_rsi_batch.call_args.args[0], "rsi")
                with patch("benchmark.runner.request_json", return_value={}) as request:
                    controller._request_json({}, object(), prompt="observation", validator=None, metrics={})
                self.assertEqual(request.call_args.kwargs["provider"], method)
                self.assertEqual(_is_fatal_method_error(config, {
                    "status": "error", "error": "irrelevant nonfatal error",
                }), False)
                self.assertEqual(_is_fatal_method_error(config, {
                    "status": "error", "error": "AHE candidate generation failed",
                }), method == "ahe")


if __name__ == "__main__":
    unittest.main()
