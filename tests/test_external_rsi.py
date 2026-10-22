"""Offline contract and integration regressions for third-party RSI methods."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmark import resume_state
from benchmark.client import request_json
from benchmark.config import AgentConfig, ReplayConfig, config_from_mapping, load_config
from benchmark.contracts import BackendResponse, EvaluationOutcome, SelectionEvaluation, TokenUsage
from benchmark.hci import calculate_hci
from benchmark.interfaces import Candidate, CandidateBatch, MethodContext, SelectionResult
from benchmark.methods import create_adaptor, MethodRuntimePolicy, runtime_policy
from benchmark.rsi_session import RSIBatchSession, make_feedback_result, ExecutionStatus
from benchmark.runner import run_benchmark
from benchmark.tasks import GAMES, build_manifest
from examples.minimal_rsi import FixedAgent, MinimalAdapter


ROOT = Path(__file__).resolve().parents[1]


def agent_config(spec="examples.minimal_rsi:MinimalAdapter"):
    return AgentConfig(provider="rsi", adapter=spec)


class SelectingAdapter(MinimalAdapter):
    def refine(self, feedback, *, output_dir):
        super().refine(feedback, output_dir=output_dir)
        return CandidateBatch([Candidate("proposal", self.load(), output_dir / "proposal")], output_dir)

    def select(self, *, batch, parent_evaluation, candidate_evaluations, output_dir, selection_mode):
        return SelectionResult("committed", "proposal")


class ExternalMethodTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_external_runtime_policy_is_optional_and_method_owned(self):
        default = runtime_policy(agent_config())
        self.assertIsInstance(default, MethodRuntimePolicy)
        self.assertFalse(default.is_fatal_error("AHE selection failed"))
        self.assertFalse(default.reload_holdout_agent)
        custom = MethodRuntimePolicy()
        with patch.object(MinimalAdapter, "runtime_policy", custom, create=True):
            self.assertIs(runtime_policy(agent_config()), custom)

    def test_farm_combat_success_requires_reference_score(self):
        from benchmark.game_adapters import evaluate_task, GAME_MODULES
        for game in ("farm", "combat"):
            for score, status, expected in ((0.4, "ok", False), (1.0, "ok", True),
                                            (1.2, "ok", True), (1.2, "error", False)):
                with self.subTest(game=game, score=score, status=status):
                    record = {"status": status, "raw_score": score, "reference_score": 1.0,
                              "relative_score": score, "success": True}
                    with patch.object(GAME_MODULES[game], "evaluate", return_value=record):
                        result = evaluate_task(game, {}, request=Mock())
                    self.assertEqual(result.record["success"], expected)
                    self.assertEqual(result.record["relative_score"], score)

    def test_runner_keeps_task_success_distinct_from_execution_status(self):
        from benchmark import rsi_session
        from benchmark.runner import AgentController
        from benchmark.tasks import PlannedTask
        controller = AgentController.__new__(AgentController)
        controller.session = object()
        controller.bridge = rsi_session
        task = PlannedTask("test", "train", "farm", 3, {"task_id": "test"}, 1, 0)
        result = controller.make_feedback(task, {
            "status": "ok", "success": False, "relative_score": 0.4,
        })
        self.assertEqual(result.evaluation.execution_status, ExecutionStatus.COMPLETED)
        self.assertFalse(result.evaluation.success)
        self.assertEqual(result.evaluation.score, 0.4)

    def test_config_requires_explicit_valid_adapter_without_importing_it(self):
        mapping = {"schedule": [], "holdout": [{"game": "farm", "difficulty": 1}],
                   "agent": {"provider": "rsi", "adapter": "not_installed.module:Adapter"}}
        with patch("benchmark.methods.importlib.import_module", side_effect=AssertionError("No import")):
            self.assertEqual(config_from_mapping(mapping).agent.adapter, "not_installed.module:Adapter")
        for spec in (None, "", "module", ":Adapter", "module:", "module:Adapter()", "module:Class:other"):
            mapping["agent"]["adapter"] = spec
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, "agent.adapter"):
                config_from_mapping(mapping)
        mapping["agent"] = {"provider": "ace", "adapter": "module:Adapter"}
        with self.assertRaisesRegex(ValueError, "require"):
            config_from_mapping(mapping)
        mapping["agent"] = {"provider": "rsi", "adapter": "module:Adapter", "adapter_options": []}
        with self.assertRaisesRegex(ValueError, "adapter_options"):
            config_from_mapping(mapping)

    def test_builtin_serialization_keeps_existing_identity(self):
        for path in (ROOT / "configs/main").glob("*.json"):
            config = load_config(path)
            self.assertEqual(config.as_dict()["agent"]["adapter"], config.agent.adapter)
            self.assertNotIn("adapter_options", config.as_dict()["agent"])
        config = replace(config, agent=AgentConfig(provider="ace"))
        self.assertNotIn("adapter", config.as_dict()["agent"])

    def test_factory_context_is_isolated_and_missing_hooks_fail_early(self):
        settings = replace(agent_config(), adapter_options={"action": {"days": []}}, model="test")
        adapter = create_adaptor("rsi", artifact_dir=self.root, task_count=5, agent=settings,
                                 resuming=True, seed=73)
        self.assertTrue(adapter.context.resuming)
        self.assertEqual(adapter.context.seed, 73)
        self.assertEqual(adapter.context.model, "test")
        self.assertEqual(adapter.context.task_count, 5)
        adapter.context.options["action"]["days"].append("changed")
        self.assertEqual(settings.adapter_options["action"]["days"], [])
        with patch("benchmark.methods.importlib.import_module", return_value=SimpleNamespace(Adapter=lambda c: object())):
            with self.assertRaisesRegex(TypeError, "missing hooks"):
                create_adaptor("rsi", artifact_dir=self.root, task_count=1,
                               agent=agent_config("external:Adapter"))
        with self.assertRaisesRegex(RuntimeError, "Cannot load RSI adapter"):
            create_adaptor("rsi", artifact_dir=self.root, task_count=1,
                           agent=agent_config("not_installed.module:Adapter"))

    def test_session_passes_seed_to_method_context(self):
        session = RSIBatchSession("rsi", self.root, task_count=5,
                                  agent=agent_config(), seed=73)
        self.addCleanup(session.close)
        self.assertEqual(session.adaptor.context.seed, 73)

    def test_custom_agent_uses_json_validation_and_telemetry(self):
        metrics = {}
        action = request_json({}, prompt="game observation", validator=lambda action, task: action,
                              provider="rsi", rsi_agent=FixedAgent({"days": []}), metrics=metrics)
        self.assertEqual(action, {"days": []})
        self.assertEqual(metrics["tokens"]["total_tokens"], 0)
        self.assertEqual(metrics["attempts"][0]["rsi_agent_call"]["response_text"], '{"days": []}')

    def test_state_roundtrip_and_adapter_identity(self):
        session = RSIBatchSession("rsi", self.root, task_count=5, agent=agent_config())
        self.addCleanup(session.close)
        session.adaptor.rounds = 3
        state = resume_state.export(session)
        session.adaptor.rounds = 99
        resume_state.restore(session, state)
        self.assertEqual(session.adaptor.rounds, 3)
        state["adapter"] = "other:Adapter"
        with self.assertRaisesRegex(RuntimeError, "adapter identity"):
            resume_state.restore(session, state)
        session.adaptor.export_state = lambda: {"score": float("nan")}
        with self.assertRaises(ValueError):
            resume_state.export(session)

    def test_optional_candidates_use_public_contracts(self):
        adapter = SelectingAdapter(MethodContext(self.root, 1, {}))
        session = RSIBatchSession("rsi", self.root, task_count=1, agent=agent_config(), adaptor=adapter)
        self.addCleanup(session.close)
        session.begin_task("training")
        feedback = make_feedback_result(task_id="training", task={}, trajectory=[],
                                       execution_status=ExecutionStatus.COMPLETED,
                                       success=True, score=0.4, feedback="done", metrics={})
        batch = session.refine(feedback, 1)
        parent = SelectionEvaluation("ok", 0.4, True, "parent.json")
        candidate = SelectionEvaluation("ok", 0.7, True, "candidate.json")
        result = session.select(parent, {"proposal": candidate})
        self.assertEqual(result.committed_candidate_id, "proposal")
        self.assertIsNone(session.pending_batch)

    def test_runner_validates_external_candidates_only_on_training_task(self):
        config = config_from_mapping({
            "schedule": [{"game": "farm", "difficulty": 1, "count": 1}],
            "holdout": [{"game": "farm", "difficulty": 1}],
            "agent": {"provider": "rsi", "adapter": f"{__name__}:SelectingAdapter"},
            "output_dir": str(self.root / "candidates"),
        })
        calls = []

        def evaluate(task, **kwargs):
            calls.append((task.phase, task.run_id, task.task))
            return EvaluationOutcome({"status": "ok", "game": task.game,
                                      "difficulty": task.difficulty,
                                      "relative_score": 0.7 if len(calls) == 2 else 0.4,
                                      "success": True, "interactions": [], "llm_metrics": {}})

        def generate(difficulty, *, seed, task_id):
            return {"difficulty": difficulty, "seed": seed, "task_id": task_id}

        with patch.object(GAMES["farm"], "generate", side_effect=generate), \
                patch("benchmark.runner._evaluate", side_effect=evaluate), \
                redirect_stdout(io.StringIO()):
            result = run_benchmark(config)
        self.assertEqual([call[0] for call in calls], ["train", "train", "holdout"])
        self.assertEqual(calls[0][1:], calls[1][1:])
        training = result["training_runs"][0]
        self.assertEqual(training["relative_score"], 0.4)
        self.assertEqual(training["post_refinement_score"], 0.7)
        self.assertEqual(training["selection"]["committed_candidate_id"], "proposal")
        self.assertEqual(len(result["holdout_runs"]), 1)

    def test_checkpoint_resume_replay_and_holdout_do_not_train(self):
        config = replace(load_config(ROOT / "configs/examples/minimal_rsi.json"),
                         output_dir=str(self.root / "run"))

        def generate(difficulty, *, seed, task_id):
            return {"difficulty": difficulty, "seed": seed, "task_id": task_id}

        def evaluate(task, request=None, agent=None, reload_agent=None, output_dir=None):
            # Keep the runner, feedback, replay, checkpoints and method real.
            self.assertIsInstance(agent, FixedAgent)
            agent.act("observation")
            return EvaluationOutcome({"status": "ok", "game": task.game,
                                      "difficulty": task.difficulty, "relative_score": 0.4,
                                      "success": True, "interactions": [], "llm_metrics": {}})

        with patch.object(GAMES["farm"], "generate", side_effect=generate), \
                patch("benchmark.runner._evaluate", side_effect=evaluate), redirect_stdout(io.StringIO()):
            paused = run_benchmark(config, max_new_train_runs=2)
            self.assertTrue(paused["summary"]["paused_after_checkpoint"])
            self.assertEqual(paused["summary"]["completed_training_runs"], 2)
            self.assertFalse(paused["holdout_runs"])
            # Resuming must restore state, not call init again.
            with patch.object(MinimalAdapter, "init", side_effect=AssertionError("Must not re-init")):
                complete = run_benchmark(config, resume=True)
            self.assertEqual(complete["summary"]["completed_training_runs"], 6)
            self.assertEqual(complete["summary"]["completed_holdout_runs"], 1)
            self.assertEqual(complete["metrics"]["capability_retention"]["replay_pair_count"], 1)
            self.assertEqual(complete["metrics"]["capability_retention"]["pairs"][0]["gap"], 3)
            self.assertIsNone(complete["training_runs"][0]["post_refinement_score"])
            state_path = sorted((self.root / "run/checkpoints").glob("holdout_*.json"))[-1]
            self.assertEqual(json.loads(state_path.read_text())["state"]["adapter_state"]["rounds"], 5)
            again = run_benchmark(config, resume=True)
            self.assertEqual(again["training_runs"], complete["training_runs"])
            self.assertEqual(again["holdout_runs"], complete["holdout_runs"])
            initial_config = replace(config, schedule=(), replay=ReplayConfig(),
                                     output_dir=str(self.root / "initial"))
            initial = run_benchmark(initial_config)
            self.assertFalse(initial["training_runs"])
            initial_state = json.loads((self.root / "initial/checkpoints/initial.json").read_text())
            self.assertEqual(initial_state["state"]["adapter_state"]["rounds"], 0)
        with self.assertRaisesRegex(ValueError, "--resume"):
            run_benchmark(config, target_dir=self.root / "copy")

    def test_hci_accepts_external_method_with_matching_initial(self):
        config = replace(load_config(ROOT / "configs/main/farm_ace.json"), agent=agent_config())
        initial_config = replace(config, schedule=(), replay=ReplayConfig())

        def generate(difficulty, *, seed, task_id):
            return {"difficulty": difficulty, "seed": seed, "task_id": task_id}

        with ExitStack() as stack:
            for name, module in GAMES.items():
                if name != "gameworld":
                    stack.enter_context(patch.object(module, "generate", side_effect=generate))
            manifest = build_manifest(config).as_dict()
            initial_manifest = build_manifest(initial_config).as_dict()

        def result(settings, tasks, baseline=False):
            def record(task):
                return {**{k: v for k, v in task.items() if k not in {"task", "task_fingerprint"}},
                        "status": "ok", "relative_score": 0.2 if baseline else 0.6}
            return {"config": settings.as_dict(),
                    "training_runs": [record(t) for t in tasks["training"]],
                    "holdout_runs": [record(t) for t in tasks["holdout"]]}
        report = calculate_hci(result(config, manifest), result(initial_config, initial_manifest, True),
                               manifest, initial_manifest)
        self.assertEqual(report["status"], "completed")
        for name in ("G0", "G1", "G2", "G3"):
            self.assertIsInstance(report["groups"][name]["hci"], float)

    def test_all_game_configs_can_use_the_same_external_adapter(self):
        from benches.gameworld.frameworks import client_import_spec
        for path in (ROOT / "configs/main").glob("*_ace.json"):
            mapping = json.loads(path.read_text())
            mapping["agent"] = {"provider": "rsi", "adapter": "examples.minimal_rsi:MinimalAdapter"}
            if "gameworld" in mapping.get("game_options", {}):
                mapping["game_options"]["gameworld"].pop("framework")
            with self.subTest(game=path.stem):
                config = config_from_mapping(mapping)
                self.assertEqual(config.agent.provider, "rsi")
                if "gameworld" in config.game_options:
                    self.assertEqual(config.game_options["gameworld"]["framework"], "rsi")
        self.assertEqual(client_import_spec("rsi"), ("benches.gameworld.rsi_game_client", "RSIGameClient"))

    def test_gameworld_step_recovery_is_explicitly_unsupported(self):
        from benches.gameworld.benchmark import GameWorldBenchmark
        with self.assertRaisesRegex(ValueError, "task-level"):
            GameWorldBenchmark().evaluate({"framework": "rsi", "resumable_steps": True}, object())


class VisualClientTest(unittest.TestCase):
    def test_all_methods_use_shared_gameworld_transport(self):
        from benches.gameworld.frameworks import SUPPORTED_FRAMEWORKS, client_import_spec
        for method in SUPPORTED_FRAMEWORKS:
            with self.subTest(method=method):
                self.assertEqual(client_import_spec(method), (
                    "benches.gameworld.rsi_game_client", "RSIGameClient"))

    def test_method_unwraps_ace_answer_before_game_parsing(self):
        from benches.sweepmine.play import extract_action
        from rsi_methods.ACE.ACEAgent import ACEAgent

        action = '{"act":"click","row":0,"col":0}'
        raw = json.dumps({"final_answer": json.loads(action), "reasoning": "test"})
        response = ACEAgent._unwrap_response(BackendResponse(raw))
        self.assertEqual(json.loads(extract_action(response.text)), json.loads(action))
        self.assertEqual(response.metadata["raw_response_text"], raw)
        self.assertEqual(response.metadata["ace"]["raw_response_text"], raw)
        # Games must not interpret a method's private response envelope.
        self.assertEqual(extract_action(raw), raw)

    def test_shared_transport_retains_raw_response_usage_and_legacy_constructors(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        screenshot = Path(directory.name) / "images/frame.png"
        core = str(ROOT / "benches/gameworld/assets/core/code")
        with patch.object(sys, "path", [core, *sys.path]):
            from agents.mm_agents.base.base_client import BaseClientConfig
            from benches.gameworld.rsi_game_client import RSIGameClient
            from benches.gameworld.ace_game_client import ACEGameClient
            from benches.gameworld.ahe_game_client import AHEGameClient
            from benches.gameworld.hyperagents_game_client import HyperAgentsGameClient
        action = '{"tool_name":"move","arguments":{}}'
        actor = Mock()
        actor.act.return_value = BackendResponse(
            action, reasoning="test", usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
            metadata={"raw_response_text": "original model output"})
        config = BaseClientConfig(system_prompt="Game rules")
        for client_type, keyword in (
            (RSIGameClient, "rsi_agent"), (ACEGameClient, "ace_agent"),
            (AHEGameClient, "ahe_agent"), (HyperAgentsGameClient, "runtime_agent"),
        ):
            with self.subTest(client=client_type.__name__):
                client = client_type(config, **{keyword: actor},
                                     semantic_controls_specs=[{"id": "move"}])
                self.assertIs(client._rsi_agent, actor)
                self.assertIs(client_type.get_action, RSIGameClient.get_action)
                client._complete_action = Mock()
                client.get_action(screenshot)
                record = client._complete_action.call_args.kwargs
                self.assertEqual(record["raw_response"], "original model output")
                self.assertEqual(record["usage"]["total_tokens"], 5)
                self.assertEqual(record["reasoning"], "test")
                self.assertEqual(record["tool_call"], json.loads(action))

    def test_ahe_step_recovery_uses_shared_agent_handle(self):
        from benches.gameworld.astray_step_resume import AstrayTransaction
        transaction = AstrayTransaction.__new__(AstrayTransaction)
        transaction.framework = "ahe"
        transaction._pack_worker = Mock(side_effect=lambda value: value)
        transaction._unpack_worker = Mock(side_effect=lambda value: value)
        actor = Mock()
        actor.snapshot_session.return_value = {"history": ["saved message"]}
        client = SimpleNamespace(_rsi_agent=actor, _runtime_advisory=None,
                                 _runtime_reasoning_required=False, memory_store=None)
        with patch("benches.gameworld.astray_step_resume._world.snapshot_memory", return_value=[]), \
                patch("benches.gameworld.astray_step_resume._world.restore_memory"):
            state = transaction._snapshot_client(client, 1)
            transaction._restore_client(client, state)
        self.assertEqual(state["schema"], 1)
        actor.restore_session.assert_called_once_with({"history": ["saved message"]})

    def test_screenshot_and_game_prompt_reach_external_agent(self):
        # Load only the bundled client contract, not the browser or model backends.
        core = str(ROOT / "benches/gameworld/assets/core/code")
        with patch.object(sys, "path", [core, *sys.path]):
            from agents.mm_agents.base.base_client import BaseClientConfig
            from benches.gameworld.rsi_game_client import RSIGameClient
        actor = Mock()
        actor.act.return_value = BackendResponse('{"tool_name":"move","arguments":{}}')
        client = RSIGameClient(BaseClientConfig(system_prompt="Game rules"), actor,
                               semantic_controls_specs=[{"id": "move"}])
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "images/frame.png"
            client._complete_action = Mock(return_value={"tool_name": "move"})
            self.assertEqual(client.get_action(screenshot), {"tool_name": "move"})
            args, kwargs = actor.act.call_args
            self.assertIn("Game rules", args[0])
            self.assertIn("# Current Observation", args[0])
            self.assertEqual(kwargs["image_paths"], [screenshot])
            self.assertIsNone(client._complete_action.call_args.kwargs["error"])
            actor.act.return_value = BackendResponse('{"tool_name":"unregistered"}')
            client.get_action(screenshot)
            self.assertIsNotNone(client._complete_action.call_args.kwargs["error"])


if __name__ == "__main__":
    unittest.main()
