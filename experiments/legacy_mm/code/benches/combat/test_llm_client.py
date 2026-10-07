import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from combat_planner import CombatSession, generate_task, solve_task
from llm_client import (
    LLMClientError,
    build_prompt,
    call_deepseek_harness,
    load_rsi_integration,
    request_plan,
)


class LLMClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = generate_task(1, seed=3, task_id="llm-test")
        cls.stage_input = CombatSession(cls.task).current_request()
        cls.plan = {
            "task_id": "llm-test",
            "stage_id": cls.stage_input["stage_id"],
            "turns": [],
        }

    def test_prompt_hides_reference_and_verification_metadata(self):
        stage_input = dict(self.stage_input)
        stage_input.update(
            {"reference_score": 999, "reference_method": "hidden", "solvability": {}}
        )
        prompt = build_prompt(stage_input)
        self.assertNotIn("reference_score", prompt)
        self.assertNotIn("reference_method", prompt)
        self.assertNotIn("solvability", prompt)
        self.assertNotIn('"stages"', prompt)
        self.assertIn(self.task["task_id"], prompt)

    def test_openai_dispatch_accepts_stages_plan(self):
        metrics = {}
        fake = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        return_value=SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content=json.dumps(self.plan))
                                )
                            ]
                        )
                    )
                )
            )
        )
        self.assertEqual(
            request_plan(self.stage_input, provider="openai", client=fake, metrics=metrics), self.plan
        )
        self.assertEqual(metrics["attempt_count"], 1)
        self.assertGreater(metrics["duration_seconds"], 0)
        self.assertGreater(metrics["tokens"]["prompt_tokens"], 0)
        self.assertGreater(metrics["tokens"]["answer_tokens"], 0)

    def test_hyperagents_dispatch_uses_active_batch_agent(self):
        bridge = load_rsi_integration()
        response = bridge.BackendResponse(
            text=json.dumps(self.plan),
            usage=bridge.TokenUsage(
                input_tokens=30,
                output_tokens=8,
                total_tokens=38,
            ),
            provider="openai",
            model="test-model",
        )
        session = SimpleNamespace(act=Mock(return_value=response))
        metrics = {}
        result = request_plan(
            self.stage_input,
            provider="hyperagents",
            rsi_session=session,
            metrics=metrics,
        )
        self.assertEqual(result, self.plan)
        self.assertEqual(session.act.call_count, 1)
        self.assertEqual(metrics["tokens"]["total_tokens"], 38)
        self.assertIn("rsi_agent_call", metrics["attempts"][0])

    def test_deepseek_multiline_prompt_uses_configuration_patch(self):
        observed = {}

        def fake_runner(args, **kwargs):
            patch_path = Path(args[args.index("--patch") + 1])
            observed["path"] = patch_path
            observed["text"] = patch_path.read_text(encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="answer", stderr="")

        response = call_deepseek_harness(
            "line one\nline two", command="dsh", runner=fake_runner
        )
        self.assertEqual(response, "answer")
        self.assertIn('task: "line one\\nline two"', observed["text"])
        self.assertFalse(observed["path"].exists())

    def test_claude_uses_json_usage_including_cached_input(self):
        envelope = {
            "type": "result",
            "result": json.dumps(self.plan),
            "usage": {
                "input_tokens": 90,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 200,
                "output_tokens": 25,
            },
        }
        runner = Mock(return_value=SimpleNamespace(
            returncode=0, stdout=json.dumps(envelope), stderr=""
        ))
        metrics = {}
        result = request_plan(
            self.stage_input,
            provider="claude",
            command="claude",
            runner=runner,
            metrics=metrics,
        )
        self.assertEqual(result, self.plan)
        args = runner.call_args.args[0]
        self.assertEqual(args[args.index("--output-format") + 1], "json")
        self.assertEqual(metrics["tokens"]["prompt_tokens"], 300)
        self.assertIsNone(metrics["tokens"]["reasoning_tokens"])
        self.assertEqual(metrics["tokens"]["answer_tokens"], 25)
        self.assertEqual(metrics["tokens"]["total_tokens"], 325)

    def test_claude_complete_total_includes_all_internal_models(self):
        envelope = {
            "type": "result",
            "result": json.dumps(self.plan),
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "modelUsage": {
                "helper": {
                    "inputTokens": 300,
                    "cacheCreationInputTokens": 20,
                    "cacheReadInputTokens": 100,
                    "outputTokens": 15,
                },
                "planner": {
                    "inputTokens": 1000,
                    "cacheCreationInputTokens": 0,
                    "cacheReadInputTokens": 500,
                    "outputTokens": 100,
                },
            },
        }
        runner = Mock(return_value=SimpleNamespace(
            returncode=0, stdout=json.dumps(envelope), stderr=""
        ))
        metrics = {}
        result = request_plan(
            self.stage_input,
            provider="claude",
            command="claude",
            runner=runner,
            metrics=metrics,
        )
        self.assertEqual(result, self.plan)
        self.assertEqual(metrics["tokens"]["prompt_tokens"], 1920)
        self.assertEqual(metrics["tokens"]["answer_tokens"], 115)
        self.assertEqual(metrics["tokens"]["total_tokens"], 2035)
        self.assertEqual(
            metrics["tokens"]["method"], "claude_cli_model_usage_all_models"
        )

    def test_deepseek_complete_total_uses_session_projection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            projection_dir = Path(temp_dir)

            def fake_runner(args, **kwargs):
                prompt = build_prompt(self.stage_input)
                projection = {
                    "record": {
                        "identity": {"cwd": str(Path.cwd())},
                        "rows": {
                            "titleInput": {"val": {"first": {"text": prompt}}},
                            "tokenUsage": {
                                "val": {
                                    "totals": {
                                        "uncachedInputTokens": 1000,
                                        "cacheReadTokens": 300,
                                        "cacheWriteTokens": 200,
                                        "outputTokens": 700,
                                    }
                                }
                            },
                        },
                    }
                }
                (projection_dir / "session-test.json").write_text(
                    json.dumps(projection), encoding="utf-8"
                )
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps(self.plan), stderr="reasoning"
                )

            metrics = {}
            with patch(
                "llm_client._dsh_projection_directory", return_value=projection_dir
            ):
                result = request_plan(
                    self.stage_input,
                    provider="deepseek",
                    command="dsh",
                    runner=fake_runner,
                    metrics=metrics,
                )
        self.assertEqual(result, self.plan)
        self.assertEqual(metrics["tokens"]["prompt_tokens"], 1500)
        self.assertIsNone(metrics["tokens"]["reasoning_tokens"])
        self.assertEqual(metrics["tokens"]["answer_tokens"], 700)
        self.assertEqual(metrics["tokens"]["total_tokens"], 2200)
        self.assertEqual(
            metrics["tokens"]["method"], "dsh_session_projection_provider_reported"
        )

    def test_show_agent_output_streams_stderr_but_captures_stdout(self):
        def fake_runner(args, **kwargs):
            self.assertEqual(kwargs["stdout"], subprocess.PIPE)
            self.assertIsNone(kwargs["stderr"])
            self.assertNotIn("capture_output", kwargs)
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.plan), stderr=None)

        with patch("sys.stderr") as terminal:
            result = request_plan(
                self.stage_input,
                provider="codex",
                command="codex",
                runner=fake_runner,
                show_agent_output=True,
            )
        self.assertEqual(result, self.plan)
        printed = " ".join(str(call) for call in terminal.write.call_args_list)
        self.assertIn("starting", printed)
        self.assertIn("completed", printed)

    def test_three_invalid_attempts_include_last_response(self):
        metrics = {}
        with patch("llm_client.call_codex", return_value="invalid final response") as call:
            with self.assertRaises(LLMClientError) as context:
                request_plan(self.stage_input, provider="codex", metrics=metrics)
        self.assertEqual(call.call_count, 3)
        self.assertEqual(metrics["attempt_count"], 3)
        self.assertEqual(len(metrics["attempts"]), 3)
        self.assertEqual(metrics["tokens"]["prompt_tokens"], sum(
            attempt["tokens"]["prompt_tokens"] for attempt in metrics["attempts"]
        ))
        self.assertIn("Last response:\ninvalid final response", str(context.exception))

    def test_shop_request_accepts_only_a_local_shop_decision(self):
        session = CombatSession(self.task)
        reference = solve_task(self.task)
        session.apply(reference["stages"][0])
        shop_input = session.current_request()
        shop_plan = {
            "task_id": self.task["task_id"],
            "stage_id": shop_input["stage_id"],
            "buy_cards": [],
            "heal": False,
        }
        with patch("llm_client.call_codex", return_value=json.dumps(shop_plan)):
            self.assertEqual(request_plan(shop_input, provider="codex"), shop_plan)
        invalid = {"task_id": self.task["task_id"], "stages": []}
        with patch("llm_client.call_codex", return_value=json.dumps(invalid)):
            with self.assertRaises(LLMClientError):
                request_plan(shop_input, provider="codex", max_attempts=1)


if __name__ == "__main__":
    unittest.main()
