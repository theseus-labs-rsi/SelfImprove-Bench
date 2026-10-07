import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from farm_planner import generate_task
from llm_client import (
    DSH_INSTALL_HELP,
    LLMClientError,
    build_prompt,
    call_claude,
    call_deepseek_harness,
    call_openai,
    extract_json_object,
    load_rsi_integration,
    request_plan,
)


class LLMClientTests(unittest.TestCase):
    def setUp(self):
        self.task = generate_task(1, seed=3, task_id="llm-test")
        self.plan = {"task_id": "llm-test", "days": []}

    def test_extracts_fenced_or_prefixed_json(self):
        self.assertEqual(extract_json_object("answer:\n```json\n{\"days\": []}\n```"), {"days": []})
        with self.assertRaises(LLMClientError):
            extract_json_object("no JSON here")

    def test_prompt_hides_reference_metadata(self):
        task = dict(self.task)
        task.update({
            "reference_daily_profits": {"wheat": 1.0},
            "reference_profit": 123,
            "reference_method": "greedy_daily_profit_lookahead",
        })
        prompt = build_prompt(task)
        self.assertNotIn("reference_daily_profits", prompt)
        self.assertNotIn("reference_profit", prompt)
        self.assertNotIn("greedy_daily_profit_lookahead", prompt)
        self.assertEqual(task["reference_profit"], 123)

    def test_openai_chat_adapter_and_dispatch(self):
        metrics = {}
        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.plan)))],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=30,
                    total_tokens=130,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
                ),
            )
        ))))
        result = request_plan(self.task, provider="openai", client=fake, metrics=metrics)
        self.assertEqual(result, self.plan)
        self.assertEqual(metrics["tokens"]["prompt_tokens"], 100)
        self.assertEqual(metrics["tokens"]["reasoning_tokens"], 10)
        self.assertEqual(metrics["tokens"]["answer_tokens"], 20)
        self.assertEqual(metrics["tokens"]["total_tokens"], 130)
        fake.chat.completions.create.assert_called_once()

    def test_ace_dispatch_extracts_answer_and_records_backend_usage(self):
        bridge = load_rsi_integration()
        response = bridge.BackendResponse(
            text=json.dumps(
                {
                    "reasoning": "use the current playbook",
                    "bullet_ids": [],
                    "final_answer": self.plan,
                }
            ),
            usage=bridge.TokenUsage(
                input_tokens=40,
                output_tokens=15,
                total_tokens=55,
                reasoning_tokens=5,
            ),
            provider="openai",
            model="test-model",
        )
        session = SimpleNamespace(act=Mock(return_value=response))
        metrics = {}
        result = request_plan(
            self.task,
            provider="ace",
            rsi_session=session,
            metrics=metrics,
        )
        self.assertEqual(result, self.plan)
        self.assertEqual(metrics["tokens"]["total_tokens"], 55)
        self.assertEqual(metrics["tokens"]["reasoning_tokens"], 5)
        self.assertEqual(metrics["tokens"]["answer_tokens"], 10)
        self.assertEqual(
            metrics["attempts"][0]["rsi_agent_call"]["reasoning"],
            "use the current playbook",
        )

    def test_rsi_provider_requires_batch_session(self):
        with self.assertRaisesRegex(LLMClientError, "active RSI batch session"):
            request_plan(self.task, provider="hyperagents")

    def test_responses_adapter(self):
        fake = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=SimpleNamespace(output_text=json.dumps(self.plan)))))
        self.assertEqual(call_openai("prompt", client=fake), json.dumps(self.plan))
        fake.responses.create.assert_called_once()

    def test_codex_runner_and_prompt_contains_task(self):
        runner = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(self.plan), stderr=""))
        result = request_plan(self.task, provider="codex", command="codex", runner=runner)
        self.assertEqual(result, self.plan)
        self.assertEqual(runner.call_args.args[0][-1], "-")
        self.assertIn(self.task["task_id"], runner.call_args.kwargs["input"])
        child_env = runner.call_args.kwargs["env"]
        self.assertNotIn("CODEX_SANDBOX_NETWORK_DISABLED", child_env)
        self.assertNotIn("CODEX_SESSION_ID", child_env)
        self.assertIn("non-interactive API call", runner.call_args.kwargs["input"])

    def test_show_agent_output_configures_streaming_capture(self):
        def fake_runner(args, **kwargs):
            self.assertEqual(kwargs["stdout"], subprocess.PIPE)
            self.assertIsNone(kwargs["stderr"])
            self.assertNotIn("capture_output", kwargs)
            return SimpleNamespace(
                returncode=0, stdout=json.dumps(self.plan), stderr=None
            )

        with patch("sys.stderr") as terminal:
            result = request_plan(
                self.task,
                provider="codex",
                command="codex",
                runner=fake_runner,
                show_agent_output=True,
            )
        self.assertEqual(result, self.plan)
        printed = " ".join(str(call) for call in terminal.write.call_args_list)
        self.assertIn("starting", printed)
        self.assertIn("completed", printed)

    def test_claude_runner_and_dispatch(self):
        envelope = {
            "type": "result",
            "result": json.dumps(self.plan),
            "duration_ms": 1234,
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 300,
                "output_tokens": 30,
            },
        }
        runner = Mock(return_value=SimpleNamespace(
            returncode=0, stdout=json.dumps(envelope), stderr=""
        ))
        metrics = {}
        result = request_plan(
            self.task, provider="claude", command="claude", runner=runner,
            metrics=metrics,
        )
        self.assertEqual(result, self.plan)
        args = runner.call_args.args[0]
        self.assertEqual(args[0], "claude")
        self.assertIn("-p", args)
        self.assertIn("--no-session-persistence", args)
        self.assertIn("--output-format", args)
        self.assertEqual(args[args.index("--output-format") + 1], "json")
        self.assertEqual(args[-1], "--tools=")
        self.assertIn(self.task["task_id"], runner.call_args.kwargs["input"])
        self.assertEqual(metrics["tokens"]["prompt_tokens"], 420)
        self.assertIsNone(metrics["tokens"]["reasoning_tokens"])
        self.assertEqual(metrics["tokens"]["answer_tokens"], 30)
        self.assertEqual(metrics["tokens"]["total_tokens"], 450)
        self.assertEqual(
            metrics["attempts"][0]["tokens"]["input_token_details"]["cache_read_input_tokens"],
            300,
        )

    def test_claude_show_agent_output_uses_streaming_json(self):
        events = [
            {"type": "system", "subtype": "init"},
            {"type": "stream_event", "event": {"type": "content_block_delta"}},
            {
                "type": "result",
                "result": json.dumps(self.plan),
                "usage": {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 20,
                },
            },
        ]

        def fake_runner(args, **kwargs):
            self.assertEqual(args[args.index("--output-format") + 1], "stream-json")
            self.assertIn("--include-partial-messages", args)
            self.assertIn("--verbose", args)
            return SimpleNamespace(
                returncode=0,
                stdout="\n".join(json.dumps(event) for event in events) + "\n",
                stderr=None,
            )

        metrics = {}
        result = request_plan(
            self.task,
            provider="claude",
            command="claude",
            runner=fake_runner,
            show_agent_output=True,
            metrics=metrics,
        )
        self.assertEqual(result, self.plan)
        self.assertEqual(metrics["tokens"]["total_tokens"], 120)

    def test_claude_timeout_has_actionable_message(self):
        import subprocess
        runner = Mock(side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=1))
        with self.assertRaisesRegex(LLMClientError, "authenticated"):
            call_claude("prompt", command="claude", timeout=1, runner=runner)

    def test_claude_uses_all_model_usage_when_top_level_usage_is_zero(self):
        envelope = {
            "type": "result",
            "result": json.dumps(self.plan),
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
            "modelUsage": {
                "helper-model": {
                    "inputTokens": 300,
                    "cacheCreationInputTokens": 20,
                    "cacheReadInputTokens": 100,
                    "outputTokens": 15,
                },
                "planner-model": {
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
            self.task, provider="claude", command="claude", runner=runner,
            metrics=metrics,
        )
        self.assertEqual(result, self.plan)
        self.assertEqual(metrics["tokens"]["prompt_tokens"], 1920)
        self.assertEqual(metrics["tokens"]["answer_tokens"], 115)
        self.assertEqual(metrics["tokens"]["total_tokens"], 2035)
        self.assertEqual(
            metrics["tokens"]["method"], "claude_cli_model_usage_all_models"
        )

    def test_deepseek_harness_runner_and_dispatch(self):
        observed = {}

        def fake_runner(args, **kwargs):
            patch_path = Path(args[args.index("--patch") + 1])
            observed["patch_path"] = patch_path
            observed["patch"] = patch_path.read_text(encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.plan), stderr="")

        runner = Mock(side_effect=fake_runner)
        result = request_plan(
            self.task,
            provider="deepseek",
            model="deepseek-v4-flash",
            command="dsh",
            runner=runner,
        )
        self.assertEqual(result, self.plan)
        args = runner.call_args.args[0]
        self.assertEqual(args[:3], ["dsh", "--profile", "headless"])
        self.assertIn("--patch", args)
        self.assertEqual(args[-1], "farm-planning-task")
        self.assertNotIn(self.task["task_id"], " ".join(args))
        self.assertIn('model: "deepseek-v4-flash"', observed["patch"])
        self.assertIn("- id: headless-runner", observed["patch"])
        self.assertIn(self.task["task_id"], observed["patch"])
        self.assertFalse(observed["patch_path"].exists())

    def test_deepseek_harness_uses_patch_without_model_override(self):
        observed = {}

        def fake_runner(args, **kwargs):
            patch_path = Path(args[args.index("--patch") + 1])
            observed["patch"] = patch_path.read_text(encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        response = call_deepseek_harness(
            "first line\nsecond line", command="dsh", runner=fake_runner
        )
        self.assertEqual(response, "ok")
        self.assertIn('task: "first line\\nsecond line"', observed["patch"])
        self.assertNotIn("agent-default-model", observed["patch"])

    def test_deepseek_uses_complete_session_projection_usage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            projection_dir = Path(temp_dir)

            def fake_runner(args, **kwargs):
                patch_path = Path(args[args.index("--patch") + 1])
                patch_text = patch_path.read_text(encoding="utf-8")
                self.assertIn(self.task["task_id"], patch_text)
                prompt = kwargs.get("input")
                self.assertIsNone(prompt)
                built_prompt = build_prompt(self.task)
                projection = {
                    "record": {
                        "identity": {"cwd": str(Path.cwd())},
                        "rows": {
                            "titleInput": {"val": {"first": {"text": built_prompt}}},
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
                    returncode=0, stdout=json.dumps(self.plan), stderr="visible reasoning"
                )

            metrics = {}
            with patch(
                "llm_client._dsh_projection_directory", return_value=projection_dir
            ):
                result = request_plan(
                    self.task,
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

    def test_deepseek_harness_missing_command_shows_install_help(self):
        with patch("llm_client.shutil.which", return_value=None):
            with self.assertRaisesRegex(LLMClientError, "npm.cmd install"):
                call_deepseek_harness("prompt")
        self.assertIn("DEEPSEEK_API_KEY", DSH_INSTALL_HELP)

    def test_deepseek_harness_timeout_has_actionable_message(self):
        import subprocess
        runner = Mock(side_effect=subprocess.TimeoutExpired(cmd="dsh", timeout=1))
        with self.assertRaisesRegex(LLMClientError, "DEEPSEEK_API_KEY"):
            call_deepseek_harness("prompt", command="dsh", timeout=1, runner=runner)

    def test_codex_timeout_has_actionable_message(self):
        import subprocess
        runner = Mock(side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=1))
        with self.assertRaisesRegex(LLMClientError, "logged in"):
            from llm_client import call_codex
            call_codex("prompt", command="codex", timeout=1, runner=runner)

    def test_codex_timeout_keeps_complete_json_response(self):
        import subprocess
        timeout = subprocess.TimeoutExpired(cmd="codex", timeout=1)
        timeout.stdout = 'thinking\n{"task_id":"llm-test","days":[]}\ntokens used\n'
        runner = Mock(side_effect=timeout)
        result = request_plan(self.task, provider="codex", command="codex", timeout=1, runner=runner)
        self.assertEqual(result, self.plan)

    def test_unknown_provider(self):
        with self.assertRaises(LLMClientError):
            request_plan(self.task, provider="unknown")

    def test_retries_invalid_json_then_returns_plan(self):
        with patch("llm_client.call_codex", side_effect=["not json", json.dumps(self.plan)]) as call:
            result = request_plan(self.task, provider="codex", max_attempts=3)
        self.assertEqual(result, self.plan)
        self.assertEqual(call.call_count, 2)
        self.assertIn("previous response failed validation", call.call_args_list[1].args[0])

    def test_retries_at_most_three_times(self):
        metrics = {}
        with patch("llm_client.call_codex", return_value="not json") as call:
            with self.assertRaises(LLMClientError) as context:
                request_plan(self.task, provider="codex", max_attempts=3, metrics=metrics)
        self.assertEqual(call.call_count, 3)
        self.assertEqual(metrics["attempt_count"], 3)
        self.assertEqual(len(metrics["attempts"]), 3)
        self.assertIn("after 3 attempts", str(context.exception))
        self.assertIn("Last response:\nnot json", str(context.exception))

    def test_final_invalid_response_preview_is_bounded(self):
        response = "start-marker" + ("x" * 9000) + "end-marker"
        with patch("llm_client.call_codex", return_value=response):
            with self.assertRaises(LLMClientError) as context:
                request_plan(self.task, provider="codex", max_attempts=1)
        message = str(context.exception)
        self.assertIn("<truncated; showing last 8000 characters>", message)
        self.assertIn("end-marker", message)
        self.assertNotIn("start-marker", message)

    def test_rejects_invalid_attempt_count(self):
        with self.assertRaises(LLMClientError):
            request_plan(self.task, provider="codex", max_attempts=4)


if __name__ == "__main__":
    unittest.main()
