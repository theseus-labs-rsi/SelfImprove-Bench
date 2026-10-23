"""Offline regressions for direct visual evidence and restored seed prompts."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsi_methods.HyperAgents.lifecycle import _stage_feedback_screenshots
from rsi_methods.HyperAgents.visual_evidence import collect_game_screenshots


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hyperagents"))


class ScreenshotEvidenceTests(unittest.TestCase):
    def test_fresh_candidate_keeps_restored_prompt_and_direct_hook(self):
        from rsi_methods.HyperAgents.archive_utils import setup_initial_candidate
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {
                "GIT_AUTHOR_NAME": "Offline Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "Offline Test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
                "RSI_MAX_REFINEMENT_IMAGES": "8",
            }):
                root, commit = setup_initial_candidate(directory)
            root = Path(root)
            self.assertTrue(commit)
            source = (root / "meta_agent.py").read_text()
            self.assertIn('instruction = f"Modify any part of the codebase at `{repo_path}`."', source)
            self.assertNotIn("Recursively inspect", source)
            self.assertIn("image_paths=image_paths", source)
            self.assertTrue((root / "visual_evidence.py").is_file())
            runtime = (root / "agent/llm_withtools.py").read_text()
            self.assertIn("ONLY USE ONE TOOL PER RESPONSE", runtime)
            self.assertNotIn("Do not claim that tool access is unavailable", runtime)

    def make_evidence(self, root, count=1):
        steps = []
        for index in range(count):
            name = f"artifacts/step_{index}.png"
            image = root / name
            image.parent.mkdir(exist_ok=True)
            image.write_bytes(b"\x89PNG\r\n\x1a\nsource")
            steps.append({"step_index": index, "image_records": [{"path": name}]})
        result = {"cases": [{"case_id": "case-a", "feedback": {"trajectory": steps}}]}
        (root / "result.json").write_text(json.dumps(result))
        return result

    def test_pinned_source_receives_direct_hook_once(self):
        from rsi_methods.HyperAgents.archive_utils import _install_visual_evidence_hook
        with tempfile.TemporaryDirectory() as directory:
            meta = Path(directory) / "meta_agent.py"
            meta.write_text(
                "class MetaAgent:\n"
                "    def forward(self, eval_path):\n"
                '        instruction = "improve"\n'
                "        new_msg_history = chat_with_agent(\n"
                "            instruction,\n"
                "        )\n"
                "        return new_msg_history\n"
            )
            _install_visual_evidence_hook(directory)
            first = meta.read_text()
            self.assertIn("collect_game_screenshots(eval_path)", first)
            self.assertIn("image_paths=image_paths", first)
            self.assertNotIn("inspect_game_screenshots", first)
            compile(first, str(meta), "exec")
            _install_visual_evidence_hook(directory)
            self.assertEqual(meta.read_text(), first)

    def test_unexpected_meta_agent_fails(self):
        from rsi_methods.HyperAgents.archive_utils import _install_visual_evidence_hook
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "meta_agent.py").write_text("class MetaAgent:\n    pass\n")
            with self.assertRaisesRegex(RuntimeError, "Cannot install visual evidence"):
                _install_visual_evidence_hook(directory)

    def test_feedback_images_are_bundled_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.png"
            source.write_bytes(b"\x89PNG\r\n\x1a\nsource")
            original = {"cases": [{"feedback": {"trajectory": [
                {"step_index": i, "image_records": [{"path": str(source)}]}
                for i in (1, 2)
            ]}}]}
            staged = copy.deepcopy(original)
            paths = _stage_feedback_screenshots(staged, root / "stage")
            self.assertEqual(len(set(paths)), 2)
            for path in paths:
                self.assertEqual((root / "stage" / path).read_bytes(), source.read_bytes())
            self.assertEqual(
                original["cases"][0]["feedback"]["trajectory"][0]["image_records"][0]["path"],
                str(source),
            )

    def test_uniform_sample_has_case_and_step_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_evidence(root, 20)
            paths, context = collect_game_screenshots(root)
            self.assertEqual(len(paths), 8)
            self.assertEqual(Path(paths[0]).name, "step_0.png")
            self.assertEqual(Path(paths[-1]).name, "step_19.png")
            self.assertIn("case_id=case-a, step=19", context)
            self.assertEqual(len(set(paths)), 8)

    def test_configured_zero_one_and_invalid_limits(self):
        from rsi_methods.HyperAgents import visual_evidence
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_evidence(root, 3)
            config = root / "visual_evidence_limit.json"
            with patch.object(visual_evidence, "__file__", str(root / "visual_evidence.py")):
                config.write_text('{"max_refinement_images": 0}')
                self.assertEqual(collect_game_screenshots(root), ([], ""))
                config.write_text('{"max_refinement_images": 1}')
                self.assertEqual(Path(collect_game_screenshots(root)[0][0]).name, "step_2.png")
                config.write_text('{"max_refinement_images": -1}')
                with self.assertRaises(ValueError):
                    collect_game_screenshots(root)

    def test_missing_or_escaped_image_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.make_evidence(root)
            result["cases"][0]["feedback"]["trajectory"][0]["image_records"][0]["path"] = "../outside.png"
            (root / "result.json").write_text(json.dumps(result))
            with self.assertRaises(FileNotFoundError):
                collect_game_screenshots(root)

    def test_meta_agent_uses_one_request_with_raw_images(self):
        from meta_agent import MetaAgent
        instance = object.__new__(MetaAgent)
        for key, value in {
            "model": "test", "reasoning_effort": "low", "backend": "openai",
            "api_mode": "responses", "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL", "log": lambda *_: None,
        }.items():
            setattr(instance, key, value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_evidence(root)
            with patch("meta_agent.chat_with_agent", return_value=[]) as chat:
                instance.forward("/repo", str(root), iterations_left=19)
            chat.assert_called_once()
            self.assertTrue(chat.call_args.args[0].startswith(
                "Modify any part of the codebase at `/repo`."
            ))
            self.assertNotIn("Strategy surface", chat.call_args.args[0])
            self.assertEqual(len(chat.call_args.kwargs["image_paths"]), 1)
            self.assertEqual(chat.call_args.kwargs["tools_available"], "all")
            self.assertFalse((root / "screenshot_analysis.json").exists())
            (root / "result.json").write_text('{"cases": []}')
            with patch("meta_agent.chat_with_agent", return_value=[]) as chat:
                instance.forward("/repo", str(root))
            self.assertEqual(
                chat.call_args.args[0],
                f"Modify any part of the codebase at `/repo`.\n\nEvaluation records: {root}/result.json",
            )
            self.assertEqual(chat.call_args.kwargs["image_paths"], [])

    def test_images_persist_in_followup_history_for_both_api_modes(self):
        from agent import llm
        from types import SimpleNamespace
        for mode, expected_type in (("responses", "input_image"), ("chat_completions", "image_url")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / "step.png"
                image.write_bytes(b"\x89PNG\r\n\x1a\nsource")
                response = SimpleNamespace(
                    text="ok", status="completed", finish_reason="stop",
                    usage=None, tool_calls=[], reasoning=None,
                )
                dispatcher = SimpleNamespace()
                with patch.object(llm, "_dispatcher", return_value=dispatcher):
                    with patch.object(dispatcher, "distribute", return_value=response, create=True) as send:
                        _, history = llm.generate_response(
                            "look", model="test", api_mode=mode, image_paths=[str(image)]
                        )
                        llm.generate_response("tool result", model="test", api_mode=mode, msg_history=history)
                first = send.call_args_list[0].kwargs["messages"][0]["content"]
                second = send.call_args_list[1].kwargs["messages"][0]["content"]
                self.assertEqual(first, second)
                self.assertEqual(first[1]["type"], expected_type)

    def test_budget_telemetry_does_not_exit_before_runner_saves_patch(self):
        from rsi_methods.HyperAgents.lifecycle import _USAGE_RECORDING_BOOTSTRAP
        from types import SimpleNamespace
        history = [{
            "role": "assistant", "usage": {"total_tokens": 7},
            "tool_budget_exhausted": True,
            "tool_calls_executed": 40, "tool_call_limit": 40,
            "tool_protocol_error": {
                "type": "invalid_tool_call_format", "attempts": 3,
            },
        }]
        module = SimpleNamespace(chat_with_agent=lambda *_args, **_kwargs: history)
        with tempfile.TemporaryDirectory() as directory:
            usage_path = Path(directory) / "usage.json"

            def run_runner(*_args, **_kwargs):
                self.assertIs(module.chat_with_agent("edit"), history)

            with patch.dict(sys.modules, {"meta_agent": module}), \
                    patch.object(sys, "argv", ["python", str(usage_path)]), \
                    patch("runpy.run_path", side_effect=run_runner):
                exec(_USAGE_RECORDING_BOOTSTRAP, {})
            recorded = json.loads(usage_path.read_text())
            self.assertTrue(recorded["tool_budget_exhausted"])
            self.assertEqual(recorded["tool_protocol_error"],
                             history[0]["tool_protocol_error"])


if __name__ == "__main__":
    unittest.main()
