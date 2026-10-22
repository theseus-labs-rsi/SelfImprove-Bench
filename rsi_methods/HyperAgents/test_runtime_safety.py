"""Offline regressions for the HyperAgents budget-safe runtime overlay."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "hyperagents"))


def load_overlay(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "runtime_overlay" / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeSafetyTests(unittest.TestCase):
    def test_text_protocol_executes_one_tool_and_returns_text_feedback(self):
        llm = load_overlay("test_restored_protocol", "agent/llm_withtools.py")
        usage = SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2,
                                cached_input_tokens=0, reasoning_tokens=0)
        executed = []
        tool = {
            "info": {"name": "bash", "input_schema": {"type": "object"}},
            "function": lambda command: executed.append(command) or "file contents",
        }
        tool_text = (
            '<json>{"tool_name":"bash","tool_input":{"command":"first"}}</json>'
            '<json>{"tool_name":"bash","tool_input":{"command":"second"}}</json>'
        )
        responses = [
            (SimpleNamespace(text=tool_text, usage=usage),
             [{"role": "assistant", "text": tool_text}]),
            (SimpleNamespace(text="done", usage=usage),
             [{"role": "assistant", "text": "done"}]),
        ]
        with patch.object(llm, "load_tools", return_value=[tool]), \
                patch.object(llm, "generate_response", side_effect=responses) as send, \
                patch.dict("os.environ", {"HYPERAGENTS_META_MAX_TOKENS": "0"}):
            llm.chat_with_agent("edit", tools_available="all", image_paths=["step.png"],
                                logging=lambda *_: None)
        self.assertEqual(executed, ["first"])
        self.assertIn("ONLY USE ONE TOOL PER RESPONSE", send.call_args_list[0].args[0])
        self.assertIn("file contents", send.call_args_list[1].args[0])
        self.assertNotIn("tools", send.call_args_list[0].kwargs)
        self.assertEqual(send.call_args_list[0].kwargs["image_paths"], ["step.png"])
        self.assertIsNone(send.call_args_list[1].kwargs["image_paths"])

    def test_bare_json_tool_calls_are_recovered(self):
        llm = load_overlay("test_bare_json_protocol", "agent/llm_withtools.py")
        usage = SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2,
                                cached_input_tokens=0, reasoning_tokens=0)
        executed = []
        tool = {
            "info": {"name": "bash", "input_schema": {"type": "object"}},
            "function": lambda command: executed.append(command) or "file contents",
        }
        tool_text = (
            '{"tool_name":"bash","tool_input":{"command":"first"}}'
            '{"tool_name":"bash","tool_input":{"command":"second"}}'
        )
        responses = [
            (SimpleNamespace(text=tool_text, usage=usage),
             [{"role": "assistant", "text": tool_text}]),
            (SimpleNamespace(text="done", usage=usage),
             [{"role": "assistant", "text": "done"}]),
        ]
        with patch.object(llm, "load_tools", return_value=[tool]), \
                patch.object(llm, "generate_response", side_effect=responses), \
                patch.dict("os.environ", {"HYPERAGENTS_META_MAX_TOKENS": "0"}):
            llm.chat_with_agent("edit", tools_available="all", logging=lambda *_: None)
        self.assertEqual(executed, ["first"])

    def test_bash_heredoc_without_trailing_newline(self):
        bash = load_overlay("test_safe_bash_heredoc", "agent/tools/bash.py")

        async def run():
            shell = bash.BashSession()
            await shell.start()
            try:
                return await asyncio.wait_for(shell.run("python3 - <<'PY'\nprint('ok')\nPY"), 8)
            finally:
                await shell.stop()

        output, error = asyncio.run(run())
        self.assertEqual(output, "ok")
        self.assertEqual(error, "")

    def test_bash_large_stdout_stderr_does_not_deadlock(self):
        bash = load_overlay("test_safe_bash_large", "agent/tools/bash.py")

        async def run(command):
            shell = bash.BashSession()
            await shell.start()
            try:
                return await asyncio.wait_for(shell.run(command), 8)
            finally:
                await shell.stop()

        output, _ = asyncio.run(run('python3 -c \'print("x"*1800000)\''))
        self.assertIn("tool output truncated", output)
        output, _ = asyncio.run(run('python3 -c \'import sys; sys.stderr.write("e"*1800000); print("ok")\''))
        self.assertIn("ok", output)

    def test_tool_cap_preserves_upstream_final_response(self):
        llm = load_overlay("test_safe_llm", "agent/llm_withtools.py")
        calls = []
        logs = []
        tool = {"info": {"name": "bash", "description": "", "input_schema": {"type": "object"}},
                "function": lambda **kwargs: "edited"}
        usage = SimpleNamespace(input_tokens=10, output_tokens=2, total_tokens=12,
                                cached_input_tokens=0, reasoning_tokens=1)

        def respond(*args, **kwargs):
            calls.append(1)
            response = SimpleNamespace(
                text='<json>{"tool_name":"bash","tool_input":{}}</json>', usage=usage,
            )
            return response, [{"role": "assistant", "text": "", "content": ""}]

        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "attempts.jsonl"
            with patch.object(llm, "generate_response", side_effect=respond), \
                 patch.object(llm, "load_tools", return_value=[tool]), \
                 patch.dict("os.environ", {"HYPERAGENTS_TOKEN_LEDGER": str(ledger)}):
                history = llm.chat_with_agent("edit", tools_available=["bash"],
                                               logging=logs.append, max_tool_calls=2)
            records = [json.loads(line) for line in ledger.read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(records), 3)
        self.assertEqual(history[-1]["usage"]["total_tokens"], 36)
        self.assertTrue(any("saving the current candidate" in line for line in logs))

    def test_meta_cap_across_visual_and_edit_calls_with_live_ledger(self):
        llm = load_overlay("test_safe_llm_budget", "agent/llm_withtools.py")
        count = []
        usage = SimpleNamespace(input_tokens=12, output_tokens=2, total_tokens=14,
                                cached_input_tokens=0, reasoning_tokens=0)
        def respond(*args, **kwargs):
            count.append(1)
            response = SimpleNamespace(
                text='<json>{"tool_name":"bash","tool_input":{}}</json>', usage=usage,
            )
            return response, [{"role": "assistant", "content": "done"}]
        tool = {"info": {"name": "bash", "input_schema": {"type": "object"}},
                "function": lambda **kwargs: "edited"}
        with tempfile.TemporaryDirectory() as directory:
            ledger, live = Path(directory) / "model_attempts.jsonl", Path(directory) / "live.jsonl"
            with patch.object(llm, "generate_response", side_effect=respond), \
                 patch.object(llm, "load_tools", return_value=[tool]), \
                 patch.dict("os.environ", {"HYPERAGENTS_TOKEN_LEDGER": str(ledger),
                                        "HYPERAGENTS_META_LIVE_LEDGER": str(live),
                                        "HYPERAGENTS_META_MAX_TOKENS": "28"}):
                llm.chat_with_agent("visual", tools_available=["bash"], logging=lambda *a: None,
                                    max_tool_calls=40)
                with self.assertRaisesRegex(RuntimeError, "cap exhausted"):
                    llm.chat_with_agent("extra", tools_available=["bash"], logging=lambda *a: None)
            self.assertEqual(len(count), 2)
            self.assertEqual(ledger.read_bytes(), live.read_bytes())

    def test_stage_runtime_overlay_is_idempotent_and_source_checked(self):
        from rsi_methods.HyperAgents.runtime_safety import stage_runtime_safety
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("agent/llm_withtools.py", "agent/tools/bash.py"):
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / "hyperagents" / relative, dest)
            stage_runtime_safety(root)
            stage_runtime_safety(root)
            self.assertEqual((root / "agent/llm_withtools.py").read_bytes(),
                             (ROOT / "runtime_overlay/agent/llm_withtools.py").read_bytes())
            (root / "agent/llm_withtools.py").write_text("unknown evolved runtime\n")
            with self.assertRaisesRegex(RuntimeError, "Unexpected HyperAgents runtime"):
                stage_runtime_safety(root)


if __name__ == "__main__":
    unittest.main()
