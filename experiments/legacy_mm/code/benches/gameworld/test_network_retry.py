"""No-network regression tests for transient GameWorld action failures."""

import unittest
import json
import tempfile
from pathlib import Path
from benchmark.contracts import BackendResponse, TokenUsage
from copy import deepcopy
from types import SimpleNamespace

from benches.gameworld.network_retry import call_with_connection_retry, is_transient_connection_error
from rsi_methods.AHE.agent import compact_prior_screenshots


class RetryTests(unittest.TestCase):
    def test_docker_wrapped_proxy_error_retries_then_succeeds(self):
        calls = []
        delays = []
        def action():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError('TaskAgent failed with exit code 1: httpx2.ProxyError: 502 Bad Gateway')
            return 'success'
        self.assertEqual(call_with_connection_retry(action, sleep=delays.append), 'success')
        self.assertEqual(len(calls), 3)
        self.assertEqual(delays, [2., 4.])

    def test_bad_request_and_agent_bug_are_not_retried(self):
        class BadRequest(Exception):
            status_code = 400
        for exception in (BadRequest('Exceeded maximum number of images (50)'), ValueError('bad action')):
            calls = []
            with self.assertRaises(type(exception)):
                call_with_connection_retry(lambda: calls.append(1) or (_ for _ in ()).throw(exception), sleep=lambda _: None)
            self.assertEqual(len(calls), 1)

    def test_retry_attempt_ledger_unknown_cost_blocks_stateless_total(self):
        with tempfile.TemporaryDirectory() as td:
            audit = Path(td) / "attempts.jsonl"
            attempts = []
            def call():
                attempts.append(1)
                if len(attempts) == 1:
                    raise ConnectionError("temporary gateway")
                return BackendResponse(text="ok", usage=TokenUsage(total_tokens=9))
            result = call_with_connection_retry(call, sleep=lambda _: None, audit_path=audit,
                                                invalidate_usage_on_retry=True)
            self.assertIsNone(result.usage)
            self.assertEqual(result.metadata["token_coverage"], "unknown_prior_failed_attempts")
            lines = [json.loads(x) for x in audit.read_text().splitlines()]
            self.assertEqual([x["status"] for x in lines], ["failed_usage_unknown", "completed"])
            self.assertEqual(lines[-1]["usage"]["total_tokens"], 9)

    def test_retry_has_strict_limit(self):
        calls = []
        def action():
            calls.append(1)
            raise RuntimeError('ProxyError: 502 Bad Gateway')
        with self.assertRaises(RuntimeError):
            call_with_connection_retry(action, max_attempts=4, sleep=lambda _: None)
        self.assertEqual(len(calls), 4)
        self.assertFalse(is_transient_connection_error(RuntimeError('random 502 in model text')))


class FakeMessage:
    def __init__(self, metadata):
        self.metadata = metadata
    def model_copy(self, *, deep):
        assert deep
        return deepcopy(self)


class FakeHistory(list):
    flushed = 0
    def replace_all(self, new):
        self[:] = new
    def flush(self):
        self.flushed += 1


class ScreenshotHistoryTests(unittest.TestCase):
    def test_keep_only_recent_images_but_keep_all_text(self):
        msgs = []
        for i in range(55):
            msgs.append(FakeMessage({'response_items': [{
                'type': 'message', 'role': 'user', 'content': [
                    {'type': 'input_text', 'text': f'game step {i}'},
                    {'type': 'input_image', 'image_url': f'data:image/png;base64,{i}'},
                ]}]}))
        old = deepcopy(msgs[0]); history = FakeHistory(msgs)
        stripped = compact_prior_screenshots(SimpleNamespace(history=history), keep_recent=8)
        self.assertEqual(stripped, 47)
        self.assertEqual(history.flushed, 1)
        for i, msg in enumerate(history):
            content = msg.metadata['response_items'][0]['content']
            self.assertEqual(content[0]['text'], f'game step {i}')
            self.assertEqual(len(content), 1 if i < 47 else 2)
        self.assertEqual(old.metadata['response_items'][0]['content'], msgs[0].metadata['response_items'][0]['content'])
        self.assertEqual(compact_prior_screenshots(SimpleNamespace(history=history)), 0)


    def test_installed_nexau_response_converter_at_turn_60(self):
        try:
            from nexau.archs.main_sub.history_list import HistoryList
            from nexau.archs.main_sub.execution.llm_caller import _prepare_responses_api_input
            from nexau.core.adapters.legacy import messages_to_legacy_openai_chat
            from nexau.core.messages import Message
        except ImportError:
            self.skipTest('NexAU lives in the AHE Python 3.13 environment')
        history = HistoryList()
        for i in range(60):
            compact_prior_screenshots(SimpleNamespace(history=history))
            message = Message.user(f'game step {i}')
            message.metadata['response_items'] = [{
                'type': 'message', 'role': 'user', 'content': [
                    {'type': 'input_text', 'text': f'game step {i}'},
                    {'type': 'input_image', 'image_url': f'data:image/png;base64,{i}'},
                ]
            }]
            history.append(message)
            items, _ = _prepare_responses_api_input(messages_to_legacy_openai_chat(
                list(history), tool_image_policy='embed_in_tool_message',
            ))
            image_count = sum(
                part.get('type') == 'input_image'
                for item in items for part in item.get('content', []) if isinstance(part, dict)
            )
            self.assertLessEqual(image_count, 9)
        self.assertEqual(len(history), 60)
        self.assertEqual(image_count, 9)
        self.assertEqual(history[0].get_text_content(), 'game step 0')


if __name__ == '__main__':
    unittest.main()
