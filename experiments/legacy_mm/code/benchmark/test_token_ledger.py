"""No inferred zero for missing usage; no cumulative double-counting."""
import json
import sys
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path

from benchmark.rsi_session import refinement_token_metrics
from benchmark.token_ledger import hyperagents_usage_report, nexau_usage_snapshot
from rsi_methods.AHE.agent import AHEAgent
from rsi_methods.AHE.refine import _evolve_token_report


class _Connection:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.request = None

    def send(self, request):
        self.request = request

    def poll(self, timeout):
        return True

    def recv(self):
        reply = next(self.replies)
        folder = Path(self.request['call_dir'])
        if reply.get('snapshot'):
            (folder / 'nexau_usage_snapshot.json').write_text(json.dumps(reply['snapshot']))
        (folder / 'nexau_in_memory_tracer.cleaned.json').write_text('{}')
        if reply['ok']:
            return {'ok': True, 'text': '{}', 'trace_path': str(folder / 'nexau_in_memory_tracer.cleaned.json')}
        return {'ok': False, 'error': 'proxy refused', 'status_code': 502}


class LedgerTests(unittest.TestCase):
    def test_ahe_evolution_reports_provider_subfields_without_estimates(self):
        entries = [{"candidate_id": "a", "usage": {
            "input_tokens": 7, "output_tokens": 5,
            "reasoning_tokens": 2, "total_tokens": 12}},
            {"candidate_id": "b", "usage": {
                "input_tokens": 11, "output_tokens": 6,
                "reasoning_tokens": 3, "total_tokens": 17}}]
        report = _evolve_token_report(entries, debugger_invoked=False,
                                       source_path=Path("usage.json"))
        metrics = refinement_token_metrics({"token_usage_report": report})
        self.assertEqual((metrics["prompt_tokens"], metrics["reasoning_tokens"],
                          metrics["answer_tokens"], metrics["total_tokens"]), (18, 5, 6, 29))
        self.assertEqual(metrics["evidence_path"], "usage.json")
        entries[1]["usage"].pop("reasoning_tokens")
        metrics = refinement_token_metrics({"token_usage_report": _evolve_token_report(
            entries, debugger_invoked=False, source_path=Path("usage.json"))})
        self.assertIsNone(metrics["reasoning_tokens"])
        self.assertEqual(metrics["total_tokens"], 29)
        self.assertIsNone(_evolve_token_report(entries, debugger_invoked=True,
                            source_path=Path("usage.json"))["total_tokens"])

    def test_ahe_cumulative_delta_includes_retry_then_no_double_count(self):
        with tempfile.TemporaryDirectory() as td:
            agent = object.__new__(AHEAgent)
            agent.artifact_dir = Path(td)
            agent.timeout_seconds = 1
            agent.model = 'test'
            agent._last_usage_snapshot = None
            agent._attempt_index = 0
            agent._unreconciled_failed_attempts = False
            agent.trace_paths = []
            agent._connection = _Connection([
                {'ok': True, 'snapshot': {'trace_id': 'trace', 'total_tokens': 20}},
                {'ok': False, 'snapshot': {'trace_id': 'trace', 'total_tokens': 27}},
                {'ok': True, 'snapshot': {'trace_id': 'trace', 'total_tokens': 35}},
                {'ok': True, 'snapshot': {'trace_id': 'trace', 'total_tokens': 40}},
            ])
            self.assertEqual(agent.act('1').usage.total_tokens, 20)
            with self.assertRaises(ConnectionError):
                agent.act('failed')
            self.assertIsNone(agent.act('retry').usage)
            self.assertEqual(agent.act('next').usage.total_tokens, 5)
            records = [json.loads(x) for x in (Path(td) / 'token_attempts.jsonl').read_text().splitlines()]
            self.assertEqual([r['billed_tokens'] for r in records], [20, None, 15, 5])
            self.assertEqual(records[2]['status'], 'usage_unknown')

    def test_ahe_nonmonotonic_or_unknown_cannot_be_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as td:
            agent = object.__new__(AHEAgent)
            agent.artifact_dir = Path(td)
            agent.timeout_seconds = 1
            agent.model = 'test'
            agent._last_usage_snapshot = {'trace_id': 'trace', 'total_tokens': 200}
            agent._attempt_index = 0
            agent._unreconciled_failed_attempts = False
            agent.trace_paths = []
            agent._connection = _Connection([{'ok': True, 'snapshot': {'trace_id': 'trace', 'total_tokens': 5}}])
            self.assertIsNone(agent.act('reset').usage)

    def test_ahe_one_span_without_usage_does_not_poison_later_steps(self):
        def snap(spans):
            calls = [{'span_id': f's{i}', 'total_tokens': t, 'input_tokens': t, 'output_tokens': 0,
                      'cached_input_tokens': 0, 'reasoning_tokens': 0} for i, t in spans]
            known = [c['total_tokens'] for c in calls]
            return {'trace_id': 'trace', 'calls': calls,
                    'total_tokens': sum(known) if None not in known else None}
        steps = [[(0, 10)], [(0, 10), (1, None)], [(0, 10), (1, None), (2, 30)],
                 [(0, 10), (1, None), (2, 30), (3, 40)]]
        with tempfile.TemporaryDirectory() as td:
            agent = object.__new__(AHEAgent)
            agent.artifact_dir = Path(td)
            agent.timeout_seconds = 1
            agent.model = 'test'
            agent._last_usage_snapshot = None
            agent._attempt_index = 0
            agent._unreconciled_failed_attempts = False
            agent.trace_paths = []
            agent._connection = _Connection([{'ok': True, 'snapshot': snap(x)} for x in steps])
            usages = [agent.act(str(i)).usage for i in range(4)]
            self.assertEqual([u.total_tokens if u else None for u in usages], [10, None, 30, 40])
            records = [json.loads(x) for x in (Path(td) / 'token_attempts.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in records],
                             ['completed', 'usage_unknown', 'completed', 'completed'])
            self.assertEqual(records[1]['missing_usage_spans'], ['s1'])

    def test_provider_span_missing_usage_invalidates_snapshot(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rsi_methods' / 'AHE' / 'agentic-harness-engineering'))
        def raw(usage):
            return [{'id': 'root', 'name': 'agent', 'children': [
                {'id': 'c1', 'name': 'openai chat', 'outputs': {'output': [{'type': 'message', 'content': 'ok'}],
                                                                'usage': {'total_tokens': usage} if usage is not None else {}}}]}]
        self.assertEqual(nexau_usage_snapshot(raw(123))['total_tokens'], 123)
        self.assertIsNone(nexau_usage_snapshot(raw(None))['total_tokens'])
        self.assertEqual(nexau_usage_snapshot(raw(123) + raw(44))['trace_id'], 'root')

    def test_hyper_task_empty_answer_retry_counts_both_model_calls(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rsi_methods' / 'HyperAgents' / 'hyperagents'))
        import task_agent
        agent = object.__new__(task_agent.TaskAgent)
        for name in ('model', 'reasoning_effort', 'backend', 'api_mode', 'api_key_env', 'base_url_env'):
            setattr(agent, name, 'test')
        agent.log = lambda *_: None
        responses = [
            [{'role': 'assistant', 'text': '', 'usage': {'total_tokens': 2, 'input_tokens': 1, 'output_tokens': 1}}],
            [{'role': 'assistant', 'text': 'action', 'usage': {'total_tokens': 3, 'input_tokens': 2, 'output_tokens': 1}}],
        ]
        with patch.object(task_agent, 'chat_with_agent', side_effect=responses):
            prediction, history = agent.forward({'instruction': 'act'})
        self.assertEqual(prediction, 'action')
        self.assertEqual(history[-1]['usage']['total_tokens'], 5)
        self.assertEqual(len(history[-1]['usage_attempts']), 2)

    def test_hyper_missing_attempt_is_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            history = folder / 'meta_agent_history.json'
            history.write_text('[{"role":"assistant","text":"ok"}]')
            ledger = folder / 'model_attempts.jsonl'
            ledger.write_text('\n'.join(json.dumps({'status': 'completed', 'usage': {'total_tokens': n,
                'input_tokens': n-1, 'output_tokens': 1}}) for n in [3, 4]) + '\n')
            report = hyperagents_usage_report(history)
            self.assertEqual(refinement_token_metrics({'token_usage_report': report})['total_tokens'], 7)
            with ledger.open('a') as stream:
                stream.write(json.dumps({'status': 'failed_unknown', 'usage': None}) + '\n')
            self.assertIsNone(refinement_token_metrics({'token_usage_report': hyperagents_usage_report(history)})['total_tokens'])

    def test_ahe_evolution_unknown_debugger_blocks_total(self):
        self.assertIsNone(refinement_token_metrics({'token_usage_report': {
            'complete': False, 'debugger_invoked': True, 'total_tokens': 42}})['total_tokens'])


if __name__ == '__main__':
    unittest.main()
