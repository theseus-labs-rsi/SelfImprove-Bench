import unittest

from benchmark.contracts import BackendResponse
from benches.sweepmine.core import SweepmineCore, SweepmineEvent
from benches.sweepmine.play import extract_action, play_feedback, resolve_act


class SweepmineParsingTest(unittest.TestCase):
    def core(self):
        return SweepmineCore({
            "task_id": "format-test", "difficulty": 3, "seed": 7,
            "rows": 3, "cols": 3, "mine_count": 1, "max_steps": 5,
        })

    def test_plain_json_is_not_rewritten(self):
        raw = ' \n{"act": "mark", "row": 0, "col": 0}\n'
        self.assertEqual(extract_action(raw), raw)

    def test_complete_json_fences_are_unwrapped(self):
        action = '{"act":"mark","row":0,"col":0}'
        for prefix in ("```json\n", "```\n", "```JSON\r\n"):
            with self.subTest(prefix=prefix):
                raw = f" \n{prefix}{action}\n```\n "
                self.assertEqual(extract_action(raw), action)
                self.assertEqual(extract_action(extract_action(raw)), action)

    def test_prose_partial_fences_arrays_and_multiple_objects_are_not_unwrapped(self):
        action = '{"act":"mark","row":0,"col":0}'
        for raw in (
            f"Here is the action:\n```json\n{action}\n```",
            f"```json\n{action}\n```\nExtra explanation",
            f"```json\n{action}",
            f"```python\n{action}\n```",
            f"```json\n{action}\n{action}\n```",
            "```json\n[]\n```", "```json\ninvalid\n```",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(extract_action(raw), raw)

    def test_method_envelopes_are_not_interpreted(self):
        raw = '{"final_answer":{"act":"mark","row":0,"col":0}}'
        self.assertEqual(extract_action(raw), raw)
        core = self.core()
        self.assertEqual(resolve_act(raw, core), SweepmineEvent.ABNORMAL_ACTION)
        self.assertEqual(core.steps, 4)

    def test_fenced_and_plain_actions_have_identical_transitions(self):
        for action in (
            '{"act":"mark","row":0,"col":0}',
            '{"act":"click","row":-1,"col":0}',
            '{"act":"invalid","row":0,"col":0}',
        ):
            with self.subTest(action=action):
                plain, fenced = self.core(), self.core()
                event = resolve_act(action, plain)
                self.assertEqual(resolve_act(f"```json\n{action}\n```", fenced), event)
                self.assertEqual(fenced.inspect(), plain.inspect())
                self.assertEqual(fenced.steps, plain.steps)
                self.assertEqual(fenced.restart_count, plain.restart_count)

    def test_feedback_retains_raw_response(self):
        core = self.core()
        row, col = next(
            (r, c) for r in range(core.rows) for c in range(core.cols)
            if core.groundtruth[r][c] == -1
        )
        action = f'{{"act":"mark","row":{row},"col":{col}}}'
        raw = f"```json\n{action}\n```"

        class Agent:
            def act(self, prompt):
                return BackendResponse(text=raw)

        feedback = play_feedback(core, Agent())
        self.assertTrue(feedback.evaluation.success)
        self.assertEqual(feedback.trajectory[0].action, action)
        self.assertEqual(feedback.trajectory[0].agent_call.response_text, raw)


if __name__ == "__main__":
    unittest.main()
