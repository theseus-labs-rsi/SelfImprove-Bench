"""Offline regressions for the HyperAgents refinement objective."""

import unittest

from benchmark.contracts import ExecutionStatus, FeedbackResult, TaskEvaluation
from rsi_methods.HyperAgents.HyperAgentsAdaptor import HyperAgentsAdaptor


def feedback(score, *, success=False, status=ExecutionStatus.COMPLETED):
    return FeedbackResult(
        task_id="synthetic",
        task={},
        trajectory=(),
        evaluation=TaskEvaluation(
            execution_status=status, success=success, score=score,
        ),
    )


class HyperAgentsFeedbackTest(unittest.TestCase):
    def test_same_success_different_scores_remain_distinguishable(self):
        for success in (False, True):
            for score in (0.2, 0.9):
                with self.subTest(success=success, score=score):
                    result = HyperAgentsAdaptor._feedback_to_result([
                        feedback(score, success=success),
                    ])
                    self.assertTrue(result["valid"])
                    self.assertEqual(result["selection_score"], score)
                    self.assertEqual(result["metrics"]["score_mean"], score)
                    self.assertEqual(result["metrics"]["success_rate"], float(success))

    def test_batch_uses_mean_score_not_success_rate(self):
        result = HyperAgentsAdaptor._feedback_to_result([
            feedback(0.0), feedback(0.4), feedback(0.8, success=True),
        ])
        self.assertAlmostEqual(result["selection_score"], 0.4)
        self.assertEqual(result["selection_score"], result["metrics"]["score_mean"])
        self.assertAlmostEqual(result["metrics"]["success_rate"], 1 / 3)

    def test_zero_score_is_valid(self):
        result = HyperAgentsAdaptor._feedback_to_result([feedback(0.0)])
        self.assertTrue(result["valid"])
        self.assertEqual(result["selection_score"], 0.0)

    def test_incomplete_execution_remains_invalid(self):
        for status in (ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT,
                       ExecutionStatus.CANCELLED):
            with self.subTest(status=status):
                result = HyperAgentsAdaptor._feedback_to_result([
                    feedback(0.8, success=True), feedback(0.0, status=status),
                ])
                self.assertFalse(result["valid"])
                self.assertIsNone(result["selection_score"])
                self.assertEqual(result["metrics"]["score_mean"], 0.4)
                self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    unittest.main()
