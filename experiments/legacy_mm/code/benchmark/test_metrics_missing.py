"""No missing measurements should be silently interpreted as zero."""

import unittest

from benchmark.metrics import calculate_metrics
from benchmark.runner import _suite_feedback


class MetricsMissingTests(unittest.TestCase):
    def test_error_does_not_contribute_zero_score_or_token_cost(self):
        successful = [
            {"run_id": str(i), "status": "ok", "relative_score": float(i),
             "total_tokens": 10, "total_seconds": 1, "replay": False}
            for i in range(1, 4)
        ]
        failed = {"run_id": "4", "status": "error", "relative_score": 0,
                  "total_tokens": None, "total_seconds": 1, "replay": False}
        metrics = calculate_metrics(successful + [failed], [], evolution_window=2,
                                    expected_training_runs=20)
        self.assertEqual(metrics["evolution_gain"]["first_window_average"], 1.5)
        self.assertIsNone(metrics["evolution_gain"]["last_window_average"])
        self.assertIsNone(metrics["evolution_cost"]["total_tokens"])
        self.assertIsNone(metrics["evolution_cost"]["tokens_per_absolute_gain"])
        self.assertEqual(metrics["stability"]["transition_count"], 2)

    def test_real_zero_is_valid_and_failure_is_not_suite_failure_score(self):
        records = [
            {"game": "gameworld", "difficulty": 1, "status": "ok", "relative_score": 0,
             "success": False},
            {"game": "gameworld", "difficulty": 1, "status": "error", "relative_score": 0,
             "success": False},
        ]
        summary = _suite_feedback(records)[0]
        self.assertEqual(summary["rule_count"], 2)
        self.assertEqual(summary["evaluated_count"], 1)
        self.assertEqual(summary["mean_score"], 0)
        self.assertEqual(summary["rule_success_rate"], 0)
        records[0]["status"] = "error"
        summary = _suite_feedback(records)[0]
        self.assertIsNone(summary["mean_score"])
        self.assertIsNone(summary["rule_success_rate"])


if __name__ == "__main__":
    unittest.main()
