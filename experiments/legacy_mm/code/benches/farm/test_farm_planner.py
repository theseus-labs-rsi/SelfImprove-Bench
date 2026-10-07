import unittest

from farm_planner import FarmError, calculate_reference_profit, generate_task, load_crops, simulate_plan


class FarmPlannerTests(unittest.TestCase):
    def test_generation_is_reproducible(self):
        a = generate_task(2, seed=42, task_id="x")
        b = generate_task(2, seed=42, task_id="x")
        self.assertEqual(a, b)
        self.assertEqual(len(a["available_crops"]), 6)
        self.assertEqual(a["initial_money"], 200)
        self.assertEqual(a["farmland_count"], 12)
        self.assertEqual(a["duration_days"], 12)
        self.assertIsInstance(a["reference_profit"], int)
        self.assertEqual(a["reference_method"], "greedy_daily_profit_lookahead")
        self.assertEqual(set(a["reference_daily_profits"]), set(a["available_crops"]))

    def test_simulation_profit_and_harvest_timing(self):
        task = generate_task(1, seed=1, task_id="t")
        crop = task["available_crops"][0]
        crop_info = load_crops()[crop]
        plan = {"task_id": "t", "days": [
            {"day": 1, "buy_seeds": [{"crop": crop, "quantity": 1}], "plant": [{"crop": crop, "quantity": 1}]},
            {"day": 2, "harvest": [{"crop": crop, "quantity": 1}], "sell": []},
        ]}
        with self.assertRaises(FarmError):
            simulate_plan(task, plan)
        maturity = crop_info.maturity_days
        result = simulate_plan(task, {"task_id": "t", "days": [
            {"day": 1, "buy_seeds": [{"crop": crop, "quantity": 1}], "plant": [{"crop": crop, "quantity": 1}]},
            {"day": 1 + maturity, "harvest": [{"crop": crop, "quantity": 1}], "sell": [{"crop": crop, "quantity": crop_info.yield_per_harvest}]},
        ]})
        self.assertEqual(result["profit"], crop_info.yield_per_harvest * crop_info.fruit_price - crop_info.seed_price)
        self.assertEqual(result["unsold_fruits"], {})
        self.assertEqual(result["score"], result["profit"] / result["reference_profit"])

    def test_reference_profit_and_score(self):
        task = {
            "task_id": "reference",
            "initial_money": 100,
            "farmland_count": 1,
            "duration_days": 3,
            "available_crops": ["fast"],
            "crop_catalog": {
                "fast": {
                    "id": "fast", "name": "Fast", "seed_price": 10,
                    "maturity_days": 1, "yield_per_harvest": 1, "fruit_price": 20,
                }
            },
        }
        self.assertEqual(calculate_reference_profit(task), 20)
        result = simulate_plan(task, {"task_id": "reference", "days": []})
        self.assertEqual(result["reference_profit"], 20)
        self.assertEqual(result["score"], 0.0)

    def test_reference_lookahead_reinvests_after_short_cycle(self):
        task = {
            "task_id": "lookahead",
            "initial_money": 200,
            "farmland_count": 16,
            "duration_days": 16,
            "available_crops": ["green", "tulip", "blue"],
            "crop_catalog": {
                "green": {"id": "green", "name": "Green", "seed_price": 200,
                           "maturity_days": 10, "yield_per_harvest": 8, "fruit_price": 40},
                "tulip": {"id": "tulip", "name": "Tulip", "seed_price": 20,
                           "maturity_days": 6, "yield_per_harvest": 2, "fruit_price": 30},
                "blue": {"id": "blue", "name": "Blue", "seed_price": 30,
                         "maturity_days": 7, "yield_per_harvest": 2, "fruit_price": 40},
            },
        }
        # The lookahead reference chooses tulips first, then blue flowers,
        # yielding 1,200 instead of the myopic green-bean result of 120.
        self.assertEqual(calculate_reference_profit(task), 1200)

    def test_zero_reference_score_is_defined(self):
        task = {
            "task_id": "zero-reference",
            "initial_money": 0,
            "farmland_count": 1,
            "duration_days": 1,
            "available_crops": ["slow"],
            "crop_catalog": {
                "slow": {
                    "id": "slow", "name": "Slow", "seed_price": 10,
                    "maturity_days": 2, "yield_per_harvest": 1, "fruit_price": 20,
                }
            },
        }
        result = simulate_plan(task, {"task_id": "zero-reference", "days": []})
        self.assertEqual(result["reference_profit"], 0)
        self.assertEqual(result["score"], 1.0)

    def test_rejects_overspending(self):
        task = generate_task(1, seed=2)
        crop = task["available_crops"][0]
        with self.assertRaises(FarmError):
            simulate_plan(task, {"days": [{"day": 1, "buy_seeds": [{"crop": crop, "quantity": 999}]}]})

    def test_zero_difficulty_is_valid(self):
        task = generate_task(0, seed=9, task_id="zero")
        self.assertEqual(task["initial_money"], 200)
        result = simulate_plan(task, {"task_id": "zero", "days": []})
        self.assertEqual(result["profit"], 0)

    def test_simulation_uses_task_crop_catalog_snapshot(self):
        task = generate_task(1, seed=1, task_id="snapshot")
        crop = task["available_crops"][0]
        info = task["crop_catalog"][crop]
        info["seed_price"] = 1
        info["yield_per_harvest"] = 10
        info["fruit_price"] = 2
        plan = {"task_id": "snapshot", "days": [
            {"day": 1, "buy_seeds": [{"crop": crop, "quantity": 1}], "plant": [{"crop": crop, "quantity": 1}]},
            {"day": 1 + info["maturity_days"], "harvest": [{"crop": crop, "quantity": 1}], "sell": [{"crop": crop, "quantity": 10}]},
        ]}
        result = simulate_plan(task, plan)
        self.assertEqual(result["profit"], 19)

    def test_rejects_malformed_task_crop_catalog(self):
        task = generate_task(1, seed=1)
        task["crop_catalog"][task["available_crops"][0]]["maturity_days"] = 0
        with self.assertRaises(FarmError):
            simulate_plan(task, {"days": []})


if __name__ == "__main__":
    unittest.main()
