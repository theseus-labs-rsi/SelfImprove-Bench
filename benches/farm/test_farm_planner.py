import json
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
        self.assertNotIn("daily_operation_order", a["rules"])
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
            "initial_money": 10,
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
        plan = {"task_id": "reference", "days": [
            {"day": 1, "buy_seeds": [{"crop": "fast", "quantity": 1}],
             "plant": [{"crop": "fast", "quantity": 1}]},
            {"day": 2, "harvest": [{"crop": "fast", "quantity": 1}],
             "sell": [{"crop": "fast", "quantity": 1}],
             "buy_seeds": [{"crop": "fast", "quantity": 1}],
             "plant": [{"crop": "fast", "quantity": 1}]},
            {"day": 3, "harvest": [{"crop": "fast", "quantity": 1}],
             "sell": [{"crop": "fast", "quantity": 1}]},
        ]}
        result = simulate_plan(task, plan)
        self.assertEqual(result["profit"], result["reference_profit"])
        self.assertEqual(result["relative_score"], 1.0)

    def test_daily_actions_follow_json_field_order(self):
        task = {
            "task_id": "ordered", "initial_money": 2, "farmland_count": 2,
            "duration_days": 3, "available_crops": ["fast"],
            "crop_catalog": {
                "fast": {"id": "fast", "name": "Fast", "seed_price": 1,
                         "maturity_days": 1, "yield_per_harvest": 1, "fruit_price": 2}
            },
        }
        first_day = {"day": 1, "buy_seeds": [{"crop": "fast", "quantity": 1}],
                     "plant": [{"crop": "fast", "quantity": 1}]}
        second_day = {
            "day": 2,
            "buy_seeds": [{"crop": "fast", "quantity": 1}],
            "plant": [{"crop": "fast", "quantity": 1}],
            "harvest": [{"crop": "fast", "quantity": 1}],
            "sell": [{"crop": "fast", "quantity": 1}],
        }
        plan = json.loads(json.dumps({"days": [first_day, second_day]}))
        result = simulate_plan(task, plan)
        self.assertEqual(result["final_money"], 2)
        self.assertEqual(sum(batch["quantity"] for batch in result["growing_crops"]), 1)
        with self.assertRaisesRegex(FarmError, "Not enough farmland"):
            simulate_plan({**task, "farmland_count": 1}, plan)

        with self.assertRaisesRegex(FarmError, "Not enough fast fruit"):
            simulate_plan(task, {"days": [first_day, {
                "day": 2, "sell": [{"crop": "fast", "quantity": 1}],
                "harvest": [{"crop": "fast", "quantity": 1}],
            }]})
        with self.assertRaisesRegex(FarmError, "Not enough fast seeds"):
            simulate_plan(task, {"days": [{
                "day": 1, "plant": [{"crop": "fast", "quantity": 1}],
                "buy_seeds": [{"crop": "fast", "quantity": 1}],
            }]})

    def test_buying_before_sale_cannot_spend_future_proceeds(self):
        task = {
            "task_id": "cash-flow", "initial_money": 1, "farmland_count": 1,
            "duration_days": 3, "available_crops": ["fast"],
            "crop_catalog": {
                "fast": {"id": "fast", "name": "Fast", "seed_price": 1,
                         "maturity_days": 1, "yield_per_harvest": 1, "fruit_price": 2}
            },
        }
        first_day = {"day": 1, "buy_seeds": [{"crop": "fast", "quantity": 1}],
                     "plant": [{"crop": "fast", "quantity": 1}]}
        second_day = {"day": 2, "buy_seeds": [{"crop": "fast", "quantity": 1}],
                      "harvest": [{"crop": "fast", "quantity": 1}],
                      "sell": [{"crop": "fast", "quantity": 1}],
                      "plant": [{"crop": "fast", "quantity": 1}]}
        with self.assertRaisesRegex(FarmError, "Insufficient funds"):
            simulate_plan(task, {"days": [first_day, second_day]})
        second_day = {"day": 2, "harvest": second_day["harvest"],
                      "sell": second_day["sell"], "buy_seeds": second_day["buy_seeds"],
                      "plant": second_day["plant"]}
        result = simulate_plan(task, {"days": [first_day, second_day]})
        self.assertEqual(result["final_money"], 1)

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

    def test_rejects_overplanting_across_batches(self):
        task = {
            "task_id": "capacity", "initial_money": 10, "farmland_count": 2,
            "duration_days": 3, "available_crops": ["fast"],
            "crop_catalog": {
                "fast": {"id": "fast", "name": "Fast", "seed_price": 1,
                         "maturity_days": 1, "yield_per_harvest": 1, "fruit_price": 2}
            },
        }
        day_one = {"day": 1, "buy_seeds": [{"crop": "fast", "quantity": 3}],
                   "plant": [{"crop": "fast", "quantity": 2}]}
        with self.assertRaises(FarmError):
            simulate_plan(task, {"days": [
                {**day_one, "plant": [*day_one["plant"], {"crop": "fast", "quantity": 1}]}
            ]})
        result = simulate_plan(task, {"days": [
            day_one,
            {"day": 2, "harvest": [{"crop": "fast", "quantity": 1}],
             "sell": [{"crop": "fast", "quantity": 1}],
             "plant": [{"crop": "fast", "quantity": 1}]},
        ]})
        self.assertEqual(sum(batch["quantity"] for batch in result["growing_crops"]), 2)

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
