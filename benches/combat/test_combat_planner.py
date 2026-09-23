import copy
import unittest

from combat_planner import (
    CombatError,
    CombatSession,
    draw_hand,
    generate_task,
    load_combat_data,
    simulate_plan,
    solve_task,
)


class CombatPlannerTests(unittest.TestCase):
    def test_global_catalog_has_tiers_one_through_five(self):
        cards, enemies = load_combat_data()
        self.assertEqual({card.tier for card in cards.values()}, set(range(1, 6)))
        self.assertEqual({enemy.tier for enemy in enemies.values()}, set(range(1, 6)))
        for tier in range(1, 6):
            self.assertTrue(any(enemy.tier == tier and enemy.boss for enemy in enemies.values()))

    def test_generation_is_reproducible_and_verified(self):
        first = generate_task(1, seed=42, task_id="same")
        second = generate_task(1, seed=42, task_id="same")
        self.assertEqual(first, second)
        self.assertEqual(first["player"]["initial_gold"], 120)
        self.assertTrue(first["solvability"]["verified"])

    def test_difficulty_five_exposes_only_tiers_one_through_five(self):
        task = generate_task(5, seed=105, task_id="tier-five")
        self.assertEqual(
            {record["tier"] for record in task["card_catalog"].values()}, set(range(1, 6))
        )
        self.assertEqual(
            {record["tier"] for record in task["enemy_catalog"].values()}, set(range(1, 6))
        )
        self.assertTrue(all(record["tier"] <= 5 for record in task["card_catalog"].values()))

    def test_public_draw_order_distributes_copies_round_robin(self):
        hand = draw_hand({"strike": 2, "guard": 1}, ["guard", "strike"], 1, 5)
        self.assertEqual(hand, ["guard", "strike", "strike", "guard", "strike"])
        self.assertEqual(
            draw_hand({"strike": 2, "guard": 1}, ["guard", "strike"], 2, 3),
            ["guard", "strike", "strike"],
        )

    def test_reference_plan_replays_to_victory(self):
        task = generate_task(2, seed=12, task_id="reference")
        plan = solve_task(task)
        result = simulate_plan(task, plan)
        self.assertTrue(result["victory"])
        self.assertEqual(result["objective_score"], task["reference_score"])
        self.assertEqual(result["relative_score"], 1.0)

    def test_stage_session_reveals_only_current_state_and_updates_deck(self):
        task = generate_task(1, seed=12, task_id="staged")
        reference = solve_task(task)
        decisions = {stage["stage_id"]: stage for stage in reference["stages"]}
        session = CombatSession(task)

        combat_input = session.current_request()
        self.assertEqual(combat_input["decision_type"], "combat")
        self.assertNotIn("stages", combat_input)
        self.assertNotIn("enemy_catalog", combat_input)
        self.assertNotIn("reference_score", combat_input)
        self.assertEqual(len(combat_input["combat"]["draws"]), 15)
        session.apply(decisions[combat_input["stage_id"]])

        shop_input = session.current_request()
        self.assertEqual(shop_input["decision_type"], "shop")
        offered = shop_input["shop"]["offers"][0]
        before = session.deck.get(offered["id"], 0)
        session.apply(
            {
                "stage_id": shop_input["stage_id"],
                "buy_cards": [offered["id"]],
                "heal": False,
            }
        )
        next_combat = session.current_request()
        self.assertEqual(next_combat["deck"][offered["id"]], before + 1)
        self.assertIn(offered["id"], next_combat["combat"]["draw_cycle"])

    def test_stage_session_stops_after_a_lost_combat(self):
        task = generate_task(1, seed=8, task_id="stage-failure")
        session = CombatSession(task)
        session.apply({"stage_id": "combat_1", "turns": []})
        self.assertTrue(session.finished)
        self.assertEqual(session.result()["stages_reached"], 1)
        with self.assertRaises(CombatError):
            session.current_request()

        sequential_plan = {
            "task_id": task["task_id"],
            "mode": "sequential_stage_decisions",
            "stages": [{"stage_id": "combat_1", "turns": []}],
        }
        replay = simulate_plan(task, sequential_plan)
        self.assertFalse(replay["victory"])

    def test_completed_sequential_plan_replays(self):
        task = generate_task(2, seed=22, task_id="sequential-replay")
        reference = solve_task(task)
        plan = {
            "task_id": task["task_id"],
            "mode": "sequential_stage_decisions",
            "stages": reference["stages"],
        }
        result = simulate_plan(task, plan)
        self.assertTrue(result["victory"])
        self.assertEqual(result["objective_score"], task["reference_score"])

    def test_strategy_failure_is_scored_but_illegal_card_is_rejected(self):
        task = generate_task(1, seed=8, task_id="failure")
        blank_stages = []
        for stage in task["stages"]:
            if stage["type"] == "combat":
                blank_stages.append({"stage_id": stage["stage_id"], "turns": []})
            else:
                blank_stages.append(
                    {"stage_id": stage["stage_id"], "buy_cards": [], "heal": False}
                )
        blank_plan = {"task_id": task["task_id"], "stages": blank_stages}
        failed = simulate_plan(task, blank_plan)
        self.assertFalse(failed["victory"])
        self.assertEqual(
            failed["objective_score"],
            100 + 5 * failed["enemies_defeated"] + failed["final_hp"] + failed["final_gold"],
        )
        self.assertLess(failed["relative_score"], 1.0)

        illegal = copy.deepcopy(blank_plan)
        illegal["stages"][0]["turns"] = [{"turn": 1, "play": ["not_a_card"]}]
        with self.assertRaises(CombatError):
            simulate_plan(task, illegal)

    def test_solver_rejects_artificially_impossible_turn_limit(self):
        task = generate_task(1, seed=3, task_id="short")
        task["rules"]["max_combat_turns"] = 1
        with self.assertRaises(CombatError):
            solve_task(task)


if __name__ == "__main__":
    unittest.main()
